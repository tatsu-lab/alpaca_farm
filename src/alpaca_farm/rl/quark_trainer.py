# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward conditioning a la QUARK.

For all-quantiles formulation, during decoding, each instance takes the following form (except first decoding stage):
    <bos_token><reward_cond_token><query><response><eos_token>
E.g.,
    <s><reward_0>Tell me something about alpacas.Alpacas are cute.</s>
"""

import contextlib
import os
from typing import Callable, Dict, Sequence, Tuple

import accelerate
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
import transformers
from torch import nn
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.utils.data import DataLoader
from transformers.modeling_utils import unwrap_model

from .. import accelerate_patch, common, constants, data_preprocessor, data_utils, logging, utils
from ..models import reward_model as reward_model_module
from ..models import rl_models
from ..types import AnyPath, AnyPathOrNone, LRScheduler, Optional, Tensor
from . import kl_controller, rl_trainer

FIRST_STEP_IDX = 1

logger = logging.get_logger(__name__)


def ignore_tokens(input_ids: Tensor, attention_mask: Tensor, tokens_to_ignore: Sequence[int]):
    """Clear out positions where input_ids has tokens_to_ignore in attention_mask."""
    attention_mask = attention_mask.clone()
    for token_to_ignore in tokens_to_ignore:
        attention_mask[input_ids == token_to_ignore] = 0
    return input_ids, attention_mask


class DataPool(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.additional_special_tokens = tokenizer.additional_special_tokens

        self.queries = []
        self.responses = []
        self.rewards = []

    def add(self, queries, responses, rewards):
        for main_list, this_list in utils.zip_(
            (self.queries, self.responses, self.rewards), (queries, responses, rewards)
        ):
            main_list.extend(this_list)

    def clear(self):
        (self.queries, self.responses, self.rewards) = [], [], []

    def sort_and_get(self, train_on_best_quantile=True):
        queries, responses, rewards = utils.parallel_sort(
            self.queries,
            self.responses,
            self.rewards,
            key=lambda x: x[-1],
            reverse=True,
        )

        size = len(queries)
        chunk_sizes = [size // len(self.additional_special_tokens)] * len(self.additional_special_tokens)
        chunk_sizes[-1] = chunk_sizes[-1] + size % len(self.additional_special_tokens)
        assert sum(chunk_sizes) == size, "Internal error: Sum of chunk sizes doesn't match up with total size."

        if train_on_best_quantile:  # Don't inject any tokens here.
            queries, responses, rewards = tuple(l[: chunk_sizes[0]] for l in (queries, responses, rewards))
        else:
            injected_tokens = []
            for chunk_index, chunk_size in enumerate(chunk_sizes):
                injected_tokens.extend([self.additional_special_tokens[chunk_index]] * chunk_size)
            queries = [f"{injected_token}{query}" for injected_token, query in utils.zip_(injected_tokens, queries)]
        return queries, responses, rewards


class QuarkTrainer(rl_trainer.RLTrainer):
    def __init__(
        self,
        args,
        train_dataset: data_utils.QueryDataset,
        eval_dataset: data_utils.QueryDataset,
        data_collator: Callable,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        accelerator: accelerate_patch.MyAccelerator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super().__init__(
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        self.data_pool = DataPool(self.tokenizer)
        self.entropy_ctl = kl_controller.FixedKLController(kl_coef=args.entropy_coef)
        self.sft_dataloader = None  # Must be instantiated in `rollout`.

    def train(self):
        total_epochs = self.args.total_epochs
        total_episodes = len(self.train_dataset) * total_epochs  # noqa
        total_steps = total_episodes // self.args.rollout_batch_size  # noqa
        logger.warning(
            f"***Training starts***\n"
            f"Total epochs: {total_epochs} => Total episodes: {total_episodes} => Total steps: {total_steps}",
        )

        self.create_optimizer_and_scheduler(total_steps)
        infinite_train_dataloader = self.get_train_dataloader()
        for step_idx in tqdm.tqdm(
            range(FIRST_STEP_IDX, total_steps + FIRST_STEP_IDX),
            disable=not self.accelerator.is_main_process,
            desc="steps",
            total=total_steps,
        ):
            if step_idx % self.args.save_steps == 0 or step_idx in self.args.save_steps_extra_list:
                self.save_model(utils.join(self.args.output_dir, f"checkpoint-{step_idx}"))
            if self.args.eval_steps is not None and step_idx % self.args.eval_steps == 0:
                unwrapped_policy = self.accelerator.unwrap_model(self.policy, keep_fp32_wrapper=True)
                unwrapped_policy = unwrapped_policy.base_model
                self.evaluate(step_idx, unwrapped_policy=unwrapped_policy)
            self.log_history.append(self.step(infinite_train_dataloader, step_idx))
        return self.log_history

    def step(self, train_dataloader, step_idx, **kwargs):
        rollouts_dataloader = self.rollout(train_dataloader, step_idx)

        stats_list = []
        for _ in tqdm.tqdm(
            range(self.args.num_gradient_steps_per_step), disable=not self.accelerator.is_main_process, desc="gradstep"
        ):
            for substep_idx in range(1, self.accelerator.gradient_accumulation_steps + 1):
                # WARNING: self.accelerator.accumulate can lead to misleading results, since sync_gradients is
                # dependent on whether the registered dataloader ends or not (or step % accumulation_steps).
                # If your dataloader ends before the last step, gradients are not synced, and the optimizer wants to
                # update. This gives you a shape mismatch error.
                should_sync = substep_idx == self.accelerator.gradient_accumulation_steps
                context = contextlib.nullcontext if should_sync else self.accelerator.no_sync
                # no_sync here results in higher memory usage because FSDP will accumulate the full model gradients
                # (instead of gradient shards) until the eventual sync.
                with context(self.policy):
                    batch = next(rollouts_dataloader)
                    loss, stats_for_this_step = self.compute_loss(batch, **kwargs)
                    self.accelerator.backward(loss)
                    if should_sync:
                        if self.args.max_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                        stats_for_this_step["loss/grad_norm"] = self._compute_grad_norm()
                        stats_list.append(stats_for_this_step)
                        self.accelerator.unwrap_optimizer(self.optimizer).step()
                        self.policy.zero_grad(set_to_none=True)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        stats = common.merge_dict(stats_list, torch.stack)  # list of dict -> dict: str -> 1-D tensor
        stats = self.record_step_stats(stats, step_idx=step_idx)
        return stats

    def compute_loss(
        self, batch: Dict[str, Tensor], logprobs_coef=1.0, kl_coef=None, entropy_coef=None
    ) -> Tuple[Tensor, Dict]:
        self.policy.train()

        queries, query_attn_masks, responses = common.unpack_dict(
            common.prepare_inputs(batch, device=self.accelerator.device),
            keys=("queries", "query_attn_masks", "responses"),
            return_type=tuple,
        )
        queries_no_quark, query_attn_masks_no_quark = ignore_tokens(
            input_ids=queries,
            attention_mask=query_attn_masks,
            tokens_to_ignore=self.tokenizer.additional_special_tokens_ids,
        )

        policy_outputs = self.policy(queries, query_attn_masks, responses, temperature=self.args.temperature)
        with torch.inference_mode():
            ref_policy_outputs = self.ref_policy(
                queries_no_quark, query_attn_masks_no_quark, responses, temperature=self.args.temperature
            )

        logits, logprobs = common.unpack_dict(policy_outputs, keys=("logits", "logprobs"))
        (ref_logits,) = common.unpack_dict(ref_policy_outputs, keys=("logits",))

        original_vocab_size = len(self.tokenizer) - self.args.num_reward_tokens
        logits, ref_logits = tuple(t[..., :original_vocab_size] for t in (logits, ref_logits))

        kl_per_token = F.kl_div(F.log_softmax(ref_logits, dim=-1), F.softmax(logits, dim=-1), reduction="none").sum(
            dim=-1
        )
        entropies = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)

        # https://github.com/GXimingLu/Quark/blob/a4baf754de15f4d9675dd394571a7dd35fc0abd9/main.py#L252
        assert responses.size() == logprobs.size() == kl_per_token.size() == entropies.size()

        masks = responses == self.tokenizer.pad_token_id
        kl_per_token.masked_fill_(masks, 0.0)
        entropies.masked_fill_(masks, 0.0)

        kl_coef = self.kl_ctl.value if kl_coef is None else kl_coef
        entropy_coef = self.entropy_ctl.value if entropy_coef is None else entropy_coef
        loss = -logprobs * logprobs_coef + kl_per_token * kl_coef - entropies * entropy_coef
        loss = loss.mean()

        kl_avg_seq = kl_per_token.sum() / (~masks).sum()  # noqa
        kl_sum_seq = kl_per_token.sum() / kl_per_token.size(0)

        stats = dict(
            train=dict(
                logprobs=logprobs.mean(),
                entropies=entropies.mean(),
                kl_avg_seq=kl_avg_seq,
                kl_sum_seq=kl_sum_seq,
                loss=loss,
                masks=masks.float().sum(dim=1).mean(dim=0),  # noqa
            ),
        )
        return loss, common.flatten_dict(stats, sep="/", postprocess_fn=lambda x: x.detach())

    def get_train_dataloader(self):
        logger.warning(f"Train dataset size: {len(self.train_dataset)}")
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.args.rollout_per_device_batch_size,
            shuffle=True,  # Don't actually need to shuffle; shuffle to make consistent.
            drop_last=True,
        )
        train_dataloader = self.accelerator.prepare(train_dataloader)  # noqa
        self._log_batch_size(train_dataloader, "train_dataloader")
        return utils.InfiniteLoader(train_dataloader)

    @torch.inference_mode()
    def rollout(self, train_dataloader: utils.InfiniteLoader, step_idx: int) -> utils.InfiniteLoader:
        """Get responses conditioned on top reward token and add to data pool."""
        self.policy.eval()
        self._make_fsdp_happy()
        unwrapped_policy = self.accelerator.unwrap_model(self.policy, keep_fp32_wrapper=True)

        if self.args.clear_data_pool_on_each_rollout:
            self.data_pool.clear()

        text_queries_all, text_responses_all, rewards_all = [], [], []
        for batch_idx in tqdm.tqdm(
            range(self.args.rollout_accumulation_steps), disable=not self.accelerator.is_main_process, desc="rollout"
        ):
            batch = next(train_dataloader)
            queries, query_attn_masks = common.unpack_dict(
                common.prepare_inputs(batch, device=self.accelerator.device), keys=("queries", "query_attn_masks")
            )
            if step_idx == FIRST_STEP_IDX:  # Must ignore the reward token on first generation.
                queries, query_attn_masks = ignore_tokens(
                    input_ids=queries,
                    attention_mask=query_attn_masks,
                    tokens_to_ignore=self.tokenizer.additional_special_tokens_ids,
                )

            respond_outputs = unwrapped_policy.respond(queries, query_attn_masks, temperature=self.args.temperature)
            (responses,) = common.unpack_dict(respond_outputs, ("responses",))

            # Strings below should not contain reward tokens.
            text_queries, text_responses = tuple(
                self.tokenizer.batch_decode(tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for tensor in (queries, responses)
            )
            del queries, responses  # Prevent mistakes.

            text_sequences = [q + r for q, r in utils.zip_(text_queries, text_responses)]
            sequences = self.tokenizer(text_sequences, return_tensors="pt", padding=True, truncation=True)
            rewards = self.reward_model(**sequences).rewards

            # Nothing in here should contain the reward token!
            self.data_pool.add(queries=text_queries, responses=text_responses, rewards=rewards.tolist())

            text_queries_all.extend(text_queries)
            text_responses_all.extend(text_responses)
            rewards_all.extend(rewards.tolist())

        if self.accelerator.is_main_process:
            rollouts_to_disk = {"queries": text_queries_all, "responses": text_responses_all, "rewards": rewards_all}
            rollouts_to_disk = pd.DataFrame(rollouts_to_disk).to_dict(orient="records")
            utils.jdump(rollouts_to_disk, utils.join(self.args.output_dir, "rollouts", f"step_{step_idx}.json"))
            self.accelerator.log({"train/reward": utils.mean(rewards_all)}, step=step_idx)

        text_queries, text_responses, _ = self.data_pool.sort_and_get(self.args.train_on_best_quantile)
        rollouts_dataset = data_preprocessor.QueryResponseDataset(
            tokenizer=self.tokenizer,
            queries=text_queries,
            responses=text_responses,
            query_len=self.args.query_len,
            response_len=self.args.response_len,
        )
        rollouts_dataloader = DataLoader(
            dataset=rollouts_dataset,
            collate_fn=data_utils.DataCollatorForStackableDataset(),
            batch_size=self.args.step_per_device_batch_size,
            shuffle=True,
            drop_last=True,
        )
        rollouts_dataloader = utils.InfiniteLoader(rollouts_dataloader)
        return rollouts_dataloader

    def record_step_stats(self, stats, step_idx, **kwargs):
        for k, v in stats.items():
            stats[k] = v.mean(dim=0)
        stats = {key: value.item() if torch.is_tensor(value) else value for key, value in stats.items()}
        stats["train/kl_coef"] = self.args.kl_coef
        stats["train/entropy_coef"] = self.args.entropy_coef
        stats["train/lr"] = self.optimizer.param_groups[0]["lr"]
        if self.accelerator.is_main_process:
            self.accelerator.log(stats, step=step_idx)
        return stats

    @torch.inference_mode()
    def save_model(self, output_dir: Optional[str] = None, give_rw_access=True):
        output_dir = self.args.output_dir if output_dir is None else output_dir
        utils.makedirs(output_dir)

        model, tokenizer = self.policy, self.tokenizer
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ):
            logger.warning("Gathering full state_dict...")
            state_dict = model.state_dict()
            logger.warning("Finished gathering full state_dict...")

        if self.accelerator.is_main_process:
            # Retain and remap policy keys.
            new_state_dict = dict()
            prefix = "base_model."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_state_dict[key[len(prefix) :]] = value
            state_dict = new_state_dict

            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict

            unwrapped = unwrap_model(model).base_model
            assert isinstance(
                unwrapped, (transformers.OPTForCausalLM, transformers.LlamaForCausalLM)
            ), f"Expected to save a generative policy, but found model to be of type: {type(unwrapped)}."
            if hasattr(unwrapped, "_keys_to_ignore_on_save"):
                logger.warning(f"keys to ignore on save: {unwrapped._keys_to_ignore_on_save}")
            logger.warning(f"Saving model checkpoint to {output_dir}")
            logger.warning(f"Saving {len(cpu_state_dict)} keys:\n{utils.jdumps(cpu_state_dict.keys())}")
            unwrapped.save_pretrained(output_dir, state_dict=cpu_state_dict)

            tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together with the trained model
            torch.save(self.args, os.path.join(output_dir, constants.TRAINING_ARGS_NAME))


def _make_left_padded_tokenizer(
    model_name_or_path: AnyPath,
    cache_dir: AnyPathOrNone = constants.DEFAULT_CACHE_DIR,
    **kwargs,
) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        padding_side="left",
        **kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=constants.DEFAULT_PAD_TOKEN))
    return tokenizer


def make_tokenizer(args):
    # policy_tokenizer left pads, since the policy requires batch decoding.
    policy_tokenizer = _make_left_padded_tokenizer(
        args.policy_model_name_or_path, cache_dir=args.cache_dir, use_fast=args.use_fast_tokenizer
    )
    # reward_tokenizer left pads, since we need the embedding of the right most non-pad token.
    reward_tokenizer = _make_left_padded_tokenizer(
        args.reward_model_name_or_path, cache_dir=args.cache_dir, use_fast=args.use_fast_tokenizer
    )
    if policy_tokenizer.get_vocab() != reward_tokenizer.get_vocab():
        raise ValueError("AlpacaFarm does not support different tokenizer for policy and reward models.")

    logger.warning(f"Adding {args.num_reward_tokens} reward conditioning tokens for Quark.")
    policy_tokenizer.add_special_tokens(
        {"additional_special_tokens": [f"<reward_{i}>" for i in range(args.num_reward_tokens)]}  # noqa
    )
    return policy_tokenizer


def make_models(
    tokenizer: transformers.PreTrainedTokenizer,
    args,
    accelerator: accelerate.Accelerator,
):
    def make_generative_policy():
        base_model = common.make_generative_lm(
            model_name_or_path=args.policy_model_name_or_path,
            flash_attn=args.flash_attn,
            mixed_precision=accelerator.mixed_precision,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=True,
            device_map={"": accelerator.device},
        )
        utils.stable_resize_token_embeddings(base_model, len(tokenizer), jitter_new_embeddings=True)
        return base_model

    def make_reward_model():
        return reward_model_module.RewardModel.from_pretrained(
            args.reward_model_name_or_path,
            flash_attn=args.flash_attn,
            mixed_precision=accelerator.mixed_precision,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=True,
            device_map={"": accelerator.device},
        )

    policy = rl_models.make_policy_with_base_model(args, make_generative_policy(), tokenizer)
    policy = common.prepare_model_for_custom_fn(model=policy, fn_name="respond", accelerator=accelerator)
    policy = accelerator.prepare(policy)  # noqa

    ref_policy = rl_models.make_policy_with_base_model(args, make_generative_policy(), tokenizer)
    ref_policy.requires_grad_(False)
    ref_policy = accelerator.prepare(ref_policy)  # noqa

    reward_model = make_reward_model()
    reward_model.requires_grad_(False)
    reward_model = accelerator.prepare(reward_model)

    # TODO: This is a hack to get FSDP running. Remove in the future when this is fixed.
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        inputs = tokenizer("fsdp are you happy now??? :)" * 50, return_tensors="pt")
        inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
        policy(inputs["input_ids"], inputs["attention_mask"], inputs["input_ids"])

    return dict(policy=policy, ref_policy=ref_policy, reward_model=reward_model)
