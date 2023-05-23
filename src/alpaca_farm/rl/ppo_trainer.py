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

import os
from typing import Callable, Dict, Optional, Tuple

import accelerate
import pandas as pd
import torch
import tqdm
import transformers
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from transformers.modeling_utils import unwrap_model

from .. import accelerate_patch, common, constants, data_preprocessor, logging, torch_ops, utils
from ..models import reward_model as reward_model_module
from ..models import rl_models
from ..types import AnyPath, AnyPathOrNone, LRScheduler, Tensor
from . import rl_trainer

logger = logging.get_logger(__name__)


class PPOTrainer(rl_trainer.RLTrainer):
    def __init__(
        self,
        args,
        train_dataset: data_preprocessor.QueryResponseDataset,
        eval_dataset: data_preprocessor.QueryResponseDataset,
        data_collator: Callable,
        policy: rl_models.ActorCritic,
        ref_policy: rl_models.Policy,
        reward_model,
        tokenizer: transformers.PreTrainedTokenizer,
        accelerator: accelerate_patch.MyAccelerator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(PPOTrainer, self).__init__(
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

    def _shape_reward(
        self, rewards: Tensor, responses: Tensor, logprobs: Tensor, ref_logprobs: Tensor
    ) -> Dict[str, Tensor]:
        # For some reason, line below doesn't work.
        # kl = (logits.softmax(dim=-1) * (logits.log_softmax(dim=-1) - ref_logits.log_softmax(dim=-1))).sum(dim=-1)
        kl = torch.clamp(logprobs - ref_logprobs, min=0.0)
        non_score_rewards = -self.kl_ctl.value * kl
        shaped_rewards = non_score_rewards.clone()
        # This introduces a small index off by one bug if pad_token_id == eos_token_id.
        terminal_positions = (responses != self.tokenizer.pad_token_id).sum(dim=1) - 1
        shaped_rewards[list(range(rewards.size(0))), terminal_positions] += rewards
        return dict(shaped_rewards=shaped_rewards, non_score_rewards=non_score_rewards, kl=kl)

    def _estimate_advantage(self, rewards: Tensor, values: Tensor) -> Dict[str, Tensor]:
        """Generalized advantage estimation.

        Reference:
            https://arxiv.org/abs/1506.02438
        """
        if self.args.whiten_rewards:
            rewards = torch_ops.whiten(rewards, shift_mean=False)
        lastgaelam = 0
        advantages_reversed = []
        gen_length = self.args.response_len
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        advantages = torch_ops.whiten(advantages, shift_mean=True)
        return dict(returns=returns, advantages=advantages)

    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, Tensor]:
        """Rollout trajectories with policy.

        Args:
            queries_data: Sequence of batches or DataLoader.
                Each batch is a dict with keys 'queries' and 'query_attn_masks'.

        Returns:
            Dictionary with keys
                'queries', 'query_attn_masks', 'responses',
                'logprobs', 'ref_logprobs', 'values',
                'rewards', 'non_score_rewards', 'shaped_rewards'.
        """
        # Give up dropout throughout.
        self.policy.eval()
        self._make_fsdp_happy()
        # `keep_fp32_wrapper` retains the autocast wrapper of model.forward created by accelerate:
        #  recall one sets mixed precision options with accelerator.
        # The precise value of this arg doesn't matter here, since we use the unwrapped model only for respond.
        # Generally, try to use the wrapped model as much as you can, since it's got the autocast/cast-back wrappers.
        unwrapped_policy = self.accelerator.unwrap_model(self.policy, keep_fp32_wrapper=True)

        self.ref_policy.eval()
        self.reward_model.eval()

        rollouts = []
        for batch_idx, batch in tqdm.tqdm(
            enumerate(queries_data),
            disable=not self.accelerator.is_main_process,
            desc="rollout",
        ):
            # Sample rollouts.
            queries, query_attn_masks = common.unpack_dict(
                common.prepare_inputs(batch, device=self.accelerator.device),
                keys=("queries", "query_attn_masks"),
            )
            respond_outputs = unwrapped_policy.respond(queries, query_attn_masks, temperature=self.args.temperature)
            (responses,) = common.unpack_dict(respond_outputs, ("responses",))

            # Evaluate logprobs of the samples.
            rollouts_batch = {"queries": queries, "query_attn_masks": query_attn_masks, "responses": responses}
            policy_outputs = self.policy(**rollouts_batch, temperature=self.args.temperature)
            ref_policy_outputs = self.ref_policy(**rollouts_batch, temperature=self.args.temperature)
            policy_outputs = common.unpack_dict(
                policy_outputs, keys=("logprobs", "values", "entropies"), return_type=dict
            )
            ref_policy_outputs = common.unpack_dict(
                ref_policy_outputs, keys=("logprobs", "entropies"), return_type=dict
            )
            rollouts_batch.update(policy_outputs)
            rollouts_batch.update({f"ref_{key}": value for key, value in ref_policy_outputs.items()})

            # Evaluate reward of the samples.
            text_queries, text_responses = tuple(
                self.tokenizer.batch_decode(tensor, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for tensor in (queries, responses)
            )
            del queries, responses  # Prevent mistakes.

            # We retokenizer, since policy and reward model might not have the same tokenizer.
            # TODO(lxuechen): Avoid retokenization when policy and reward tokenizer are the same.
            text_sequences = [q + r for q, r in utils.zip_(text_queries, text_responses)]
            # TODO(lxuechen): This response retokenization has issues with OPT, since the tokenizer always prepend
            #  <bos_token>. But the issue is local to post_reward, which isn't an issue if we don't penalize.
            sequences, responses = tuple(
                self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                for text in (text_sequences, text_responses)
            )
            sequences, responses = common.prepare_inputs((sequences, responses), device=self.accelerator.device)

            reward_outputs = self.reward_model(**sequences)
            reward_outputs = self.post_reward(reward_outputs, responses.input_ids)
            rollouts_batch.update(reward_outputs)

            # Shape reward with KL penalty.
            shape_reward_outputs = self._shape_reward(
                rewards=rollouts_batch["rewards"],
                responses=rollouts_batch["responses"],
                logprobs=rollouts_batch["logprobs"],
                ref_logprobs=rollouts_batch["ref_logprobs"],
            )
            rollouts_batch.update(shape_reward_outputs)

            rollouts_batch_cpu = {key: value.cpu() for key, value in rollouts_batch.items()}
            rollouts.append(rollouts_batch_cpu)

        # Items in dict need to be of same shape.
        rollouts = common.merge_dict(rollouts, merge_fn=torch.cat)
        # Estimating advantages outside the loop gives more samples for reward normalization.
        advantages = self._estimate_advantage(
            rewards=rollouts["shaped_rewards"].to(self.accelerator.device),
            values=rollouts["values"].to(self.accelerator.device),
        )
        advantages = {key: value.cpu() for key, value in advantages.items()}
        return {**rollouts, **advantages}

    def post_reward(self, reward_outputs: Dict[str, Tensor], responses: Tensor) -> Dict[str, Tensor]:
        """Assign bad reward values to sequences which didn't stop properly."""
        if self.args.truncate_token_ids is None:
            return reward_outputs

        def get_validity_mask(sequences: Tensor, end_token_id: int) -> Tensor:
            """Mark a batch element as False if the sequence doesn't end with `end_token_id` after `truncate_after`."""
            assert sequences.dim() == 2
            validity_mask = []
            for sequence in sequences:
                (nonzeros,) = (sequence == end_token_id).nonzero(as_tuple=True)
                if len(nonzeros) == 0:
                    validity_mask.append(False)
                else:
                    validity_mask.append(
                        self.args.truncate_after is None
                        or
                        # Last occurrence of `end_token_id` is after `truncate_after`.
                        nonzeros[-1] > self.args.truncate_after
                    )
            return torch.tensor(validity_mask, device=sequences.device)

        validity_masks = [get_validity_mask(responses, end_token_id) for end_token_id in self.args.truncate_token_ids]
        validity_mask = torch.stack(validity_masks).any(dim=0)  # Sequence is valid if it ends with any end token.
        rewards = reward_outputs["rewards"]
        rewards[~validity_mask] = self.args.penalty_reward_value
        return reward_outputs

    def compute_loss(self, rollouts: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        values, old_logprob, returns, advantages, queries, query_attn_masks, responses = common.prepare_inputs(
            common.unpack_dict(
                rollouts,
                keys=("values", "logprobs", "returns", "advantages", "queries", "query_attn_masks", "responses"),
            ),
            device=self.accelerator.device,
        )
        outputs = self.policy(queries, query_attn_masks, responses, temperature=self.args.temperature)

        vpred = outputs["values"]
        vpredclipped = torch.clamp(
            vpred,
            min=values - self.args.cliprange_value,
            max=values + self.args.cliprange_value,
        )
        vf_losses1 = (vpred - returns) ** 2.0
        vf_losses2 = (vpredclipped - returns) ** 2.0
        vf_loss = 0.5 * torch.maximum(vf_losses1, vf_losses2).mean()
        vf_clipfrac = (vf_losses2 > vf_losses1).to(torch.get_default_dtype()).mean()

        logprob = outputs["logprobs"]
        ratio = torch.exp(logprob - old_logprob)
        # When current policy is close to the old policy, the KL component of this advantage is approximately correct.
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange)
        pg_loss = torch.maximum(pg_losses, pg_losses2).mean()
        pg_clipfrac = (pg_losses2 > pg_losses).to(torch.get_default_dtype()).mean()  # noqa

        loss = pg_loss + self.args.vf_coef * vf_loss

        entropy = outputs["entropies"].mean()
        approxkl = 0.5 * ((logprob - old_logprob) ** 2.0).mean()

        return_mean, return_var = returns.mean(), returns.var(unbiased=False)
        value_mean, value_var = values.mean(), values.var(unbiased=False)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(
                vpred=vpred.mean(),
                error=((vpred - returns) ** 2).mean(),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
        )
        return loss, common.flatten_dict(stats, sep="/", postprocess_fn=lambda x: x.detach())

    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        kl = rollouts["kl"]
        kl_sum_seq, kl_avg_seq = kl.sum(dim=1).mean(dim=0), kl.mean()
        shaped_rewards = rollouts["shaped_rewards"].sum(dim=1).mean(dim=0)
        non_score_rewards = rollouts["non_score_rewards"].sum(dim=1).mean(dim=0)
        rewards = rollouts["rewards"].mean(dim=0)
        stats = {
            f"objective/kl_coef": kwargs["kl_coef"],
            f"objective/kl_sum_seq": kl_sum_seq,
            f"objective/kl_avg_seq": kl_avg_seq,
            f"objective/shaped_rewards": shaped_rewards,
            f"objective/non_score_rewards": non_score_rewards,
            f"objective/rewards": rewards,  # Original model reward.
            f"objective/lr": self.optimizer.param_groups[0]["lr"],
            f"objective/entropies": rollouts["entropies"].mean(),
            f"objective/ref_entropies": rollouts["ref_entropies"].mean(),
        }
        for k, v in train_stats.items():
            stats[f"ppo/{k}"] = v.mean(dim=0)
        stats = {key: value.item() if torch.is_tensor(value) else value for key, value in stats.items()}
        if self.accelerator.is_main_process:
            self.accelerator.log(stats, step=step_idx)
            if self.args.output_dir is not None:
                # Store rollout data to disk to debug.
                rollouts_to_disk = {
                    key: self.tokenizer.batch_decode(
                        tensor, skip_special_tokens=False, clean_up_tokenization_spaces=False
                    )
                    for key, tensor in common.unpack_dict(
                        rollouts, keys=("queries", "responses"), return_type=dict
                    ).items()
                }
                rollouts_to_disk = pd.DataFrame(rollouts_to_disk).to_dict(orient="records")
                utils.jdump(rollouts_to_disk, utils.join(self.args.output_dir, "rollouts", f"step_{step_idx}.json"))
        return stats

    @torch.inference_mode()
    def save_model(self, output_dir: Optional[str] = None, give_rw_access=True, check_corrupted=True):
        # We don't use accelerator here because, we want to be frugal and only store the policy.
        # Moreover, we want easy loadability -- calling .from_pretrained on the folder. Full dump wouldn't allow this.

        # Logic:
        #   1. Retrieve the complete state dict of the wrapped model.
        #       (retrieving state dict of submodule can lead to loss of keys)
        #   2. Remove keys that are part of the value network.
        #   3. Rename keys that are part of the policy network, so that they match the naming standard.
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
            prefix = "policy.base_model."
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    new_state_dict[key[len(prefix) :]] = value
            state_dict = new_state_dict

            if check_corrupted:  # Let the checks run on GPU.
                is_corrupted = any(value.isnan().any().item() for value in state_dict.values())
                logger.warning(f"Is there nans in the state_dict to be dumped? {is_corrupted}")

            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict

            unwrapped = unwrap_model(model).policy.base_model
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

            if give_rw_access:
                try:
                    os.system(f"chmod -R a+xwr {output_dir}")
                except Exception as e:
                    logger.fatal(f"Failed to give read-write access to {output_dir}: {e}")


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
    return policy_tokenizer


def make_models(
    tokenizer: transformers.PreTrainedTokenizer,
    args,
    accelerator: accelerate.Accelerator,
) -> dict:
    def make_generative_policy():
        base_model = common.make_generative_lm(
            model_name_or_path=args.policy_model_name_or_path,
            flash_attn=args.flash_attn,
            mixed_precision=accelerator.mixed_precision,
            cache_dir=args.cache_dir,
            low_cpu_mem_usage=True,
            device_map={"": accelerator.device},
        )
        utils.stable_resize_token_embeddings(base_model, len(tokenizer))
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

    # Model construction below seems convoluted, but it's made to trade time for RAM efficiency.
    # For large models, object creation could be extremely RAM intensive.
    # Especially so for multiple processes on single node, each starting off with a copy of the model.
    # General strategy is to 1) create a model, 2) move it to target device / shard it, 3) then start next model,
    # as opposed to creating all needed models on CPU first, and separately moving / sharding each.
    policy = rl_models.make_policy_with_base_model(args, make_generative_policy(), tokenizer)
    if args.init_value_with_reward:
        # Initialize value from reward model a la OAI.
        logger.warning("Initializing value model with reward model.")
        value_model = rl_models.make_value_with_base_model(args, make_reward_model().backbone_model, tokenizer)
    else:
        logger.warning("Initializing value model with policy model.")
        # Initialize value from policy. Works for sanity, but generally performs worse in instruction-following.
        value_model = rl_models.make_value_with_base_model(args, make_generative_policy(), tokenizer)
    actor_critic = rl_models.ActorCritic(policy=policy, value_model=value_model)
    # We cast how respond should run. It's important the dtypes be consistent with training, since a bf16
    # fine-tuned model might not work with fp16 inference.
    # Cast step below must precede accelerator.prepare(), since wrapped model might not have `respond` method.
    actor_critic = common.prepare_model_for_custom_fn(model=actor_critic, fn_name="respond", accelerator=accelerator)
    actor_critic = accelerator.prepare(actor_critic)  # noqa

    ref_policy = rl_models.make_policy_with_base_model(args, make_generative_policy(), tokenizer)
    ref_policy.requires_grad_(False)
    ref_policy = accelerator.prepare(ref_policy)  # noqa

    reward_model = make_reward_model()
    reward_model.requires_grad_(False)
    reward_model = accelerator.prepare(reward_model)

    # TODO: This is a hack to get FSDP running. Remove in the future when we figure things out.
    if accelerator.distributed_type == accelerate.DistributedType.FSDP:
        inputs = tokenizer("fsdp are you happy now??? :)" * 50, return_tensors="pt")
        inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
        actor_critic(inputs["input_ids"], inputs["attention_mask"], inputs["input_ids"])

    return dict(policy=actor_critic, ref_policy=ref_policy, reward_model=reward_model)
