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

import abc
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import tqdm
import transformers
from accelerate import DistributedType
from accelerate.optimizer import AcceleratedOptimizer
from torch import nn
from torch.distributed.fsdp import ShardingStrategy
from torch.utils.data import DataLoader, TensorDataset
from transformers.trainer_utils import enable_full_determinism, set_seed

from .. import accelerate_patch, common, data_preprocessor, logging, trainer_utils, utils
from ..inference import decode, score
from ..types import LRScheduler, Tensor
from . import kl_controller

FIRST_STEP_IDX = 1

logger = logging.get_logger(__name__)


class RLTrainer(object):
    def __init__(
        self,
        args,
        train_dataset: data_preprocessor.QueryResponseDataset,
        eval_dataset: data_preprocessor.QueryResponseDataset,
        data_collator: Callable,
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: nn.Module,
        tokenizer: transformers.PreTrainedTokenizer,
        accelerator: accelerate_patch.MyAccelerator,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(RLTrainer, self).__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.accelerator = accelerator
        self.lr_scheduler = lr_scheduler
        self.kl_ctl = kl_controller.make_kl_controller(args, self.accelerator)
        self.log_history = []
        self.args.set_truncate_token_ids(self.tokenizer)
        enable_full_determinism(self.args.seed) if self.args.full_determinism else set_seed(self.args.seed)

    @abc.abstractmethod
    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, Tensor]:
        raise NotImplementedError

    @abc.abstractmethod
    def compute_loss(self, rollouts: Dict[str, Tensor]) -> Tuple[Tensor, Dict]:
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        raise NotImplementedError

    @property
    def optimizable_params(self):
        return [p for p in self.policy.parameters() if p.requires_grad and p.grad is not None]

    @torch.inference_mode()
    def _compute_grad_norm(self):
        grad_norm = torch.stack([p.grad.norm(2) for p in self.optimizable_params]).norm(2)
        if (
            self.accelerator.distributed_type == DistributedType.FSDP
            and self.policy.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            # When parameters are sharded, we need to gather each grad norm and then aggregate.
            grad_norms = [torch.zeros_like(grad_norm) for _ in range(self.accelerator.num_processes)]
            dist.all_gather(grad_norms, grad_norm)
            grad_norm = torch.stack(grad_norms).norm(2)
        return grad_norm

    @torch.inference_mode()
    def _compute_param_norm(self):
        param_norm = torch.stack([p.norm(2) for p in self.optimizable_params]).norm(2)
        if (
            self.accelerator.distributed_type == DistributedType.FSDP
            and self.policy.sharding_strategy != ShardingStrategy.NO_SHARD
        ):
            # When parameters are sharded, we need to gather each grad norm and then aggregate.
            param_norms = [torch.zeros_like(param_norm) for _ in range(self.accelerator.num_processes)]
            dist.all_gather(param_norms, param_norm)
            param_norm = torch.stack(param_norms).norm(2)
        return param_norm

    def _make_fsdp_happy(self):
        """Simply do a forward pass with the wrapped model at first.

        FSDP has some weird bugs; need this flush before running a non-forward method!
        This function should assume grad context of caller and
        not be wrapped with `torch.no_grad` or `torch.enable_grad`!!!
        """
        if self.accelerator.distributed_type == DistributedType.FSDP:
            inputs = self.tokenizer("fsdp are you happy now? :)" * 50, return_tensors="pt")
            inputs = common.prepare_inputs(inputs, device=self.accelerator.device)
            self.policy(inputs["input_ids"], inputs["attention_mask"], inputs["input_ids"])

    def step_with_rollouts(self, rollouts):
        """Based on fixed rollouts, run PPO for multiple epochs."""
        assert isinstance(self.optimizer, AcceleratedOptimizer), (
            "`optimizer` must be pushed through `accelerator.prepare`. "
            "Otherwise the `accelerator.accumulate` context manager won't correctly disable `zero_grad` or `step`."
        )
        rollouts_dataloader = self.get_rollouts_dataloader(rollouts=rollouts)
        stats_list = []
        for epoch_idx in range(self.args.noptepochs):
            for batch_idx, rollouts_batch in tqdm.tqdm(
                enumerate(rollouts_dataloader, 1), disable=not self.accelerator.is_main_process, desc="gradstep"
            ):
                with self.accelerator.accumulate(self.policy):
                    ppo_loss, stats_for_this_step = self.compute_loss(rollouts_batch)
                    self.accelerator.backward(ppo_loss)
                    if self.accelerator.sync_gradients:
                        # Gradient norm almost blows up at some point, but stabilizes eventually, even w/o clipping.
                        if self.args.max_grad_norm is not None:
                            self.accelerator.clip_grad_norm_(self.policy.parameters(), self.args.max_grad_norm)
                        stats_for_this_step["loss/grad_norm"] = self._compute_grad_norm()
                        stats_list.append(stats_for_this_step)
                    self.optimizer.step()
                    self.policy.zero_grad(set_to_none=True)
        return common.merge_dict(stats_list, torch.stack)  # list of dict -> dict: str -> 1-D tensor

    def step(self, train_dataloader, step_idx: int):
        queries_batches = [next(train_dataloader) for _ in range(self.args.rollout_accumulation_steps)]
        rollouts = self.rollout(queries_batches)
        train_stats = self.step_with_rollouts(rollouts)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        stats = self.record_step_stats(
            rollouts=rollouts, train_stats=train_stats, step_idx=step_idx, kl_coef=self.kl_ctl.value
        )
        self.kl_ctl.step(stats["objective/kl_sum_seq"])
        return stats

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        optimizer = trainer_utils.create_optimizer(args=self.args, model=self.policy, optimizer=self.optimizer)
        lr_scheduler = trainer_utils.create_scheduler(
            args=self.args, optimizer=optimizer, lr_scheduler=self.lr_scheduler, num_training_steps=num_training_steps
        )
        self.optimizer, self.lr_scheduler = self.accelerator.prepare(optimizer, lr_scheduler)
        self.accelerator.register_for_checkpointing(self.lr_scheduler)  # LR scheduler needs another call to save.
        return self.optimizer, self.lr_scheduler

    def train(self):
        """Entry point for training."""
        total_epochs = self.args.total_epochs
        total_episodes = len(self.train_dataset) * total_epochs  # noqa
        total_steps = total_episodes // self.args.rollout_batch_size  # noqa
        logger.warning(
            f"***Training starts***\n"
            f"Total epochs: {total_epochs} => Total episodes: {total_episodes} => Total steps: {total_steps}"
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
                self.evaluate(step_idx)
            self.log_history.append(self.step(infinite_train_dataloader, step_idx))
        return self.log_history

    @torch.inference_mode()
    def evaluate(self, step_idx: int, unwrapped_policy=None):
        """Evaluate by generating sequences with test prefixes.

        FSDP compat: all devices should do the forward pass, since sharded params need to be summoned.
                     only write results in the main process.
        """
        # TODO: unhardcode inference args.
        logger.warning(f"Start evaluation at step: {step_idx}", main_process_only=True)

        prompts, list_dict_data = self.eval_dataset.prompts, self.eval_dataset.list_dict_data
        if any(item is None for item in (prompts, list_dict_data)):
            logger.warning("No evaluation data, skipping evaluation.", main_process_only=True)
            return

        # Constants.
        model_name = Path(self.args.output_dir).stem  # Don't use the helper in common, as no checkpoint is saved yet.
        model_name_at_step = f"{model_name}_ckpt_{step_idx}"
        temperature = 0.7
        del model_name

        # Start evaluation.
        self.policy.eval()
        self._make_fsdp_happy()
        if unwrapped_policy is None:
            unwrapped_policy = self.accelerator.unwrap_model(self.policy, keep_fp32_wrapper=True)
            unwrapped_policy = unwrapped_policy.policy.base_model

        outputs = decode.decode_prompts_with_huggingface_given_model(
            model=unwrapped_policy,
            tokenizer=self.tokenizer,
            prompts=prompts,
            decoding_args=decode.HFDecodingArguments(max_new_tokens=self.args.response_len, temperature=temperature),
            per_device_batch_size=self.args.per_device_eval_batch_size,
            divide_work=False,
        )
        sequences = [i + o for i, o in utils.zip_(prompts, outputs)]
        rewards = score.score_sequences_with_huggingface_given_model(
            model=self.reward_model,
            tokenizer=self.tokenizer,
            sequences=sequences,
            per_device_batch_size=self.args.rollout_per_device_batch_size,
            divide_work=False,
        )

        if self.accelerator.is_main_process:
            results = [
                {"reward": reward, model_name_at_step: output, **example}
                for reward, output, example in utils.zip_(rewards, outputs, list_dict_data)
            ]
            if self.args.output_dir is not None:
                utils.jdump(results, utils.join(self.args.output_dir, f"eval_results_{step_idx}.json"))

            logger.warning(f"End evaluation at step: {step_idx}. Processed {len(results)} examples")

    @abc.abstractmethod
    @torch.inference_mode()
    def save_model(self, output_dir: Optional[str] = None):
        raise NotImplementedError

    def _log_batch_size(self, loader: DataLoader, loader_name):
        batch = next(iter(loader))
        if isinstance(batch, torch.Tensor):
            batch_size = batch.shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = batch[0]
        else:
            tensor = list(batch.values())[0]
            batch_size = tensor.size(0)
        logger.warning(f"Batch size of {loader_name} dataloader: {batch_size}", main_process_only=True)

    def get_train_dataloader(self):
        logger.warning(f"Train dataset size: {len(self.train_dataset)}", main_process_only=True)  # noqa
        train_dataloader = DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.args.rollout_per_device_batch_size,
            shuffle=True,
            drop_last=True,
        )
        train_dataloader = self.accelerator.prepare(train_dataloader)  # noqa
        self._log_batch_size(train_dataloader, "train_dataloader")
        return utils.InfiniteLoader(train_dataloader)

    def get_rollouts_dataloader(self, rollouts: Dict[str, Tensor], shuffle=True, drop_last=True, keys=None):
        if keys is None:
            keys = tuple(rollouts.keys())

        def collate_rollouts(instances: Sequence[tuple]):
            return {key: torch.stack([instance[idx] for instance in instances]) for idx, key in enumerate(keys)}

        rollouts_dataset = TensorDataset(*[rollouts[key] for key in keys])
        rollouts_dataloader = DataLoader(
            dataset=rollouts_dataset,
            batch_size=self.args.step_per_device_batch_size,
            collate_fn=collate_rollouts,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        # Do not prepare, since we don't need to shard the rollouts sampled on each batch.
        return rollouts_dataloader
