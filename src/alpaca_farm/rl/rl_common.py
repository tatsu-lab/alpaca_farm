# TODO: Rename this file. So far only PPO specific.
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import transformers

from .. import constants, logging
from ..types import AnyPath, AnyPathOrNone

logger = logging.get_logger(__name__)


def _make_tokenizer(
    model_name_or_path: AnyPath, cache_dir: AnyPathOrNone = constants.DEFAULT_CACHE_DIR
) -> transformers.PreTrainedTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=constants.DEFAULT_PAD_TOKEN))
    return tokenizer


@dataclass
class DataArguments:
    dataset_path: str = field(default="tatsu-lab/alpaca_farm")
    dataset_name: str = field(default="alpaca_instructions")
    train_splits: List[str] = field(default_factory=lambda: ["unlabeled"])
    eval_splits: List[str] = field(default_factory=lambda: ["val"])
    prompt_dict_path: str = field(
        default=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
        metadata={"help": "Path to the dictionary for the prompt to format examples."},
    )
    query_len: int = field(default=192)
    response_len: int = field(default=300)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    cache_dir: Optional[str] = field(default=constants.DEFAULT_CACHE_DIR)
    flash_attn: bool = field(default=True, metadata={"help": "Whether to use flash attention."})
    truncate_tokens: Optional[List[str]] = field(default_factory=lambda: None)
    truncate_after: Optional[int] = field(default=None)
    penalty_reward_value: float = field(default=-1.0)
    total_epochs: int = field(default=10)
    rollout_batch_size: int = field(default=512)
    step_batch_size: int = field(default=64)
    rollout_per_device_batch_size: int = field(default=4)
    step_per_device_batch_size: int = field(default=2)
    noptepochs: int = field(default=4)
    vf_coef: float = field(default=0.1)
    cliprange: float = field(default=0.2)
    cliprange_value: float = field(default=0.2)
    gamma: float = field(default=1.0)
    lam: float = field(default=1.0)
    whiten_rewards: bool = field(default=True)
    adam_epsilon: float = field(
        default=1e-5,
        metadata={
            "help": "Epsilon for AdamW optimizer. "
            "This is the default for OAI PPO code and UW Quark code. "
            "This is not the Hugging Face default."
        },
    )
    temperature: float = field(default=1.0)
    kl_coef: float = field(default=0.2)
    target_kl: float = field(default=6.0)
    k_beta: float = field(default=0.1)
    adaptive_kl: bool = field(default=False)
    eval_batches: int = field(default=sys.maxsize, metadata={"help": "Maximum number of batches to evaluate on."})
    init_value_with_reward: bool = field(
        default=True, metadata={"help": "Initialize the value model with the reward model."}
    )
    save_steps_extra: Optional[str] = field(
        default=None,
        metadata={
            "help": "A list of predetermined checkpoints to save, represented in the format 'no1__no2__no3'. "
            "Parse this with str.split('__')."
        },
    )
    policy_model_name_or_path: str = field(default=None)
    reward_model_name_or_path: str = field(default=None)

    def __post_init__(self):
        # Super class' __post_init__ is very complicated; don't do super for now in case mess something up.
        # super().__post_init__()

        if self.tf32:  # super().__post_init__() actually does this.
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True  # noqa

        # Checks on rollout_batch_size only matter for PPO.
        assert self.rollout_batch_size >= self.rollout_per_device_batch_size, (
            "`rollout_batch_size` is smaller than `rollout_per_device_batch_size`. "
            "Increase the former or decrease the latter to fix this."
        )
        assert (
            self.rollout_batch_size % self.rollout_per_device_batch_size == 0
        ), "`rollout_batch_size` is not a multiple of `rollout_per_device_batch_size`. "

        assert self.step_batch_size >= self.step_per_device_batch_size, (
            "`step_batch_size` is smaller than `step_per_device_batch_size`. "
            "Increase the former or decrease the latter to fix this."
        )
        assert (
            self.step_batch_size % self.step_per_device_batch_size == 0
        ), "`step_batch_size` is not a multiple of `step_per_device_batch_size`. "

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if self.save_steps_extra is not None:
            self.save_steps_extra_list = [int(string) for string in self.save_steps_extra.split("__")]
        else:
            self.save_steps_extra_list = []

        # TODO: refactor this ---
        self.policy_tokenizer = _make_tokenizer(self.policy_model_name_or_path)
        self.reward_tokenizer = _make_tokenizer(self.reward_model_name_or_path)

        # policy_tokenizer left pads, since the policy requires batch decoding.
        # reward_tokenizer also left pads, since we need the embedding of the right most non-pad token.
        self.policy_tokenizer.padding_side = "left"
        self.reward_tokenizer.padding_side = "left"
        # ---

        truncate_tokens = self.truncate_tokens
        if truncate_tokens is None:
            truncate_token_ids = None
        else:
            truncate_token_ids = self.reward_tokenizer.convert_tokens_to_ids(truncate_tokens)
        self.truncate_token_ids = truncate_token_ids

    def set_accumulation_steps(self, num_processes: int):
        logger.warning(
            f"rollout_batch_size: {self.rollout_batch_size}\n"
            f"rollout_per_device_batch_size: {self.rollout_per_device_batch_size}\n"
            f"num_processes: {num_processes}",
        )
        assert (self.rollout_batch_size // self.rollout_per_device_batch_size) % num_processes == 0
        self.rollout_accumulation_steps = self.rollout_batch_size // self.rollout_per_device_batch_size // num_processes

        logger.warning(
            f"step_batch_size: {self.step_batch_size}\n"
            f"step_per_device_batch_size: {self.step_per_device_batch_size}\n"
            f"num_processes: {num_processes}",  # Repeat to align log format.
        )
        assert (self.step_batch_size // self.step_per_device_batch_size) % num_processes == 0
        self.gradient_accumulation_steps = self.step_batch_size // self.step_per_device_batch_size // num_processes

        logger.warning(
            f"rollout_accumulation_steps: {self.rollout_accumulation_steps}, "
            f"gradient_accumulation_steps: {self.gradient_accumulation_steps}"
        )
