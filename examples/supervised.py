import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
import transformers
from src.alpaca_farm import constants, utils, common

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="gpt2-xl")


@dataclass
class DataArguments:
    train_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the training data. Either `train_file_path` or `train_sql` needs to be specified."},
    )
    feedme_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the feedme data. Either `feedme_file_path` or `feedme_sql` needs to be specified."},
    )
    eval_file_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the evaluation data. If none of `eval_file_path` or `eval_sql` are "
                    "specified, we do an `eval_portion` random split from the training set."
        },
    )
    train_sql: Optional[str] = field(
        default=None,
        metadata={"help": "SQL query to select training data. Overloaded for both SFT and reward conditioning."},
    )
    train_db_key: Optional[str] = field(
        default="instruction_following",
        metadata={"help": "Name of the database for SQL queries."},
    )
    feedme_sql: Optional[str] = field(
        default=None,
        metadata={"help": "SQL query to select FEEDME data."},
    )
    feedme_db_key: Optional[str] = field(
        default="instruction_following",
        metadata={"help": "Name of the database for SQL queries."},
    )
    # TODO should remove as it's not used in supervised. RL and RM seem to use it but should use
    #  `gold_eval_sql_where` or `gold_val_sql_where` instead to avoid duplicates. See #441
    eval_sql: Optional[str] = field(
        default=None,
        metadata={"help": "SQL query to select validation data."},
    )
    prompt_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the prompt to use. If using SQL for data loading, prompt_name must be provided."},
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    pad_token: str = field(default=constants.DEFAULT_PAD_TOKEN)
    model_cache_dir: str = field(default=constants.FAST_MODEL_CACHE_DIR)
    tokenizer_cache_dir: str = field(default=constants.DEFAULT_CACHE_DIR)
    wandb_project: str = field(default=constants.WANDB_PROJECT)
    flash_attn: bool = field(default=True, metadata={"help": "Whether to use flash attention."})
    # transformers.TrainingArguments prevents 'adamw_torch_fused' from being used for pt<2.0, even though in principle
    # you could use it in 1.13.1.
    # On the other hand, there have been bug reports for this native fused adamw which are unresolved.
    # https://github.com/pytorch/pytorch/issues/90752
    # We stick to apex fused by default.
    optim: str = field(default="adamw_apex_fused")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded to this length (and possibly truncated)."
                    "Enforcing a consistent max length ensures memory usage is constant and predictable."
        },
    )
    padding: Literal["max_length", "longest"] = field(
        default="longest",
        metadata={
            "help": "Padding strategy. If 'max_length', pads to `model_max_length` always; this might lead to some "
                    "redundant compute. If 'longest', pads to the longest sequence in the batch, capped by `model_max_length`."
        },
    )
    resume_from_checkpoint: bool = field(default=False, metadata={"help": "If True, loads from last check point."})

    def __post_init__(self):
        super(TrainingArguments, self).__post_init__()

        if self.optim == "adamw_apex_fused":
            if common.apex_is_installed():
                logger.warning("apex is installed. Using apex FusedAdam.")
            else:
                logger.warning("apex is not installed. Reverting to native non-fused Adam.")
                self.optim = "adamw_torch"
