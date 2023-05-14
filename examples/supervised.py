import contextlib
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
import transformers
from transformers import Trainer
from src.alpaca_farm import constants, utils, common, data_preprocessor

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

def pretrain():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ["WANDB_PROJECT"] = training_args.wandb_project

    if training_args.deepspeed is not None:
        ctx_mgr = contextlib.nullcontext()
        device_map = None
        low_cpu_mem_usage = None
    else:
        ctx_mgr = common.staggered_object_creation(local_rank=training_args.local_rank)
        device_map = {"": training_args.device.index}
        low_cpu_mem_usage = True

    with ctx_mgr:
        model: transformers.PreTrainedModel = common.make_generative_lm(
            model_name_or_path=model_args.model_name_or_path,
            flash_attn=training_args.flash_attn,
            fp16=training_args.fp16,
            bf16=training_args.bf16,
            config=transformers.AutoConfig.from_pretrained(model_args.model_name_or_path),
            cache_dir=training_args.model_cache_dir,
            low_cpu_mem_usage=low_cpu_mem_usage,
            device_map=device_map,
        )
        common.let_model_save_mem_when_zero_grad(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.tokenizer_cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",  # Ensures properly masking out the source tokens.
        use_fast=False,  # Fast GPT2 tokenizer breaks when we start counting the truncations.
    )
    tokenizer.padding = training_args.padding

    # Collect special tokens. Only add if non-existent.
    special_tokens_dict = dict(additional_special_tokens=[])
    if data_args.prompt_postprocessor is not None:
        for token in data_args.prompt_postprocessor.special_tokens:
            if token not in tokenizer.get_vocab():
                special_tokens_dict["additional_special_tokens"].append(token)
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = training_args.pad_token
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = constants.DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = constants.DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = constants.DEFAULT_UNK_TOKEN
    utils.smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    data_module: dict = data_preprocessor.make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    # Tokenizer is only supplied so that it gets saved; this makes loading easier.
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=training_args.callbacks,
        **data_module,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if training_args.should_save:
        logger.warning("hooray! training finished successfully! now on to model saving.")

    trainer.save_state()
    common.safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if training_args.should_save:
        logger.warning("hooray again! model saving worked.")
