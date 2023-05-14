# maps to common.py
import functools
import os
import re
import time
import types
import warnings
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.utils import convert_outputs_to_fp32, is_torch_version
from ml_swissknife import utils
from torch import nn
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from transformers.trainer import WEIGHTS_NAME, is_deepspeed_zero3_enabled

from . import constants, logging
from .types import AnyPath, AnyPathOrNone,
from torch import Tensor
from typing import Callable, Dict, Optional, Sequence, Union

logger = logging.get_logger(__name__)

try:
    from flash_attn import bert_padding
except ImportError as e:
    logger.warning(f"Failed to import flash attention with error {e}")

try:
    from models import hf_flash_gpt_2, hf_flash_opt
except ImportError as e:
    logger.warning(f"Failed to import flash attention gpt2 or opt with error {e}")

# Separate this out, since llama model is not stable.
try:
    from models import hf_flash_llama
except ImportError as e:
    logger.warning(f"Failed to import flash attention llama with error {e}")

def apex_is_installed():
    try:
        import apex

        return True
    except ImportError as _:
        return False

class staggered_object_creation(object):
    """
    Objection creation in a distributed setting could be very RAM-intensive.

    This function staggers the creation of objects on odd and even ranks, so that not all objects
    are created at once.

    Assumes local_rank == -1 means no distributed training.
    """

    def __init__(self, local_rank: int):
        super().__init__()
        self.local_rank = local_rank

    def __enter__(self, *args, **kwargs):
        del args, kwargs
        if self.local_rank != -1 and self.local_rank % 2 == 0:
            dist.barrier()
        return self

    def __exit__(self, *args, **kwargs):
        del args, kwargs
        if self.local_rank != -1:
            if self.local_rank % 2 == 1:
                dist.barrier()
            dist.barrier()  # Final safety barrier.

    def __call__(self, func):
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator

def make_generative_lm(
    model_name_or_path: str,
    flash_attn: bool,
    fp16: Optional[bool] = None,
    bf16: Optional[bool] = None,
    mixed_precision: Optional[str] = None,
    **kwargs,
):
    if fp16 is None:
        fp16 = mixed_precision == "fp16"
    if bf16 is None:
        bf16 = mixed_precision == "bf16"

    if flash_attn and not fp16 and not bf16:
        logger.warning(
            "Flash attention does not support fp32. Reverting to standard attention.", main_process_only=True
        )
        flash_attn = False

    if flash_attn:
        model_cls = hf_flash_llama.LlamaForCausalLM
    else:
        model_cls = transformers.LlamaForCausalLM

    return model_cls.from_pretrained(model_name_or_path, **kwargs)