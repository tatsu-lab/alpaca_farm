import math
import sys
from typing import List, Optional, Sequence

import torch
import tqdm
import transformers
from torch import nn

from .. import common, constants, distributed_utils, logging, utils
from ..models import reward_model
from .decode import load_model_and_tokenizer_for_inference

logger = logging.get_logger(__name__)


@torch.inference_mode()
def score_sequences_with_huggingface_given_model(
    model: nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    sequences: Sequence[str],
    per_device_batch_size=20,
    mixed_precision: Optional[str] = None,
    max_instances=sys.maxsize,
    tf32=True,
    divide_work=True,
):
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32  # noqa

    local_rank, world_size = distributed_utils.setup()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    model.forward = common.cast_with_native_amp(model.forward, mixed_precision=mixed_precision)
    if distributed_utils.is_main_process():
        logger.warning(f"mixed_precision = {mixed_precision}")

    sequences = sequences[:max_instances]
    ori_data_size = len(sequences)

    # To make communication work, we round up the dataset to the nearest multiple of the actual batch size.
    if divide_work:
        batch_size = per_device_batch_size * world_size
    else:
        batch_size = per_device_batch_size
    new_data_size = batch_size * int(math.ceil(ori_data_size / batch_size))  # Nearest multiple.
    new_sequences = list(sequences) + [sequences[-1]] * (new_data_size - ori_data_size)  # Pad with the last prompt.

    return_rewards = []
    for batch_idx, start_idx in tqdm.tqdm(
        enumerate(range(0, new_data_size, batch_size)),
        desc="evaluating rewards for batches",
        total=new_data_size // batch_size,
        disable=not distributed_utils.is_main_process(),
    ):
        batch = new_sequences[start_idx : start_idx + batch_size]
        if divide_work:
            local_batch = batch[local_rank * per_device_batch_size : (local_rank + 1) * per_device_batch_size]
        else:
            local_batch = batch

        source = tokenizer(
            local_batch,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        source = common.prepare_inputs(source, device=device)
        rewards = model(input_ids=source.input_ids, attention_mask=source.attention_mask).rewards
        if divide_work:
            rewards = distributed_utils.all_gather_and_cat(rewards, dim=0)
        return_rewards.extend(rewards.tolist())

    return return_rewards[:ori_data_size]


def score_sequences_with_huggingface(
    sequences: Sequence[str],
    model_name: str,
    per_device_batch_size=20,
    cache_dir=constants.DEFAULT_CACHE_DIR,
    mixed_precision: Optional[str] = None,
    max_instances=sys.maxsize,
    tf32=True,
    flash_attn=True,  # reward models are trained with flash attention by default
) -> List[float]:
    """Score samples with a reward model.

    Args:
        sequences: A sequence of strings.
        model_name: Name of the reward model.
        per_device_batch_size: The batch size per device for evaluating rewards.
        cache_dir: The directory to cache the huggingface model.
        mixed_precision: Whether to use mixed precision. If None, no casting will be performed.
        max_instances: The maximum number of prompts to rerank.
        tf32: Whether to use tensorfloat32 for matrix multiplication.
        flash_attn: Turns on flash_attn for the reward model if True.

    Returns:
        A list of floats representing rewards.
    """
    model, tokenizer = load_model_and_tokenizer_for_inference(
        model_name_or_path=model_name,
        model_cls=reward_model.RewardModel,
        cache_dir=cache_dir,
        model_kwargs=dict(
            torch_dtype=utils.convert_str_dtype_to_torch_dtype(mixed_precision),
            flash_attn=flash_attn,
        ),
    )
    return score_sequences_with_huggingface_given_model(
        model=model,
        tokenizer=tokenizer,
        sequences=sequences,
        per_device_batch_size=per_device_batch_size,
        mixed_precision=mixed_precision,
        max_instances=max_instances,
        tf32=tf32,
    )
