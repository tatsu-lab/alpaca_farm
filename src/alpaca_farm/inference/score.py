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

import math
import sys
from typing import List, Optional, Sequence, Tuple

import einops
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
    max_instances=sys.maxsize,
    mixed_precision: Optional[str] = None,
    tf32=False,
    divide_work=True,
):
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = tf32  # noqa

    local_rank, world_size = distributed_utils.setup()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    model.forward = common.cast_with_native_amp(model.forward, mixed_precision=mixed_precision)
    logger.warning(f"mixed_precision = {mixed_precision}")

    sequences = sequences[:max_instances]
    ori_data_size = len(sequences)

    # To make communication work, we round up the dataset to the nearest multiple of the actual batch size.
    if world_size > 1 and divide_work:
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
        if world_size > 1 and divide_work:
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
        if world_size > 1 and divide_work:
            rewards = distributed_utils.all_gather_and_cat(rewards, dim=0)
        return_rewards.extend(rewards.tolist())

    return return_rewards[:ori_data_size]


def score_sequences_with_huggingface(
    sequences: Sequence[str],
    model_name_or_path: str,
    per_device_batch_size=20,
    cache_dir=constants.DEFAULT_CACHE_DIR,
    max_instances=sys.maxsize,
    mixed_precision: Optional[str] = None,
    tf32=False,
    flash_attn=False,
) -> List[float]:
    """Score samples with a reward model.

    Args:
        sequences: A sequence of strings.
        model_name_or_path: Name of the reward model.
        per_device_batch_size: The batch size per device for evaluating rewards.
        cache_dir: The directory to cache the huggingface model.
        max_instances: The maximum number of prompts to rerank.
        mixed_precision: Whether to use mixed precision. If None, no casting will be performed.
        tf32: Whether to use tensorfloat32 for matrix multiplication.
        flash_attn: Turns on flash_attn for the reward model if True.

    Returns:
        A list of floats representing rewards.
    """
    model, tokenizer = load_model_and_tokenizer_for_inference(
        model_name_or_path=model_name_or_path,
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


@torch.inference_mode()
def rerank_sequences_with_huggingface(
    sequences: Sequence[Sequence[str]],
    model_name_or_path: str,
    rerank_top_k=1,
    per_device_batch_size=20,
    cache_dir=constants.DEFAULT_CACHE_DIR,
    mixed_precision: Optional[str] = None,
    max_instances=sys.maxsize,
    tf32=False,
    flash_attn=False,
) -> Tuple[List[List[str]], List[List[int]]]:
    """Rerank samples with a reward model.

    Args:
        sequences: A nested sequence of strings. Each inner sequence contains samples with the same prompt.
        model_name_or_path: Name of the reward model.
        rerank_top_k: The number of top samples to return.
        per_device_batch_size: The batch size per device for evaluating rewards.
        cache_dir: The directory to cache the huggingface model.
        max_instances: The maximum number of prompts to rerank.
        mixed_precision: Whether to use mixed precision. If None, no casting will be performed.
        tf32: Whether to use tensorfloat32 for matrix multiplication.
        flash_attn: Turns on flash_attn for the reward model if True.

    Returns:
        A tuple with two entries.
        The first is a nested sequence of strings. Each inner sequence contains the top-k samples with the same prompt.
        The second is a nested sequence of integers. Each inner sequence contains the indices of the top-k samples.
    """
    sequences = sequences[:max_instances]
    flat_sequences = [sequence_i_j for sequence_i in sequences for sequence_i_j in sequence_i]
    rewards = score_sequences_with_huggingface(
        sequences=flat_sequences,
        model_name_or_path=model_name_or_path,
        per_device_batch_size=per_device_batch_size,
        cache_dir=cache_dir,
        mixed_precision=mixed_precision,
        tf32=tf32,
        flash_attn=flash_attn,
    )
    rewards = einops.rearrange(torch.tensor(rewards), "(b m) -> b m", m=len(sequences[0]))
    # Nested list of "size" (data_size, num_options).
    top_indices = rewards.topk(rerank_top_k, dim=1).indices.tolist()
    top_sequences = [[sequence[i] for i in top_index] for sequence, top_index in utils.zip_(sequences, top_indices)]
    return top_sequences, top_indices
