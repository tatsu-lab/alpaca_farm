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

from typing import Callable

import torch
from flash_attn import bert_padding


def pad_to_multiples_of_x(tensor: torch.Tensor, x: int = 8):
    """Pad a tensor along the batch dimension to a multiple of x."""
    total_nnz, hidden_size = tensor.size()
    pad_len = (x - total_nnz % x) % x
    if pad_len != 0:
        tensor = torch.cat(
            [
                tensor,
                torch.zeros([pad_len, hidden_size], device=tensor.device, dtype=tensor.dtype),
            ],
            dim=0,
        )

    def unpad_x(padded_tensor):
        return padded_tensor[:-pad_len] if pad_len > 0 else padded_tensor

    return tensor, unpad_x


def unpad_input(padded: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, Callable, torch.Tensor, int]:
    """Wrapper for unpad_input in official flash-attn."""
    batch_size, padded_seqlen = padded.shape[:2]
    unpadded, indices, cu_seqlens, max_seqlen = bert_padding.unpad_input(padded, attention_mask)

    def pad_back(unpadded: torch.Tensor):
        return bert_padding.pad_input(unpadded, indices, batch_size, padded_seqlen)

    return unpadded, pad_back, cu_seqlens, max_seqlen
