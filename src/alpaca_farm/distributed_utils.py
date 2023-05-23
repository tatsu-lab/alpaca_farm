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

"""Utilities for PyTorch's distributed training.

Compatible with torchrun / elastic.

Internal map:
    https://github.com/lxuechen/ml-swissknife/blob/main/ml_swissknife/distributed_utils.py
"""

import os
import sys
from typing import Optional

import torch
import torch.distributed as dist

from .types import Tensor


def setup(rank: Optional[int] = None, world_size: Optional[int] = None):
    if rank is None:
        rank = get_local_rank()
    if world_size is None:
        world_size = get_world_size()

    if world_size <= 1:
        return rank, world_size

    if not dist.is_initialized():
        if sys.platform == "win32":
            # Distributed package only covers collective communications with Gloo
            # backend and FileStore on Windows platform. Set init_method parameter
            # in init_process_group to a local file.
            # Example init_method="file:///f:/libtmp/some_file"
            init_method = "file:///f:/libtmp/dist-tmp"
            dist.init_process_group(backend="gloo", init_method=init_method, rank=rank, world_size=world_size)
        elif torch.cuda.is_available():
            dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        else:
            dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    return rank, world_size


def cleanup():
    dist.destroy_process_group()


def get_local_rank():
    return int(os.getenv("LOCAL_RANK", 0))


def get_world_size():
    return int(os.getenv("WORLD_SIZE", 1))


def should_save():
    """Return True if the current process is the main process."""
    return get_local_rank() <= 0


def all_gather_and_cat(tensor: Tensor, dim=0):
    if get_world_size() > 1:
        tensor_list = [torch.empty_like(tensor) for _ in range(get_world_size())]
        dist.all_gather(tensor_list, tensor)
        tensor = torch.cat(tensor_list, dim=dim)
    return tensor


is_main_process = should_save
