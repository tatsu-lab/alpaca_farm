# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General utility functions.

Internal map:
    https://github.com/lxuechen/ml-swissknife/blob/main/ml_swissknife/utils.py
"""
import argparse
import functools
import io
import json
import os
import random
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader

from . import logging
from .types import Numeric

logger = logging.get_logger(__name__)

home = os.path.expanduser("~")
home_data = os.path.join(home, "data")
join = os.path.join
pathexists = os.path.exists
makedirs = functools.partial(os.makedirs, exist_ok=True)
dirname = os.path.dirname
basename = os.path.basename


def alleq(l: Sequence, f: Optional[Callable] = lambda x, y: x == y):
    """Check all arguments in a sequence are equal according to a given criterion.

    Args:
        f: A bi-variate boolean function.
        l: A list/tuple.

    Returns:
        True if everything is equal; otherwise False.
    """
    return all(f(l[0], li) for li in l[1:])


def zip_(*args: Sequence):
    """Assert sequences of same length before zipping."""
    if len(args) == 0:
        return []
    assert alleq(args, lambda x, y: len(x) == len(y))
    return zip(*args)


def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            makedirs(f_dirname)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jdump(obj: Union[str, dict, list], f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jdumps(obj, indent=4, default=str):
    return json.dumps(obj, indent=indent, default=default)


def mean(*seqs: Sequence[Numeric]) -> Union[Numeric, Sequence[Numeric]]:
    singleton = len(seqs) == 1
    means = [float(np.mean(seq)) for seq in seqs]
    return means[0] if singleton else means


def stable_resize_token_embeddings_and_tokenizer(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    special_tokens_dict: dict,
):
    """Resize tokenizer and embedding together.

    For new tokens, the embedding value is the average of all old embedding vectors.
    """
    tokenizer.add_special_tokens(special_tokens_dict)
    stable_resize_token_embeddings(model, len(tokenizer))


def stable_resize_token_embeddings(model: transformers.PreTrainedModel, target_size: int, jitter_new_embeddings=False):
    num_new_tokens = target_size - model.get_input_embeddings().weight.size(0)
    model.resize_token_embeddings(target_size)

    if num_new_tokens > 0:

        @torch.inference_mode()
        def stable_init(embedding):
            embedding_data = embedding.weight.data
            embedding_avg = embedding_data[:-num_new_tokens].mean(dim=0, keepdim=True)
            embedding_data[-num_new_tokens:] = embedding_avg
            if jitter_new_embeddings:
                embedding_std = embedding_data[:-num_new_tokens].std(dim=0, keepdim=True)
                # The random tensor must be of the same shape as the new embeddings.
                embedding_data[-num_new_tokens:] += torch.randn_like(embedding_data[-num_new_tokens:]) * embedding_std

        input_embeddings = model.get_input_embeddings()  # Must grab this again after resize.
        output_embeddings = model.get_output_embeddings()
        # It doesn't matter if there's weight sharing or not; with sharing, the second init will overwrite the first.
        for embeddings in (input_embeddings, output_embeddings):
            stable_init(embeddings)


def convert_str_dtype_to_torch_dtype(str_dtype: Optional[str]):
    if str_dtype in ("single", "float32", "float", "fp32", None):
        return torch.float
    elif str_dtype in ("half", "float16", "fp16"):
        return torch.float16
    elif str_dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    elif str_dtype in ("double", "float64"):
        return torch.float64
    else:
        raise ValueError(f"Unknown dtype: {str_dtype}")


def manual_seed(args_or_seed: Union[int, argparse.Namespace], fix_cudnn=False):
    if hasattr(args_or_seed, "seed"):
        args_or_seed = args_or_seed.seed
    random.seed(args_or_seed)
    np.random.seed(args_or_seed)
    torch.manual_seed(args_or_seed)
    torch.cuda.manual_seed_all(args_or_seed)
    os.environ["PYTHONHASHSEED"] = str(args_or_seed)
    if fix_cudnn:
        torch.backends.cudnn.deterministic = True  # noqa
        torch.backends.cudnn.benchmark = False  # noqa


class InfiniteLoader(object):
    """Wraps an existing loader so that it outputs stuff indefinitely; useful for semi-supervised learning."""

    def __init__(self, loader: DataLoader):
        super(InfiniteLoader, self).__init__()
        self.loader = loader
        self.iterator = iter(loader)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def parallel_sort(*args: Sequence, key=None, reverse=False):
    """Parallel sort of multiple lists."""
    if key is None:
        # Parallel sort based on the order of the first list.
        key = lambda inputs: inputs[0]  # noqa
    ret = sorted(zip_(*args), key=key, reverse=reverse)
    return tuple([ret_i[j] for ret_i in ret] for j in range(len(args)))
