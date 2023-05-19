import copy
import itertools
import re
from pathlib import Path
import random
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

from .. import types
from .. import utils


def read_or_return(to_read: Union[types.AnyPath, str], **kwargs):
    """Read a file or return the input if it is already a string."""
    try:
        with to_read.open(Path(to_read), **kwargs) as f:
            out = f.read()
    except:
        out = to_read

    return out


def random_seeded_choice(seed: Union[str, int], choices, **kwargs):
    """Random choice with a seed."""
    if isinstance(seed, str):
        seed = hash(seed)
    return np.random.default_rng(seed).choice(choices, **kwargs)


def shuffle_pairwise_preferences(df: pd.DataFrame, arr_is_shuffle: Sequence[int]) -> pd.DataFrame:
    """Shuffle the outputs of a pairwise preference dataframe."""
    col_1 = df["output_1"].copy()
    col_2 = df["output_2"].copy()
    df["output_1"] = np.where(arr_is_shuffle, col_2, col_1)
    df["output_2"] = np.where(arr_is_shuffle, col_1, col_2)

    if "preference" in df.columns:
        df["preference"] = np.where(arr_is_shuffle, 3 - df["preference"], df["preference"])

    return df


def is_derangement(arr1, arr2):
    """Whether 2 arrays are derangements of one another"""
    return any([a != b for a, b in utils.zip_(arr1, arr2)])


def random_derangement(arr, max_loop=10, seed=None):
    """
    Make random derangement of an array. I.e. shuffle without keeping any element in place. To be efficient,
    we first try `max_loop` rejection sampling. If didn't work then computes all possible derangement.
    """
    if len(arr) < 2:
        return arr

    if seed is not None:
        random.seed(seed)

    idcs = list(range(len(arr)))
    shuffled = list(range(len(arr)))

    for _ in range(max_loop):
        random.shuffle(shuffled)
        if is_derangement(idcs, shuffled):
            return arr[shuffled]

    # if no luck then computes all possibilities
    deranged_order = list(set([s for s in itertools.permutations(idcs) if is_derangement(s, idcs)]))
    return arr[list(random.choice(deranged_order))]


def find_first_match(text: str, outputs_to_match: dict[str, Any]):
    """Given text to parse and a dictionary of compiled regex to match, return the first match and corresponding key."""
    first_match = None
    first_key = None

    for key, compiled_regex in outputs_to_match.items():
        match = compiled_regex.search(text)
        if match and (not first_match or match.start() < first_match.start()):
            first_match = match
            first_key = key

    return first_match, first_key
