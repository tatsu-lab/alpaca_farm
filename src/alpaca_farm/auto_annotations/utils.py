import copy
import itertools
import re
from collections import Counter
from pathlib import Path
import random
import logging
from typing import Any, Sequence, Union

import numpy as np
import pandas as pd

from .. import types
from .. import utils

DUMMY_EXAMPLE = dict(instruction="1+1=", output_1="2", input="", output_2="3")


def read_or_return(to_read: Union[types.AnyPath, str], **kwargs):
    """Read a file or return the input if it is already a string."""
    try:
        with open(Path(to_read), **kwargs) as f:
            out = f.read()
    except FileNotFoundError as e:
        logging.warning(f"Returning input because file not found. Error: {e}")
        out = to_read

    return out


def random_seeded_choice(seed: Union[str, int], choices, **kwargs):
    """Random choice with a seed."""
    if isinstance(seed, str):
        seed = abs(hash(seed))
    return np.random.default_rng(seed).choice(choices, **kwargs)


def shuffle_pairwise_preferences(df: pd.DataFrame, arr_is_shuffle: Sequence[int]) -> pd.DataFrame:
    """Shuffle the outputs of a pairwise preference dataframe.

    Examples
    --------
    >>> df = pd.DataFrame([dict(instruction='2+2', output_1='3', output_2='4', preference=2),
                           dict(instruction='2+3', output_1='5', output_2='4', preference=1)])
    >>> print(shuffle_pairwise_preferences(df, [True, False]))
        instruction output_1 output_2  preference
    0         2+2        4        3           1
    1         2+3        5        4           1
    """
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


def make_prompts(
    df: pd.DataFrame, template: str, batch_size: int = 1, padding_example=DUMMY_EXAMPLE
) -> tuple[list[str], pd.DataFrame]:
    """Helper function to make batch prompts for a single template.

    Parameters
    ----------
    df : pd.DataFrame
        Examples to annotate

    template : str
        Template for the prompt. Should have batch_size number of placeholder {key} where key is a column in df.

    batch_size : int
        Number of examples to batch in a single prompt.

    padding_example : dict
        Padding example to use if len(df) not divisible by batch_size.

    Returns
    -------
    prompts : list[str]
        List of formatted prompts.

    df_out : pd.DataFrame
        All examples. Will be df with potential padding examples.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"instruction": ["solve", "write backwards", "other 1"],
                           "input": ["1+1", "'abc'", ""]})
    >>> make_prompts(df, template="first: {instruction} {input}, second: {instruction} {input}",
                     batch_size=2, padding_example=dict(instruction="pad", input="pad_in"))[0]
    ["first: solve 1+1, second: write backwards 'abc'",
     'first: other 1 , second: pad pad_in']
    """

    if df.empty:
        return [], df

    text_to_format = re.findall("{(.+?)}", template)
    n_occurrences = Counter(text_to_format)

    if not all([n == batch_size for n in n_occurrences.values()]):
        raise ValueError(f"All placeholders should be repeated batch_size={batch_size} times but {n_occurrences}.")

    # padding if you don't have enough examples
    n_to_pad = (batch_size - len(df)) % batch_size
    padding = pd.DataFrame([padding_example] * n_to_pad)
    df_out = pd.concat([df, padding], axis=0, ignore_index=True)

    prompts = []
    # ugly for loops, not trivial to vectorize because of the batching
    for i in range(0, len(df_out), batch_size):
        current_prompt = copy.deepcopy(template)
        for j in range(batch_size):
            for to_format in n_occurrences.keys():
                # replace only first occurrence (that's why we don't use .format)
                current_prompt = current_prompt.replace("{" + to_format + "}", str(df_out.iloc[i + j][to_format]), 1)
        prompts.append(current_prompt)

    return prompts, df_out
