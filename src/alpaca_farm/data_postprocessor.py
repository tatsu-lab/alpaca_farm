# Stuff that reformats prompts (RC, Quark) goes here.
# maps to postprocessors.py

from dataclasses import dataclass
from typing import Callable, Sequence, Union

import pandas as pd


@dataclass
class SequentialPostProcessor(object):
    operations: Sequence[Callable]

    def __post_init__(self):
        special_tokens = []
        for operation in self.operations:
            if hasattr(operation, "special_tokens"):
                special_tokens.extend(operation.special_tokens)
        self.special_tokens = special_tokens

    def __call__(self, df: Union[pd.DataFrame, dict]) -> Union[pd.DataFrame, dict]:
        for operation in self.operations:
            df = operation(df)
        return df
