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

"""
Postprocessors for prompts and data frames.

Internal map:
    https://github.com/lxuechen/human-feedback/blob/main/instruction_following/postprocessor.py
"""

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
