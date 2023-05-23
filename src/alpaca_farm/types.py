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

import os
import pathlib
from typing import Any, List, Optional, Sequence, Union

import pandas as pd
import torch
import datasets
from torch import Tensor

AnyPath = Union[str, os.PathLike, pathlib.Path]
AnyPathOrNone = Optional[AnyPath]
AnyData = Union[Sequence[dict[str, Any]], pd.DataFrame, datasets.Dataset]

Numeric = Union[int, float]
Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]
TensorList = List[Tensor]
StrOrStrs = Union[str, Sequence[str]]

if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa
else:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler
