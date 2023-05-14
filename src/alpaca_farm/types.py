import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from torch import Tensor

AnyPath = Union[str, os.PathLike]
AnyPathOrNone = Optional[AnyPath]

Tensors = Sequence[Tensor]
TensorOrTensors = Union[Tensor, Tensors]
TensorList = List[Tensor]
StrOrStrs = Union[str, Sequence[str]]

if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa
else:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler
