from pathlib import Path
from typing import Sequence, Union

import numpy as np
import pandas as pd

from ..types import AnyPath


def read_or_return(to_read: Union[AnyPath, str], **kwargs):
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

