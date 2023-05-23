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

import abc
from typing import Union

import numpy as np
import torch
import torch.distributed as dist


class KLController(abc.ABC):
    value: Union[int, float]

    def step(self, *args, **kwargs):
        pass


class FixedKLController(KLController):
    def __init__(self, kl_coef):
        super(FixedKLController, self).__init__()
        self.value = kl_coef


class AdaptiveKLController(KLController):
    def __init__(self, init_kl_coef, target_kl, k_beta, accelerator=None):
        super(AdaptiveKLController, self).__init__()
        self.value = init_kl_coef
        self.target_kl = target_kl
        self.k_beta = k_beta
        self.accelerator = accelerator

    def step(self, current_kl: float):
        if self.accelerator is not None:
            current_kl = torch.tensor(current_kl, device=self.accelerator.device)
            dist.all_reduce(current_kl, op=dist.ReduceOp.SUM)
            current_kl = (current_kl / self.accelerator.num_processes).item()

        proportional_error = np.clip(current_kl / self.target_kl - 1, -0.2, 0.2)
        mult = 1.0 + self.k_beta * proportional_error
        self.value *= mult


def make_kl_controller(args, accelerator=None):
    if args.adaptive_kl:
        return AdaptiveKLController(
            init_kl_coef=args.kl_coef,
            target_kl=args.target_kl,
            k_beta=args.k_beta,
            accelerator=accelerator,
        )
    else:
        return FixedKLController(kl_coef=args.kl_coef)
