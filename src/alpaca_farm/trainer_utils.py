# Copyright 2023 The Alpaca Team
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from typing import Optional

from torch import nn, optim
from transformers import Trainer
from transformers.optimization import get_scheduler
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

from . import logging

logger = logging.get_logger(__name__)


def create_optimizer(args, model: nn.Module, optimizer: Optional[optim.Optimizer] = None):
    """Create optimizer for trainer.

    This is detached version of the `Trainer.create_optimizer` method.
    We don't support sagemaker and fairscale for simplicity.

    Reference:
        https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    """
    opt_model = model

    if optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        if optimizer_cls.__name__ == "Adam8bit":
            import bitsandbytes

            manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

            skipped = 0
            for module in opt_model.modules():
                if isinstance(module, nn.Embedding):
                    skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                    print(f"skipped {module}: {skipped / 2 ** 20}M params")
                    manager.register_module_override(module, "weight", {"optim_bits": 32})
                    logger.debug(f"bitsandbytes: will optimize {module} in fp32")
            print(f"skipped: {skipped / 2 ** 20}M params")

    return optimizer


def create_scheduler(args, optimizer, lr_scheduler, num_training_steps):
    """Create scheduler for trainer.

    This is detached version of the `Trainer.create_scheduler` method.

    Reference:
        https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
    """
    if lr_scheduler is None:
        lr_scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
    return lr_scheduler
