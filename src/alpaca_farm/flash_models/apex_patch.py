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

from .. import logging

logger = logging.get_logger(__name__)

try:
    import apex

    apex_is_installed = True
    logger.warning("`apex` is installed. Using fused operators.")
except ImportError as e:
    apex_is_installed = False
    logger.warning("`apex` is not installed. Reverting to non-fused operators.")


def apex_layernorm(ln_module, input_):
    if apex_is_installed:
        return apex.normalization.fused_layer_norm.FusedLayerNormAffineFunction.apply(
            input_, ln_module.weight, ln_module.bias, ln_module.normalized_shape, ln_module.eps
        )
    else:
        return ln_module(input_)


def apex_rmsnorm(ln_module, input_):
    if apex_is_installed:
        return apex.normalization.fused_layer_norm.FusedRMSNormAffineFunction.apply(
            input_, ln_module.weight, ln_module.weight.size(), ln_module.variance_epsilon
        )
    else:
        return ln_module(input_)
