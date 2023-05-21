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

import accelerate


class MyAccelerator(accelerate.Accelerator):
    """Thin wrapper for accelerate.Accelerator."""

    def __repr__(self):
        return (
            f"Accelerator(\n"
            f"  state={self.state}, \n"
            f"  gradient_accumulation_steps={self.gradient_accumulation_steps:.6f}, \n"
            f"  split_batches={self.split_batches}, \n"
            f"  step_scheduler_with_optimizer={self.step_scheduler_with_optimizer},\n"
            f")"
        )

    def unwrap_optimizer(self, optimizer: accelerate.accelerator.AcceleratedOptimizer):
        return optimizer.optimizer
