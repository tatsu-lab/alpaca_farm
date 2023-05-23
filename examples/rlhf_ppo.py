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

import os

import transformers
from accelerate import DistributedDataParallelKwargs

from alpaca_farm import accelerate_patch, data_utils, logging
from alpaca_farm.rl.ppo_trainer import PPOTrainer, make_models, make_tokenizer
from alpaca_farm.rl.ppo_utils import DataArguments, TrainingArguments

logger = logging.get_logger(__name__)


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = transformers.HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = accelerate_patch.MyAccelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with=["wandb"],
        even_batches=True,  # Make sure the batch size on each device is the same.
        split_batches=False,  # Don't break a batch into smaller chunks.
        step_scheduler_with_optimizer=False,  # Untie optimizer and scheduler step.
        # Value model might not use all parameters (e.g., lm-head) in the forward pass.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    accelerator.init_trackers(
        training_args.wandb_project,
        init_kwargs={"wandb": {"name": training_args.run_name}},
        config=training_args.__dict__,
    )
    logger.warning(accelerator.state, main_process_only=False)  # Each process log their own state.

    tokenizer: transformers.PreTrainedTokenizer = make_tokenizer(args=training_args)
    model_module: dict = make_models(tokenizer=tokenizer, args=training_args, accelerator=accelerator)
    data_module: dict = data_utils.make_rl_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )

    trainer = PPOTrainer(
        args=training_args,
        accelerator=accelerator,
        **data_module,
        **model_module,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
