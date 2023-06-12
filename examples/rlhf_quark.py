import os

import transformers
from accelerate import DistributedDataParallelKwargs

from alpaca_farm import accelerate_patch, data_utils, logging
from alpaca_farm.rl.quark_trainer import QuarkTrainer, make_models, make_tokenizer
from alpaca_farm.rl.quark_utils import DataArguments, TrainingArguments

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

    trainer = QuarkTrainer(
        args=training_args,
        accelerator=accelerator,
        **data_module,
        **model_module,
        tokenizer=tokenizer,
    )
    trainer.train()


if __name__ == "__main__":
    main()
