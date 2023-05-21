import transformers
from accelerate import DistributedDataParallelKwargs

from alpaca_farm import accelerate_patch, data_utils, logging
from alpaca_farm.rl.ppo_trainer import PPOTrainer, make_model_module
from alpaca_farm.rl.ppo_utils import DataArguments, TrainingArguments

logger = logging.get_logger(__name__)


def main():
    parser = transformers.HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parser.parse_args_into_dataclasses()

    # TODO: gradient_accumulation_steps set roundabout.
    accelerator = accelerate_patch.MyAccelerator(
        # Set to 1 temporarily. This will get updated later in this function.
        gradient_accumulation_steps=1,
        log_with=["wandb"],
        even_batches=True,  # Make sure the batch size on each device is the same.
        split_batches=False,  # Don't break a batch into smaller chunks.
        step_scheduler_with_optimizer=False,  # Untie optimizer and scheduler step.
        # Value model might not use all parameters (e.g., lm-head) in the forward pass.
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    accelerator.init_trackers(
        training_args.wandb_project,
        init_kwargs={"wandb": {"entity": training_args.wandb_entity, "name": training_args.run_name}},
        config=training_args.__dict__,
    )
    logger.warning(accelerator.state, main_process_only=False)  # Each process log their own state.

    # `accelerator.num_processes` should be set through config. Not equal to `torch.cuda.device_count()` generally!
    training_args.set_accumulation_steps(num_processes=accelerator.num_processes)
    accelerator.gradient_accumulation_steps = training_args.gradient_accumulation_steps  # Important!

    model_module: dict = make_model_module(args=training_args, accelerator=accelerator)
    data_module: dict = data_utils.make_rl_data_module(data_args=data_args, training_args=training_args)

    trainer = PPOTrainer(
        args=training_args,
        accelerator=accelerator,
        **model_module,
        **data_module,
    )
    trainer.train()


if __name__ == "__main__":
    main()
