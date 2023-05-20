import datasets
import transformers

from . import utils
from .data_preprocessor import (
    BinaryRewardModelingDataset,
    DataCollatorForBinaryRewardModelingDataset,
    DataCollatorForSFTDataset,
    SFTDataset,
    format_prompt,
    split_train_into_train_and_eval,
)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    training_args,
    data_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)

    alpaca_instructions = datasets.load_dataset(data_args.dataset_path, data_args.dataset_name)
    alpaca_instructions = alpaca_instructions.map(lambda row: format_prompt(row, prompt_dict, return_dict=True))

    # support for multiple splits
    train_prompts = utils.flatten_nested_pystruct(
        [alpaca_instructions[split]["prompt"] for split in data_args.train_splits]
    )
    train_outputs = utils.flatten_nested_pystruct(
        [alpaca_instructions[split]["output"] for split in data_args.train_splits]
    )

    train_dataset = SFTDataset(
        tokenizer=tokenizer,
        prompts=train_prompts,
        targets=train_outputs,
    )
    eval_dataset = SFTDataset(
        tokenizer=tokenizer,
        prompts=alpaca_instructions["val"]["prompt"],
        targets=alpaca_instructions["val"]["output"],
    )

    data_collator = DataCollatorForSFTDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)


def make_binary_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)
    alpaca_human_preference = datasets.load_dataset(data_args.dataset_path, data_args.dataset_name)
    train_dataset = BinaryRewardModelingDataset(
        huggingface_dataset=alpaca_human_preference["preference"],
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
        df_postprocessor=data_args.train_df_postprocessor,
        end_sequence_with_eos=training_args.end_sequence_with_eos,
    )
    train_dataset, eval_dataset = split_train_into_train_and_eval(
        train_dataset=train_dataset,
        eval_size=data_args.eval_size,
        seed=training_args.seed,
    )
    data_collator = DataCollatorForBinaryRewardModelingDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
