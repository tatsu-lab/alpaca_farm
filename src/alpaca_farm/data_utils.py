import datasets
import pandas as pd
import transformers

from . import utils
from .data_preprocessor import (
    BinaryRewardModelingDataset,
    DataCollatorForBinaryRewardModelingDataset,
    DataCollatorForQueryResponseDataset,
    DataCollatorForSFTDataset,
    QueryResponseDataset,
    SFTDataset,
    split_train_into_train_and_eval,
)


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    training_args,
    data_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)

    alpaca_instructions = datasets.load_dataset(data_args.dataset_path, data_args.dataset_name)
    train_df = pd.concat([pd.DataFrame(alpaca_instructions[split]) for split in data_args.train_splits])
    eval_df = pd.concat([pd.DataFrame(alpaca_instructions[split]) for split in data_args.eval_splits])

    train_dataset = SFTDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
    )
    eval_dataset = SFTDataset(
        df=eval_df,
        prompt_dict=prompt_dict,
        tokenizer=tokenizer,
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
    train_df = pd.DataFrame(alpaca_human_preference["preference"])

    train_dataset = BinaryRewardModelingDataset(
        df=train_df,
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


def make_rl_data_module(
    data_args,
    training_args,
):
    prompt_dict = utils.jload(data_args.prompt_dict_path)

    alpaca_instructions = datasets.load_dataset(data_args.dataset_path, data_args.dataset_name)
    train_df = pd.concat([pd.DataFrame(alpaca_instructions[split]) for split in data_args.train_splits])
    eval_df = pd.concat([pd.DataFrame(alpaca_instructions[split]) for split in data_args.eval_splits])

    train_dataset = QueryResponseDataset(
        df=train_df,
        prompt_dict=prompt_dict,
        tokenizer=training_args.policy_tokenizer,
        query_len=data_args.query_len,
    )
    eval_dataset = QueryResponseDataset(
        df=eval_df,
        prompt_dict=prompt_dict,
        tokenizer=training_args.policy_tokenizer,
        query_len=data_args.query_len,
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=DataCollatorForQueryResponseDataset()
    )
