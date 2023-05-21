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

import copy
import dataclasses
from typing import Callable, Dict, Optional, Sequence, Union

import einops
import pandas as pd
import torch
import transformers
from torch.utils.data import Dataset

from . import constants, logging, torch_ops, utils
from .types import Tensor

logger = logging.get_logger(__name__)


def format_prompt(example: dict, prompt_dict: dict) -> str:
    """Formats a prompt with a prompt_dict formatter.

    Args:
        example: A dict-like object with required keys "instruction" and "input"
        prompt_dict: Dictionary containing the keys "prompt_noinputs" and "prompt_inputs" which have
            placeholders corresponding to the keys from `example`. E.g. "{instruction}".

    Returns:
        A formatted prompt string.

    Examples
    --------
    >>> format_prompt(dict(instruction="test", input=""), prompt_dict=dict(prompt_noinputs="prompt {instruction} "))
    "prompt test"
    """
    assert "instruction" in example and "input" in example, "Internal error: example missing required keys."

    if example["input"] is None or len(example["input"]) == 0:
        formatted_prompt = prompt_dict["prompt_noinputs"].format_map(example)
    else:
        formatted_prompt = prompt_dict["prompt_inputs"].format_map(example)

    return formatted_prompt


def format_output(example: dict, eos_token: Optional[str] = None, output_key="output") -> str:
    if eos_token is None:
        eos_token = ""
    return f"{example[output_key]}{eos_token}"


def format_prompt_with_data_frame(
    df: pd.DataFrame,
    prompt_dict: dict,
    df_postprocessor: Optional[Callable] = None,
    return_dict=False,
):
    if df_postprocessor is not None:
        df = df_postprocessor(df)
    list_dict_data = df.to_dict(orient="records")

    prompts = [format_prompt(example, prompt_dict) for example in list_dict_data]
    metadata = {"prompt_dict": prompt_dict}

    if return_dict:
        return dict(prompts=prompts, list_dict_data=list_dict_data, metadata=metadata)
    return prompts, list_dict_data, metadata


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> dict:
    """Tokenize a list of strings and return the tokenized content as well metadata (e.g., truncation statistics)."""
    padding = getattr(tokenizer, "padding", "max_length")
    return_overflowing_tokens = transformers.__version__ <= "4.26.1"
    # TODO(lxuechen): Until HF supports fast tokenizer for OPT, we can't make a joint call on the list of strings
    #  when `return_overflowing_tokens=True`.
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding=padding,
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_overflowing_tokens=return_overflowing_tokens,
        )
        for text in strings
    ]

    if padding == "max_length":
        input_ids = labels = torch.cat([tokenized.input_ids for tokenized in tokenized_list])
    else:  # "longest"
        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]

    if return_overflowing_tokens:
        input_ids_lens = labels_lens = [
            tokenizer.model_max_length + tokenized.num_truncated_tokens.item() for tokenized in tokenized_list
        ]
        # `num_truncated_tokens` can be negative, if no truncation occurred.
        num_truncated_tokens = sum(max(tokenized.num_truncated_tokens.item(), 0) for tokenized in tokenized_list)
        num_truncated_examples = sum(tokenized.num_truncated_tokens.item() > 0 for tokenized in tokenized_list)
    else:
        logger.warning(
            "You are using a `transformers` version that does not support `return_overflowing_tokens=True`. "
            "The tokenization metadata will not be recorded."
            "In order to see truncation statistics, please downgrade to `transformers<=4.26.1`."
        )
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
        ]
        num_truncated_tokens = num_truncated_examples = -1

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        tokenization_metadata=dict(
            num_examples=len(tokenized_list),
            num_truncated_tokens=num_truncated_tokens,
            num_truncated_examples=num_truncated_examples,
            input_ids_avg_len=utils.mean(input_ids_lens),
            input_ids_max_len=max(input_ids_lens),
            input_ids_min_len=min(input_ids_lens),
            labels_avg_len=utils.mean(labels_lens),
            labels_max_len=max(labels_lens),
            labels_min_len=min(labels_lens),
            model_max_length=tokenizer.model_max_length,
        ),
    )


def preprocess_for_sft(
    df: pd.DataFrame,
    prompt_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    df_postprocessor=None,
    verbose=True,
) -> dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
    """Tokenize each example and create the labels.

    Args:
        df: DataFrame containing the data. Must have columns 'instruction', 'input', and 'output'.
        prompt_dict: Dictionary for formatting prompts.
        tokenizer: Tokenizer to use. If None, use the tokenizer for the given model.
        df_postprocessor: Function to apply to the DataFrame before tokenization.
        verbose: Whether to print tokenization metadata.

    Returns:
        A dictionary mapping str to torch.Tensor.
    """
    if df_postprocessor is not None:
        df = df_postprocessor(df)
    list_dict_data = df.to_dict(orient="records")

    sources = [format_prompt(dict_data, prompt_dict) for dict_data in list_dict_data]
    targets = [format_output(dict_data, eos_token=tokenizer.eos_token) for dict_data in list_dict_data]

    examples = [s + t for s, t in utils.zip_(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in utils.zip_(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = constants.IGNORE_INDEX  # Input context should not contribute to loss.

    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        metadata=dict(),
        tokenization_metadata=examples_tokenized["tokenization_metadata"],
    )
    if verbose:
        logger.warning(f"Tokenization metadata:\n{utils.jdumps(packaged_data['tokenization_metadata'])}")

    return packaged_data


def preprocess_for_reward_modeling(
    df: pd.DataFrame,
    prompt_dict: dict,
    tokenizer: transformers.PreTrainedTokenizer,
    df_postprocessor: Optional[Callable] = None,
    end_sequence_with_eos: bool = False,
    verbose=True,
) -> dict[str, torch.Tensor]:
    if df_postprocessor is not None:
        df = df_postprocessor(df)
    list_dict_data = df.to_dict(orient="records")

    index_0, index_1 = tuple(
        torch.full(size=(len(list_dict_data), 1), fill_value=fill_value, dtype=torch.long) for fill_value in (0, 1)
    )

    def _get_numeric_preference(example: dict):
        # 1 vs 2 is stored in table, but for modeling we use 0 vs 1; remap here.
        return {1: 0, 2: 1}[example["preference"]]

    choice = torch.tensor([[_get_numeric_preference(dict_data)] for dict_data in list_dict_data])

    def _get_text(example: dict, output_key: str):
        source = format_prompt(example, prompt_dict=prompt_dict)
        target = format_output(
            example,
            eos_token=tokenizer.eos_token if end_sequence_with_eos else None,
            output_key=output_key,
        )
        return source + target

    text_list_0, text_list_1 = tuple(
        [_get_text(dict_data, key) for dict_data in list_dict_data] for key in ("output_1", "output_2")
    )

    def _merge_tokenization_metadata(metadata_list: Sequence[dict]) -> dict:
        num_examples = sum(metadata["num_examples"] for metadata in metadata_list)
        num_truncated_tokens = sum(metadata["num_truncated_tokens"] for metadata in metadata_list)
        num_truncated_examples = sum(metadata["num_truncated_examples"] for metadata in metadata_list)
        input_ids_avg_lens = (
            sum([metadata["input_ids_avg_len"] * metadata["num_examples"] for metadata in metadata_list]) / num_examples
        )
        input_ids_max_len = max(metadata["input_ids_max_len"] for metadata in metadata_list)
        input_ids_min_len = min(metadata["input_ids_min_len"] for metadata in metadata_list)
        labels_avg_lens = (
            sum([metadata["labels_avg_len"] * metadata["num_examples"] for metadata in metadata_list]) / num_examples
        )
        labels_max_len = max(metadata["labels_max_len"] for metadata in metadata_list)
        labels_min_len = min(metadata["labels_min_len"] for metadata in metadata_list)
        return dict(
            num_examples=num_examples,
            num_truncated_tokens=num_truncated_tokens,
            num_truncated_examples=num_truncated_examples,
            input_ids_avg_len=input_ids_avg_lens,
            input_ids_max_len=input_ids_max_len,
            input_ids_min_len=input_ids_min_len,
            labels_avg_len=labels_avg_lens,
            labels_max_len=labels_max_len,
            labels_min_len=labels_min_len,
        )

    logger.warning(f"Tokenizing {len(list_dict_data)} pairs...")
    tokenized_0, tokenized_1 = tuple(_tokenize_fn(text_list, tokenizer) for text_list in (text_list_0, text_list_1))
    # "size" (bsz, 2, seq_len)
    input_ids = [list(pair) for pair in utils.zip_(tokenized_0["input_ids"], tokenized_1["input_ids"])]
    labels = [list(pair) for pair in utils.zip_(tokenized_0["labels"], tokenized_1["labels"])]
    tokenization_metadata = _merge_tokenization_metadata(
        [tokenized_0["tokenization_metadata"], tokenized_1["tokenization_metadata"]]
    )

    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        index_0=index_0,
        index_1=index_1,
        choice=choice,
        tokenization_metadata=tokenization_metadata,
        metadata=dict(mean_choice=choice.float().mean().item()),
    )
    if verbose:
        logger.warning(f"Tokenization metadata:\n{utils.jdumps(packaged_data['tokenization_metadata'])}")

    return packaged_data


def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def split_train_into_train_and_eval(train_dataset: Dataset, eval_size: int, seed: int) -> tuple[Dataset, Dataset]:
    assert eval_size < len(
        train_dataset  # noqa
    ), "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size  # noqa
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset


class SFTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        df_postprocessor: Optional[Callable] = None,
    ):
        super(SFTDataset, self).__init__()
        data_dict = preprocess_for_sft(
            df=df, prompt_dict=prompt_dict, tokenizer=tokenizer, df_postprocessor=df_postprocessor
        )
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.metadata = data_dict["metadata"]
        self.tokenization_metadata = data_dict["tokenization_metadata"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclasses.dataclass
class DataCollatorForSFTDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=constants.IGNORE_INDEX)
        # When sequences are right padded, `attention_mask` is only useful for T5 training.
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )


class BinaryRewardModelingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        df_postprocessor: Optional[Callable] = None,
        end_sequence_with_eos: bool = False,
    ):
        super(BinaryRewardModelingDataset, self).__init__()
        data_dict = preprocess_for_reward_modeling(
            df=df,
            prompt_dict=prompt_dict,
            tokenizer=tokenizer,
            df_postprocessor=df_postprocessor,
            end_sequence_with_eos=end_sequence_with_eos,
        )
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.index_0 = data_dict["index_0"]
        self.index_1 = data_dict["index_1"]
        self.choice = data_dict["choice"]
        self.metadata = data_dict["metadata"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            index_0=self.index_0[i],
            index_1=self.index_1[i],
            choice=self.choice[i],
        )


@dataclasses.dataclass
class DataCollatorForBinaryRewardModelingDataset(object):
    """
    This collation assumes data preprocessing converts text into *padded* tensors of the same length.
    For autoregressive models like OPT and GPT2, `input_ids` alone is sufficient to produce the rewards.
    For enc-dec models like T5, we need `labels`.

    `input_ids` and `labels` are tensors of size (bsz, num_candidates, max_seq_len), i.e., each batch instance has
    `num_candidates` generations/completions.
    `index_0` and `index_1` are tensors of size (bsz, num_pairs), and are used to index into `input_ids` and
    `labels` to find the first and second sequences in the pair.
    `choice` is a binary int/long tensor of size (bsz, num_pairs) indicating which sequence in the pair is better,
    i.e., 0 means the first sequence is preferred, and 1 means otherwise.
    """

    tokenizer: transformers.PreTrainedTokenizer

    def _left_pad_helper(self, instances: Sequence[dict], key: str):
        # TODO(lxuechen): Potentially replace with `transformers.PretrainedTokenizerBase.prepare_for_model`.
        # `instances` is a list of dicts, each dict has key whose value is a list of tensors, possibly of unequal length.
        input_ids = [seq for instance in instances for seq in instance[key]]  # Flatten.
        input_ids = torch_ops.pad_sequence_from_left(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = einops.rearrange(
            input_ids,
            "(bsz num_candidates) max_seq_len -> bsz num_candidates max_seq_len",
            num_candidates=len(instances[0][key]),
        )
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        index_0, index_1, choice = tuple(
            torch.stack([instance[key] for instance in instances]) for key in ("index_0", "index_1", "choice")
        )
        input_ids = self._left_pad_helper(instances, "input_ids")
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            index_0=index_0,
            index_1=index_1,
            choice=choice,
        )


class QueryResponseDataset(Dataset):
    """Dataset that emits tokenized left-padded queries."""

    def __init__(
        self,
        df: pd.DataFrame,
        prompt_dict: dict,
        tokenizer: transformers.PreTrainedTokenizer,
        query_len: int,
        df_postprocessor: Optional[Callable] = None,
    ):
        super(QueryResponseDataset, self).__init__()

        if df_postprocessor is not None:
            df = df_postprocessor(df)
        list_dict_data = df.to_dict(orient="records")

        # prompts are strings; queries are tensors.
        prompts = [format_prompt(example=dict_data, prompt_dict=prompt_dict) for dict_data in list_dict_data]
        queries = [
            tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.squeeze(dim=0) for prompt in prompts
        ]

        filtered_queries = [query for query in queries if len(query) <= query_len]
        logger.warning(
            f"Filtered out {len(queries) - len(filtered_queries)} instances out of {len(queries)} that "
            f"exceed length limit. These examples are not used for training, but will still be used in evaluation. "
        )

        queries = torch.stack(
            [
                torch_ops.left_pad(query, target_size=(query_len,), value=tokenizer.pad_token_id)
                for query in filtered_queries
            ]
        )

        self.queries = queries
        self.query_attn_masks = queries.ne(tokenizer.pad_token_id).long()

        # Auxiliary data.
        self.prompts = prompts
        self.list_dict_data = list_dict_data

    def __getitem__(self, i):
        return_dict = dict(queries=self.queries[i], query_attn_masks=self.query_attn_masks[i])
        return return_dict

    def __len__(self):
        return len(self.queries)


@dataclasses.dataclass
class DataCollatorForQueryResponseDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, Tensor]:
        return {key: torch.stack([instance[key] for instance in instances]) for key in instances[0].keys()}
