# Dataset, DataLoader, Collator goes here.
# Tokenization also goes here.
# maps to data_utils.py and data_preprocess.py
import copy
import dataclasses
from typing import Callable, Dict, Optional, Sequence, Union

import einops
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import Dataset

from . import constants, logging, torch_ops, utils
from .types import Tensor

logger = logging.get_logger(__name__)


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


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    verbose=True,
) -> dict[str, Union[torch.Tensor, Sequence[torch.Tensor]]]:
    """Tokenize each example, create the labels, and optionally store the lightweight form on disk.

    Truncate examples whose total length exceeds max_length.


    Parameters
    ----------
    sources : list of str
        Source text for each example.
    targets : list of str
        Target text for each example.
    path_pt : str
        Path to save the processed data to.
    path_metadata : str
        Path to save the metadata to.
    tokenizer: transformers.PreTrainedTokenizer
        Tokenizer to use. If None, use the tokenizer for the given model.
    export_to_cache : bool
        Whether to save the processed data to disk. If False, just return the data.

    Returns
    -------
    A dictionary mapping str to torch.Tensor.
    """
    logger.warning(f"Tokenizing {len(sources)} instances...")

    examples = [s + t for s, t in utils.zip_(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in utils.zip_(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = constants.IGNORE_INDEX  # Input context should not contribute to loss.

    packaged_data = dict(
        input_ids=input_ids,
        labels=labels,
        tokenization_metadata=examples_tokenized["tokenization_metadata"],
        metadata=dict(),  # Record other metadata of the dataset here.
        sources=sources,
        targets=targets,
    )
    if verbose:
        logger.warning(f"Tokenization metadata:\n{utils.jdumps(packaged_data['tokenization_metadata'])}")
    return packaged_data


def make_prompt(example, prompt_dict: dict) -> dict:
    """Formats an example with a prompt.

    Parameters
    ----------
    example : a dict-like object with required keys "instruction" and "input"

    prompt_dict : dict
        Dictionary containing the keys "instruction_only_prompt" and "instruction_input_prompt" which have placeholders
        corresponding to the keys from `example`. E.g. "{instruction}". You can use the output of `db_io.get_prompt_row`.

    Returns
    -------
    formatted_prompt : str
        Example with an additional key "preprocessed_input".

    Examples
    --------
    >>> make_prompt(dict(instruction="test", input=""), prompt_dict=dict(prompt_noinputs="prompt {instruction} "))
    "prompt test"
    """
    assert "instruction" in example and "input" in example

    if example["input"]:
        formatted_prompt = prompt_dict["prompt_noinputs"].format_map(example)
    else:
        formatted_prompt = prompt_dict["prompt_inputs"].format_map(example)

    return dict(prompt=formatted_prompt)


class SFTDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        prompts: Sequence[str],
        targets: Sequence[str],
    ):
        super(SFTDataset, self).__init__()
        data_dict = preprocess(
            sources=prompts,
            targets=targets,
            tokenizer=tokenizer,
        )

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.metadata = data_dict["metadata"]
        self.prompts = data_dict["sources"]
        self.targets = data_dict["targets"]

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


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    training_args,
    data_args,
):
    alpaca_instructions = load_dataset(
        "tatsu-lab/alpaca_farm", "alpaca_instructions", use_auth_token="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"
    )
    prompt_dict = {
        "prompt_inputs": open(f"src/alpaca_farm/prompts/{data_args.prompt_name.format(tag='inputs')}.txt").read(),
        "prompt_noinputs": open(f"src/alpaca_farm/prompts/{data_args.prompt_name.format(tag='noinputs')}.txt").read(),
    }
    alpaca_instructions = alpaca_instructions.map(lambda row: make_prompt(row, prompt_dict))

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


class BinaryRewardModelingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        prompt_name: Optional[str] = None,
        tokenizer: Optional[transformers.PreTrainedTokenizer] = None,
        df_postprocessor: Optional[Callable] = None,
        end_sequence_with_eos: bool = False,
    ):
        super(BinaryRewardModelingDataset, self).__init__()
        data_dict = preprocess_for_reward_modeling_with_sql(
            prompt_name=prompt_name,
            tokenizer=tokenizer,
            export_to_cache=False,
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


def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def _split_train_into_train_and_eval(train_dataset: Dataset, eval_size: int, seed: int) -> tuple[Dataset, Dataset]:
    assert eval_size < len(train_dataset), "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset


def make_binary_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    # TODO: get the df here.
    train_dataset = BinaryRewardModelingDataset(
        prompt_name=data_args.prompt_name,
        tokenizer=tokenizer,
        df_postprocessor=data_args.train_df_postprocessor,
        end_sequence_with_eos=training_args.end_sequence_with_eos,
    )
    train_dataset, eval_dataset = _split_train_into_train_and_eval(
        train_dataset=train_dataset,
        eval_size=data_args.eval_size,
        seed=training_args.seed,
    )
    data_collator = DataCollatorForBinaryRewardModelingDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
