import pathlib
import sys
from typing import Dict, Sequence, Union

import datasets
import fire
import pandas as pd

from alpaca_farm import data_preprocessor, utils
from alpaca_farm.inference import decode, score
from alpaca_farm.types import AnyPath, AnyPathOrNone


def run_decode(
    decoder_name_or_path: AnyPath,
    prompt_dict_path=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
    output_path: AnyPathOrNone = None,
    split="val",
    max_instances=sys.maxsize,
    per_device_batch_size=4,
    temperature=1.0,
    num_return_sequences=4,
    max_new_tokens=300,
):
    """Decode samples from the policy language model.

    Args:
        decoder_name_or_path: Name or path of the policy language model.
        prompt_dict_path: Path to the prompt dictionary for formatting the instruction and input into a string.
        output_path: Optional path to save the decoding results.
        split: Split of the dataset to decode.
        max_instances: Maximum number of instances to decode.
        per_device_batch_size: Batch size for reranking for each device.
        temperature: Temperature for decoding.
        num_return_sequences: Number of sequences to return per each prompt.
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        List of dict data with keys 'prompt', 'completion', and 'decoder_name_or_path'.
        If num_return_sequences > 1, each 'completion' is a list of strings. Otherwise, it is a string.
    """
    dataset = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_instructions")

    prompts, list_dict_data, metadata = data_preprocessor.format_prompt_with_data_frame(
        df=pd.DataFrame(dataset[split]),
        prompt_dict=utils.jload(prompt_dict_path),
    )
    prompts, list_dict_data = prompts[:max_instances], list_dict_data[:max_instances]

    completions = decode.decode_prompts_with_huggingface(
        model_name_or_path=decoder_name_or_path,
        prompts=prompts,
        decoding_args=decode.HFDecodingArguments(
            temperature=temperature, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences
        ),
        per_device_batch_size=per_device_batch_size,
    )

    return_list_dict_data = [
        {"prompt": prompt, "completion": completion, "decoder_name_or_path": decoder_name_or_path}
        for dict_data, prompt, completion in utils.zip_(list_dict_data, prompts, completions)
    ]
    if output_path is not None:
        utils.jdump(return_list_dict_data, output_path)

    return return_list_dict_data


def run_rerank(
    list_dict_data_or_path: Union[Sequence[Dict], AnyPath],
    scorer_name_or_path: AnyPath,
    output_path: AnyPathOrNone = None,
    per_device_batch_size=4,
):
    """Rerank sequences with reward model.

    Args:
        list_dict_data_or_path: Sequence of dict data or a path to it.
            Each dict should have the keys 'prompt' and 'completion' with string values that can be added together.
        scorer_name_or_path: Name or path of the reward model.
        output_path: Optional path to save the rerank results.
        per_device_batch_size: Batch size for reranking for each device.

    Returns:
        Rerank results as a list of dict data with keys 'top_sequence', 'top_index', and 'scorer_name_or_path'.
    """
    if isinstance(list_dict_data_or_path, AnyPath):
        list_dict_data_or_path = utils.jload(list_dict_data_or_path)

    sequences = [
        [dict_data["prompt"] + completion for completion in dict_data["completion"]]
        for dict_data in list_dict_data_or_path
    ]

    top_sequences, top_indices = score.rerank_sequences_with_huggingface(
        sequences=sequences,
        model_name_or_path=scorer_name_or_path,
        per_device_batch_size=per_device_batch_size,
    )

    return_list_dict_data = [
        {"top_sequence": top_sequence, "top_index": top_index, "scorer_name_or_path": scorer_name_or_path}
        for top_sequence, top_index in utils.zip_(top_sequences, top_indices)
    ]
    if output_path is not None:
        utils.jdump(return_list_dict_data, output_path)

    return return_list_dict_data


def run_best_of_n(
    decoder_name_or_path: AnyPath,
    scorer_name_or_path: AnyPath,
    prompt_dict_path=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
    split="val",
    per_device_batch_size=4,
    max_instances=sys.maxsize,
    temperature=1.0,
    num_return_sequences=4,
    max_new_tokens=300,
):
    """Chain together decoding and rerank."""
    decode_return_list_dict_data = run_decode(
        decoder_name_or_path=decoder_name_or_path,
        prompt_dict_path=prompt_dict_path,
        split=split,
        max_instances=max_instances,
        per_device_batch_size=per_device_batch_size,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
    )
    rerank_return_list_dict_data = run_rerank(
        list_dict_data_or_path=decode_return_list_dict_data,
        scorer_name_or_path=scorer_name_or_path,
        per_device_batch_size=per_device_batch_size,
    )
    return rerank_return_list_dict_data


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
