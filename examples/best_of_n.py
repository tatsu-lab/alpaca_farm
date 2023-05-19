import pathlib
import sys
from typing import Optional

import datasets
import fire

from alpaca_farm import data_preprocessor, utils
from alpaca_farm.inference import decode, score
from alpaca_farm.types import AnyPath, AnyPathOrNone


def run_decode(
    decoder_name_or_path: AnyPath,
    prompt_dict_path=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
    output_path: AnyPathOrNone = None,
    use_auth_token: Optional[str] = None,
    split="val",
    max_instances=sys.maxsize,
    temperature=1.0,
    num_return_sequences=4,
    max_new_tokens=300,
):
    dataset = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_instructions", use_auth_token=use_auth_token)

    prompts, list_dict_data, metadata = data_preprocessor.format_prompt_with_huggingface_dataset(
        huggingface_dataset=dataset[split],
        prompt_dict=utils.jload(prompt_dict_path),
    )
    prompts, list_dict_data = prompts[:max_instances], list_dict_data[:max_instances]

    completions = decode.decode_prompts_with_huggingface(
        model_name_or_path=decoder_name_or_path,
        prompts=prompts,
        decoding_args=decode.HFDecodingArguments(
            temperature=temperature, max_new_tokens=max_new_tokens, num_return_sequences=num_return_sequences
        ),
    )

    return_list_dict_data = [
        {"prompt": prompt, "completion": completion, "decoder_name_or_path": decoder_name_or_path}
        for dict_data, prompt, completion in utils.zip_(list_dict_data, prompts, completions)
    ]
    if output_path is not None:
        utils.jdump(return_list_dict_data, output_path)

    return return_list_dict_data


def run_rerank(
    list_dict_data_or_path,
    scorer_name_or_path: AnyPath,
    output_path: AnyPathOrNone = None,
):
    if isinstance(list_dict_data_or_path, AnyPath):
        list_dict_data_or_path = utils.jload(list_dict_data_or_path)

    sequences = [dict_data["prompt"] + dict_data["completion"] for dict_data in list_dict_data_or_path]

    top_sequences, top_indices = score.rerank_sequences_with_huggingface(sequences, scorer_name_or_path)

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
    use_auth_token: Optional[str] = None,
    split="val",
    max_instances=sys.maxsize,
    temperature=1.0,
    num_return_sequences=4,
    max_new_tokens=300,
):
    """Chain together decoding and rerank."""
    decode_return_list_dict_data = run_decode(
        decoder_name_or_path=decoder_name_or_path,
        prompt_dict_path=prompt_dict_path,
        use_auth_token=use_auth_token,
        split=split,
        max_instances=max_instances,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        max_new_tokens=max_new_tokens,
    )
    print(decoder_name_or_path)
    rerank_return_list_dict_data = run_rerank(
        list_dict_data_or_path=decode_return_list_dict_data,
        scorer_name_or_path=scorer_name_or_path,
    )
    print(rerank_return_list_dict_data)


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
