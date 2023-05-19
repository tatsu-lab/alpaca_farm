import pathlib
import sys

import datasets
import fire

from alpaca_farm import data_preprocessor, utils
from alpaca_farm.inference import decode


def main(
    model_name_or_path,
    prompt_dict_path=pathlib.Path(__file__).parent / "prompts" / "v0_inputs_noinputs.json",
    use_auth_token=None,
    split="val",
    max_instances=sys.maxsize,
):
    dataset = datasets.load_dataset("tatsu-lab/alpaca_farm", "alpaca_instructions", use_auth_token=use_auth_token)
    prompts = data_preprocessor.format_prompt_with_huggingface_dataset(
        huggingface_dataset=dataset[split],
        prompt_dict=utils.jload(prompt_dict_path),
        return_dict=False,
    )

    completions = decode.decode_prompts_with_huggingface(
        model_name_or_path,
        prompts=prompts,
        decoding_args=decode.HFDecodingArguments(temperature=0.7, max_new_tokens=300),
        max_instances=max_instances,
    )
    print(prompts[:max_instances])
    print(completions)


if __name__ == "__main__":
    fire.Fire(main)
