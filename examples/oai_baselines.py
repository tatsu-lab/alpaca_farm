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


from typing import Optional

import alpaca_eval.utils as eval_utils
import datasets
import fire
import pandas as pd

from alpaca_farm import constants, data_preprocessor, logging, openai_utils, types, utils

logger = logging.get_logger(__name__)
MODEL_TO_PROMPTS = {
    "text-davinci-003": "examples/prompts/v0_inputs_noinputs.json",
    "text-davinci-001": "examples/prompts/v0_inputs_noinputs.json",
    "gpt-3.5-turbo-0301": "examples/prompts/chatml_v0_char1k_inputs_noinputs.json",
    "gpt-4-0314": "examples/prompts/chatml_v0_char500_inputs_noinputs.json",
}


# TODO: all of this could just use alpaca_eval
def main_oai_baselines(
    all_instructions: Optional[types.AnyData] = None,
    model_name: str = "text-davinci-003",
    prompt_path: Optional[str] = None,
    save_path: Optional[str] = "examples/data/all_outputs/eval_{model_name}.json",
    decoding_args: Optional[openai_utils.OpenAIDecodingArguments] = None,
    batch_size: Optional[int] = None,
    num_procs: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Run the OAI baselines.

    Parameters
    ----------
    all_instructions : list of dict or DataFrame or Dataset, optional
        The instructions to evaluate on. If None uses Farm's eval data

    model_name : str, optional
        OpenAI model to use for completion.

    prompt_path : str, optional
        Path to the prompt dictionary. If None, uses the default prompt for the model.

    save_path : str, optional
        Path to save the outputs to. {model_name} will be formatted. If None, does not save.

    kwargs:
        Additional arguments to pass to `openai_utils.openai_completion`.
    """
    prompt_path = prompt_path or MODEL_TO_PROMPTS[model_name]

    if all_instructions is None:
        all_instructions = datasets.load_dataset(
            "tatsu-lab/alpaca_farm",
            "alpaca_farm_evaluation",
            cache_dir=constants.DEFAULT_CACHE_DIR,
        )["eval"]

    prompts, list_dict_data, _ = data_preprocessor.format_prompt_with_data_frame(
        df=eval_utils.convert_to_dataframe(all_instructions),
        prompt_dict=utils.jload(prompt_path),
    )

    if openai_utils.requires_chatml(model_name):
        decoding_args = decoding_args or openai_utils.OpenAIDecodingArgumentsChat(temperature=0.7, max_tokens=300)
        num_procs = num_procs or 5
        batch_size = batch_size or 1
    else:
        decoding_args = decoding_args or openai_utils.OpenAIDecodingArguments(temperature=0.7, max_tokens=300)
        num_procs = num_procs or 1
        batch_size = batch_size or 10

    completions = openai_utils.openai_completion(
        prompts=prompts,
        decoding_args=decoding_args,  # not useful, openai_completion should initialize this if None
        return_text=True,
        batch_size=batch_size,
        model_name=model_name,
        num_procs=num_procs,
        **kwargs,
    )

    df_data = eval_utils.convert_to_dataframe(list_dict_data)
    df_data["output"] = completions
    df_data["generator"] = model_name
    columns_to_keep = [
        "instruction",
        "input",
        "output",
        "generator",
        "dataset",
        "datasplit",
    ]

    if save_path is not None:
        logger.info(f"Saving to {save_path.format(model_name=model_name)}")
        df_data[columns_to_keep].to_json(save_path.format(model_name=model_name), orient="records", indent=2)

    return df_data[columns_to_keep]


if __name__ == "__main__":
    fire.Fire(main_oai_baselines)
