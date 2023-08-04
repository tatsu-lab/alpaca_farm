import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import alpaca_eval.annotators as eval_annotators
import alpaca_eval.utils as eval_utils
import datasets
import pandas as pd

import alpaca_eval.annotators as eval_annotators
import alpaca_eval.processors as eval_processors
from alpaca_eval import metrics

from .. import constants

__all__ = ["alpaca_leaderboard", "PairwiseAutoAnnotator"]

CURRENT_DIR = Path(__file__).parent
ANNOTATORS_CONFIG_DIR = CURRENT_DIR / "annotators"
DUMMY_EXAMPLE_NOINPUT = dict(instruction="Solve: 1+1=", output_1="2", output_2="3")
DUMMY_EXAMPLE_INPUT = dict(instruction="Solve:", output_1="2", input="1+1=", output_2="3")


PRECOMPUTED_LEADERBOARD = {
    "annotator_pool_v0/configs.yaml": {
        # Internal codename: rlhf_llama_7b_regen_v7_3ep_v12_ckpt_20
        "RLHF PPO": {
            "n_draws": 9.0,
            "n_total": 805.0,
            "n_wins": 392.0,
            "n_wins_base": 404.0,
            "standard_error": 1.753281981205392,
            "win_rate": 49.25465838509317,
        },
        # Internal codename: sft_v6_52k_llama_7b_regen_v7_3ep_recover
        "SFT 52k (Alpaca 7B)": {
            "n_draws": 16.0,
            "n_total": 805.0,
            "n_wins": 312.0,
            "n_wins_base": 477.0,
            "standard_error": 1.707927043869429,
            "win_rate": 39.75155279503105,
        },
        # Internal codename: sft_v6_llama_7b_regen_v7_3ep
        "SFT 10k": {
            "n_draws": 19.0,
            "n_total": 802.0,
            "n_wins": 278.00,
            "n_wins_base": 505.00,
            "standard_error": 1.67,
            "win_rate": 35.85,
        },
        "Davinci001": {
            "n_draws": 0.0,
            "n_total": 805.0,
            "n_wins": 201.0,
            "n_wins_base": 604.0,
            "standard_error": 1.5264851835334794,
            "win_rate": 24.96894409937888,
        },
        "ChatGPT": {
            "n_draws": 9.0,
            "n_total": 805.0,
            "n_wins": 503.0,
            "n_wins_base": 293.0,
            "standard_error": 1.6920642123984606,
            "win_rate": 63.04347826086957,
        },
        "LLaMA 7B": {
            "n_draws": 0.0,
            "n_total": 775.0,
            "n_wins": 98.0,
            "n_wins_base": 677.0,
            "standard_error": 1.1946348760380694,
            "win_rate": 12.645161290322582,
        },
        "GPT4": {
            "n_draws": 17.0,
            "n_total": 804.0,
            "n_wins": 631.0,
            "n_wins_base": 156.0,
            "standard_error": 1.4002932714785454,
            "win_rate": 79.53980099502488,
        },
    }
}


# TODO: alpaca_leaderboard could also be replaced with alpaca_eval functions
def alpaca_leaderboard(
    path_or_all_outputs: Union[eval_utils.AnyData, eval_utils.AnyPath],
    annotators_config: eval_utils.AnyPath = "annotator_pool_v0/configs.yaml",
    name: str = "Current method",
    is_add_reference_methods: bool = True,
    is_print_metrics: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Add the given model to the Alpaca leaderboard.

    Parameters
    ----------
    path_or_all_outputs : str or list of dict
        The outputs of the model to add to the leaderboard as a list of dictionaries, or a path to list of JSON. Each
        dictionary (or row) should contain the following keys: `instruction`, `input`, and `output`.

    annotators_config : str, optional
        The path to the annotator's config file. For details see the docstring of `PairwiseAutoAnnotator`.

    name : str, optional
        The name of the model to add to the leaderboard.

    is_add_reference_methods : bool, optional
        Whether to add the Alpaca reference methods to the leaderboard.

    is_print_metrics : bool, optional
        Whether to print the metrics.

    kwargs :
        Additional arguments to pass to `PairwiseAutoAnnotator`.
    """
    try:
        with open(path_or_all_outputs) as f:
            all_outputs = json.load(f)
            logging.info(f"Loaded outputs from {path_or_all_outputs}.")
    except:
        all_outputs = path_or_all_outputs

    if is_add_reference_methods:
        all_metrics = PRECOMPUTED_LEADERBOARD[annotators_config]
    else:
        all_metrics = dict()

    outputs_baseline = datasets.load_dataset(
        "tatsu-lab/alpaca_farm",
        "alpaca_farm_evaluation",
        cache_dir=constants.DEFAULT_CACHE_DIR,
    )["eval"]

    if len(all_outputs) != 805:
        logging.warning(
            f"""You gave {len(all_outputs)} outputs, but there are 805 examples in Alpaca Eval.
        We are computing the metrics on all examples you gave."""
        )

    outputs_1 = eval_utils.load_or_convert_to_dataframe(outputs_baseline)
    outputs_2 = eval_utils.load_or_convert_to_dataframe(all_outputs)
    annotator = PairwiseAutoAnnotator(annotators_config=annotators_config, **kwargs)
    annotated = annotator.annotate_head2head(outputs_1=outputs_1, outputs_2=outputs_2)
    all_metrics[name] = metrics.pairwise_to_winrate(preferences=[a["preference"] for a in annotated])

    df_results = pd.DataFrame(all_metrics).T.sort_values(by="win_rate", ascending=False)

    if is_print_metrics:
        print(df_results.to_string(float_format="%.2f"))
    else:
        return df_results


class PairwiseAutoAnnotator(eval_annotators.PairwiseAnnotator):
    def __init__(
        self,
        annotators_config: Union[eval_utils.AnyPath, list[dict[str, Any]]] = "annotator_pool_v0",
        input_keys: Sequence[str] = ("instruction", "input"),
        p_label_flip: Optional[float] = None,
        base_dir: eval_utils.AnyPath = ANNOTATORS_CONFIG_DIR,
        other_keys_to_keep: Sequence[str] = tuple(),
        **kwargs,
    ):
        super().__init__(
            annotators_config=annotators_config,
            input_keys=input_keys,
            p_label_flip=p_label_flip,
            base_dir=base_dir,
            other_keys_to_keep=other_keys_to_keep,
            **kwargs,
        )

    @property
    def SingleAnnotator(self):
        return SinglePairwiseAutoAnnotator


class SinglePairwiseAutoAnnotator(eval_annotators.SinglePairwiseAnnotator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for i, processor in enumerate(self.processors):
            # you need to use a batch padder that separates inputs and no inputs
            if isinstance(processor, eval_processors.PaddingForBatchesProcessor):
                self.processors[i] = PaddingForBatchesProcessorInputNoinput(
                    batch_size=self.batch_size,
                    padding_example_input=DUMMY_EXAMPLE_INPUT,
                    padding_example_noinput=DUMMY_EXAMPLE_NOINPUT,
                )

    def _get_prompt_template(self, prompt_template: dict[str, str]):
        # prompt_template will now be a dictionary of prompt templates of len 2 (one with and one without input)
        _get_prompt_template = super()._get_prompt_template
        return {k: _get_prompt_template(prompt) for k, prompt in prompt_template.items()}

    def _make_prompts(self, df_to_annotate, prompt_template=None):
        if prompt_template is None:
            prompt_template = self.prompt_template

        arr_is_inputs = (df_to_annotate["input"] != "") & (df_to_annotate["input"].notnull())
        df_with_inputs = df_to_annotate[arr_is_inputs]
        df_without_inputs = df_to_annotate[~arr_is_inputs]

        prompts, df = super()._make_prompts(
            df_without_inputs,
            prompt_template=prompt_template["without_inputs"],
        )
        if arr_is_inputs.any():
            prompts_i, df_i = super()._make_prompts(
                df_with_inputs,
                prompt_template=prompt_template["with_inputs"],
            )
            prompts += prompts_i
            df = pd.concat([df, df_i], axis=0, ignore_index=True)

        return prompts, df


class PaddingForBatchesProcessorInputNoinput(eval_processors.BaseProcessor):
    def __init__(self, batch_size, padding_example_input: dict, padding_example_noinput: dict, **kwargs):
        self.padded_noinput = eval_processors.PaddingForBatchesProcessor(
            batch_size=batch_size, padding_example=padding_example_noinput, **kwargs
        )
        self.padded_input = eval_processors.PaddingForBatchesProcessor(
            batch_size=batch_size, padding_example=padding_example_input, **kwargs
        )
        super().__init__(**kwargs)

    def preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        arr_is_inputs = (df_to_annotate["input"] != "") & (df_to_annotate["input"].notnull())
        if arr_is_inputs.any():
            # need to padd separately with and without inputs
            padded_df_without_inputs = self.padded_noinput.preprocess(df_to_annotate[~arr_is_inputs].copy())
            padded_df_with_inputs = self.padded_input.preprocess(df_to_annotate[arr_is_inputs].copy())

            df_out = pd.concat([padded_df_without_inputs, padded_df_with_inputs], axis=0, ignore_index=True)
        else:
            df_out = self.padded_noinput.preprocess(df_to_annotate)

        df_out["is_padding"] = df_out["is_padding"].astype(bool)

        return df_out

    def postprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        # independent of inputs
        return self.padded_noinput.postprocess(df_to_annotate)
