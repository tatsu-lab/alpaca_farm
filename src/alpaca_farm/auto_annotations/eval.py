import json
import logging
from typing import Union

import datasets
import pandas as pd

from alpaca_farm import constants
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
from alpaca_farm.auto_annotations.analysis import head2head_to_metrics
from . import utils as ann_utils

__all__ = ["alpaca_leaderboard"]

PRECOMPUTED_LEADERBOARD = {
    "annotators/annotator_pool_v0/configs.yaml": {
        "rlhf_llama_7b_regen_v7_3ep_v12_ckpt_20": {
            "n_draws": 9.0,
            "n_total": 803.0,
            "n_wins": 370.0,
            "n_wins_base": 424.0,
            "standard_error": 1.751619984513092,
            "win_rate": 46.63760896637609,
        },
        "sft_llama_7b_regen_v7_3ep": {
            "n_draws": 16.0,
            "n_total": 804.0,
            "n_wins": 320.0,
            "n_wins_base": 468.0,
            "standard_error": 1.7163543811890173,
            "win_rate": 40.79601990049751,
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
            "n_total": 804.0,
            "n_wins": 489.0,
            "n_wins_base": 306.0,
            "standard_error": 1.707975918938111,
            "win_rate": 61.38059701492538,
        },
        "LLaMA 7B": {
            "n_draws": 0.0,
            "n_total": 786.0,
            "n_wins": 94.0,
            "n_wins_base": 692.0,
            "standard_error": 1.1581361013229673,
            "win_rate": 11.959287531806616,
        },
        "GPT4": {
            "n_draws": 17.0,
            "n_total": 805.0,
            "n_wins": 639.0,
            "n_wins_base": 149.0,
            "standard_error": 1.3753918376580683,
            "win_rate": 80.43478260869566,
        },
    }
}


def alpaca_leaderboard(
    path_or_all_outputs: Union[ann_utils.AnyData, ann_utils.AnyPath],
    annotators_config: ann_utils.AnyPath = "annotators/annotator_pool_v0/configs.yaml",
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

    annotator = PairwiseAutoAnnotator(annotators_config=annotators_config, **kwargs)
    annotated = annotator.annotate_head2head(outputs_1=outputs_baseline, outputs_2=all_outputs)
    all_metrics[name] = head2head_to_metrics(preferences=[a["preference"] for a in annotated])

    df_results = pd.DataFrame(all_metrics).T.sort_values(by="win_rate", ascending=False)

    if is_print_metrics:
        print(df_results.to_string(float_format="%.2f"))
    else:
        return df_results
