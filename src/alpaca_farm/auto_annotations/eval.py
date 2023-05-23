import logging

import datasets
import pandas as pd

from alpaca_farm import constants
from alpaca_farm.auto_annotations import PairwiseAutoAnnotator
from alpaca_farm.auto_annotations.analysis import head2head_to_metrics
from alpaca_farm.types import AnyData, AnyPath

PRECOMPUTED_LEADERBOARD = {"annotators/annotator_pool_v0/configs.yaml": dict()}

__all__ = ["alpaca_leaderboard"]

def alpaca_leaderboard(
    all_outputs: AnyData,
    annotators_config: AnyPath = "annotators/annotator_pool_v0/configs.yaml",
    name: str = "Current method",
    is_add_reference_methods: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """Add the given model to the Alpaca leaderboard.

    Parameters
    ----------
    all_outputs : list of dict or pd.DataFrame
        The outputs of the model to add to the leaderboard. Each dictionary (or row) should contain the following keys:
        `instruction`, `input`, and `output`.

    annotators_config : str, optional
        The path to the annotator's config file. For details see the docstring of `PairwiseAutoAnnotator`.

    name : str, optional
        The name of the model to add to the leaderboard.

    is_add_reference_methods : bool, optional
        Whether to add the Alpaca reference methods to the leaderboard.

    kwargs :
        Additional arguments to pass to `PairwiseAutoAnnotator`.
    """
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
    annotated = annotator.annotate_head2head(
        outputs_1=outputs_baseline, outputs_2=all_outputs
    )
    all_metrics[name] = head2head_to_metrics(
        preferences=[a["preference"] for a in annotated]
    )

    if is_add_reference_methods:
        pass

    return pd.DataFrame(all_metrics).T.sort_values(by="win_rate", ascending=False)
