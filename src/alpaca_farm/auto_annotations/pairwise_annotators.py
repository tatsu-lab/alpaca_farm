import copy
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Any, Callable, Optional, Sequence, Union
import re

import yaml

from .. import types
from . import decoders, utils as ann_utils


CURRENT_DIR = Path(__file__).parent


class PairwiseAutoAnnotator:
    """Class for a pool of annotators.

    Notes
    -----
    There are three main functions for annotations depending on how the outputs to compare are given:
        - annotate_pairs: annotate a sequence of examples that contain the pair of outputs `"output_1"` and `"output_2"`
        - annotate_samples: annotate a sequence of examples that contain `"output"` from which we will sample a pair of
            outputs. Useful for collecting pairwise preferences for RLHF.
        - annotate_head2head: annotate a pair of sequence of outputs, each containing `"output"` which will be merged
            into a single sequence of paired outputs. Useful for evaluation against a reference.

    Other functions that are useful for annotating:
        - set_noise: set the noise level for the annotators.

    Parameters
    ----------
    annotators_config : Path or list of dict
        A dictionary or path to a yaml file containing the configuration for the pool of annotators. The keys in the
        fist dictionary should be the annotator's name, and the value should be a dictionary of the annotator's
        configuration which should have the following keys:
        - prompt_templates (dict): a dictionary of prompts or path to the prompts. The dictionary of (loaded) prompts
            will be given to `fn_prompter`.
        - fn_decoder (str): function in `decoders.py` to use for decoding the output.
        - decoder_kwargs (dict): kwargs for fn_decode. E.g. model_name, max_completions_tokens
        - other secondary kwargs to SinglePairwiseAutoAnnotator

    seed : int, optional
        Seed for the random number generator.

    is_avoid_reannotations : bool, optional
        Whether to avoid re-annotating examples that have already been annotated by the annotator. This will decrease
        cost but can be slightly slower if there are no annotations that can be reused.

    saving_path : Path, optional
        Path to save the annotations to. If None, will not save the annotations. If the path already exists it will load
        annotations from there.

    input_keys : tuple of str, optional
        Keys use to distinguish inputs.

    output_keys : tuple of str, optional
        Keys use to distinguish outputs.

    p_label_flip : float, optional
        Probability of flipping the label (ie adds noise by taking a mixture between predicted label and
        2*p_label_flip of independent coin flip). If None, will not flip the label. In AlpacaFarm we use 0.25
        for training. You can set this later on using `set_noise`.
    """

    def __init__(
        self,
        annotators_config: Union[types.AnyPath, list[dict[str, Any]]],
        seed: Optional[int] = None,
        is_avoid_reannotations: bool = True,
        saving_path: Optional[types.AnyPath] = None,
        input_keys: Sequence[str] = ("instruction", "input"),
        output_keys: Sequence[str] = ("output_1", "output_2"),
        p_label_flip: Optional[float] = None,
    ):
        self.seed = seed
        self.is_avoid_reannotations = is_avoid_reannotations
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)
        self.input_output_keys = self.input_keys + self.output_keys
        self.all_keys = self.input_keys + self.output_keys + ["annotator"]
        self.p_label_flip = p_label_flip

        self.annotators = self._initialize_annotators(annotators_config)
        self.saving_path = saving_path
        self.df_annotations = self._load_annotations()

    def annotate_samples(
        self,
        all_outputs: Union[Sequence[dict[str, Any]], pd.DataFrame],
        keys_to_sample_output_2: Optional[Sequence] = None,
    ) -> list[dict[str, Any]]:
        """Sample pairs of outputs from a sequence of examples and anntotate them.

        Parameters
        ----------
        all_outputs : list of dict
            All examples from which we will sample a pair of outputs to annotate. Each dictionary (or row) should
            contain all of `self.input_keys` and `keys_to_sample_output_2` and `"output"`.

        keys_to_sample_output_2 : tuple of str, optional
            Keys to use to sample paired `"output_2"` to compare to the current `"output"` which will become
            `"output_1"`. If `None` it uses `self.input_keys`.
        """
        if not isinstance(all_outputs, pd.DataFrame):
            all_outputs = pd.DataFrame.from_records(all_outputs)

        if keys_to_sample_output_2 is None:
            keys_to_sample_output_2 = self.input_keys
        keys_to_sample_output_2 = list(keys_to_sample_output_2)

        n_pre_drop = len(all_outputs)

        # set output to be unique for each keys_to_sample_output_2
        df_to_annotate = (
            all_outputs.groupby(keys_to_sample_output_2)
            .apply(lambda x: x.drop_duplicates(["output"]))
            .reset_index(drop=True)
            .rename(columns={"output": "output_1"})
        )

        if len(df_to_annotate) != n_pre_drop:
            logging.warning(
                f"""Filtered rows because of duplicate outputs for the same keys_to_sample_output_2=
                {keys_to_sample_output_2}. {n_pre_drop} -> {len(df_to_annotate)}"""
            )

        # sample an output 2 for each output 1 that are different
        # TODO: make sure that you never have output_1 == output_2
        df_to_annotate["output_2"] = df_to_annotate.groupby(list(keys_to_sample_output_2))["output_1"].transform(
            lambda x: ann_utils.random_derangement(x.values, seed=self.seed)
        )

        return self.annotate_pairs(df_to_annotate)

    def annotate_head2head(
        self,
        outputs_1: Union[Sequence[dict[str, Any]], pd.DataFrame],
        outputs_2: Union[Sequence[dict[str, Any]], pd.DataFrame],
        keys_to_merge: Sequence[str] = ("instruction", "input"),
        is_ordered: bool = False,
    ) -> list[dict[str, Any]]:
        """Head-to-head comparison between two sequence of outputs.

        Parameters
        ----------
        outputs_1 : list of dict or dataframe
            Examples to annotate. Each dictionary (or row) should contain all of `keys_to_merge` and `"output"`.
            `"output"` will become `"output_1"`.

        outputs_2 : list of dict or dataframe
            Second  to annotate. Each dictionary (or row) should contain all of `keys_to_merge` and `"output"`.
            `"output"` will become `"output_2"`.

        keys_to_merge : tuple of str, optional
            Keys to use to merge the two sequences of outputs.

        is_ordered : bool, optional
            Whether the two sequences of outputs are in matching order. If not we will be merging based on
            `keys_to_merge`, which means that the outputs can actually be shorter than the inputs (if some outputs
            are not found in the other sequence) or longer (if some outputs are duplicated in both sequences =>
            set cartesian products).

        Returns
        -------
        annotated : list of dict
            The annotated examples. Each dictionary will contain all of `keys_to_merge`, `"output_1"`, `"output_2"`, and
            `"preference"`. Preference will be 0 if output_1 == output_2, 1 if output_1 is preferred, and 2 if output_2
            is preferred.
        """
        keys_to_merge = list(keys_to_merge)

        if not isinstance(outputs_1, pd.DataFrame):
            outputs_1 = pd.DataFrame.from_records(outputs_1)

        if not isinstance(outputs_2, pd.DataFrame):
            outputs_2 = pd.DataFrame.from_records(outputs_2)

        if is_ordered:
            outputs_1 = outputs_1.copy()
            outputs_2 = outputs_2.copy()
            outputs_1["tmp_idx"] = range(len(outputs_1))
            outputs_2["tmp_idx"] = range(len(outputs_1))
            keys_to_merge += ["tmp_idx"]  # add a temporary index to merge on

        df_to_annotate = pd.merge(
            outputs_1[keys_to_merge + ["output"]],
            outputs_2[keys_to_merge + ["output"]],
            on=keys_to_merge,
            suffixes=("_1", "_2"),
        )

        if is_ordered:
            df_to_annotate = df_to_annotate.drop(columns="tmp_idx")
        else:
            # if you are taking the cartesian product, you can have undesired duplicates
            df_to_annotate = df_to_annotate.drop_duplicates()

            if not (len(outputs_1) == len(outputs_2) == len(df_to_annotate)):
                logging.warning(
                    f"""The length of outputs before and after merge are not the same. We have len(outputs_1)==
                    {len(outputs_1)}, len(outputs_2)=={len(outputs_2)}, and len(df_annotated)=={len(df_to_annotate)}. 
                    This means that there are missing examples or duplicates. We are taking a SQL inner join.
                    """
                )

        return self.annotate_pairs(df_to_annotate)

    def annotate_pairs(self, to_annotate: Union[Sequence[dict[str, Any]], pd.DataFrame]) -> list[dict[str, Any]]:
        """Annotates the given examples, which contain both `"output_1"` and `"output_2"` keys.

        Parameters
        ----------
        to_annotate : list of dict or dataframe
            Examples to annotate. Each dictionary (or row) should contain all of `self.input_output_keys`.

        Returns
        -------
        annotated : list of dict
            The annotated examples. Each dictionary will contain all of `self.input_output_keys` and `"preference"`.
            Preference will be 0 if output_1 == output_2, 1 if output_1 is preferred, and 2 if output_2 is preferred.
        """
        if len(to_annotate) == 0:
            return []

        df_to_annotate = self._preprocess(to_annotate)
        df_annotated = self._annotate(df_to_annotate)
        self._store_annotations_(df_annotated)
        annotated = self._postprocess(df_annotated)
        return annotated

    def set_noise(self, p_label_flip: float):
        """Set the noise level for the annotators.

        Parameters
        ----------
        p_label_flip : float, optional
            Probability of flipping the label (ie adds noise by taking a mixture between predicted label and
            2*p_label_flip of independent coin flip). If None, will not flip the label. In AlpacaFarm we use 0.25
            for training.
        """
        self.p_label_flip = p_label_flip

    def _preprocess(self, to_annotate: Union[Sequence[dict[str, Any]], pd.DataFrame]) -> pd.DataFrame:
        """Preprocess the examples to annotate. In particular takes care of filtering unnecessary examples."""

        if not isinstance(to_annotate, pd.DataFrame):
            df_to_annotate = pd.DataFrame.from_records(to_annotate)
        else:
            df_to_annotate = to_annotate.copy()

        if "preference" in df_to_annotate.columns:
            logging.warning("""Preference column is already in the dataframe. We will overwrite it.""")
        df_to_annotate["preference"] = np.nan

        # remove duplicates because you only need to annotate one of them
        df_to_annotate = df_to_annotate.drop_duplicates(subset=self.input_output_keys)

        if self.is_avoid_reannotations:
            # merge the old annotations
            df_to_annotate = self._merge_annotations(df_to_annotate, self.df_annotations)

        idcs_is_same_outputs = df_to_annotate["output_1"] == df_to_annotate["output_2"]
        df_to_annotate.loc[idcs_is_same_outputs, "preference"] = 0

        # adds random noise => avoids annotating examples that will be noised out.
        if self.p_label_flip:
            idcs_no_pref = df_to_annotate["preference"].isna()
            # if you have 25% change of flipping the label, you have 50% chance of selecting random label
            p_noise = self.p_label_flip * 2
            df_to_annotate.loc[idcs_no_pref, "preference"] = df_to_annotate.loc[idcs_no_pref].apply(
                # we add "noisy_label" at the beginning to use ~independent seeds between tasks
                lambda x: ann_utils.random_seeded_choice(  # seed on inputs for reproducibility
                    seed="noisy_label" + x["instruction"] + x["input"],
                    choices=[np.nan, 1, 2],
                    p=[1 - p_noise, self.p_label_flip, self.p_label_flip],
                ),
                axis=1,
            )

        return df_to_annotate

    def _initialize_annotators(
        self, annotators_config: Union[types.AnyPath, dict[str, dict[str, Any]]]
    ) -> dict[str, Callable]:
        """Load all the configs and prompts if necessary."""
        if not isinstance(annotators_config, dict):
            with open(CURRENT_DIR / annotators_config, "r") as stream:
                try:
                    annotators_config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    logging.exception(exc)

        return {
            name: SinglePairwiseAutoAnnotator(**annotator_config)
            for name, annotator_config in annotators_config.items()
        }

    def _annotate(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """Annotate the examples."""

        # set the annotater for each example
        df_to_annotate["annotator"] = df_to_annotate.apply(
            lambda x: ann_utils.random_seeded_choice(
                # we add "annotator" at the beginning to not use the same seed for all tasks
                seed="annotator" + x["instruction"] + x["input"],
                choices=list(self.annotators.keys()),
            ),
            axis=1,
        )

        df_annotated = df_to_annotate
        for annotator in self.annotators.keys():
            # only annotate examples that have not been annotated yet
            curr_idcs = (df_annotated["annotator"] == annotator) & df_annotated["preference"].isna()

            # actual annotation
            curr_annotated = self.annotators[annotator](df_annotated[curr_idcs])

            df_annotated = self._merge_annotations(df_annotated, curr_annotated)

        return df_annotated

    def _store_annotations_(self, df_annotated: pd.DataFrame):
        """Store all the annotations in memory and potentially to json."""
        if self.df_annotations is None:
            self.df_annotations = df_annotated
        else:
            self.df_annotations = pd.concat([self.df_annotations, df_annotated], axis=0, ignore_index=True)

        if self.saving_path is not None:
            self.df_annotations.to_json(self.saving_path, orient="records", indent=2)

    def _load_annotations(self) -> Optional[pd.DataFrame]:
        """Load all the annotations from json."""
        df_annotations = None
        if self.saving_path is not None and self.saving_path.exists():
            df_annotations = pd.read_json(self.saving_path)
        return df_annotations

    def _merge_annotations(self, df_to_annotate: pd.DataFrame, df_partially_annotated: pd.DataFrame) -> pd.DataFrame:
        """Merge (partial) annotations with the original df to keep the same order and avoid duplicates annotations."""
        if df_partially_annotated is None or df_partially_annotated.empty:
            return df_to_annotate

        df_to_annotate = df_to_annotate.merge(
            df_partially_annotated[self.all_keys + ["preference"]],
            on=self.all_keys,
            how="left",
            suffixes=("_old", "_new"),
        )
        df_to_annotate["preference"] = df_to_annotate["preference_old"].fillna(df_to_annotate["preference_new"])
        df_to_annotate = df_to_annotate.drop(columns=["preference_old", "preference_new"])
        return df_to_annotate

    def _postprocess(self, df_annotated: pd.DataFrame) -> list[dict[str, Any]]:
        """Return all the available annotations as a list of dict."""
        df_annotated = df_annotated[~df_annotated["preference"].isna()]
        annotated = df_annotated.to_dict(orient="records")
        return annotated


class SinglePairwiseAutoAnnotator:
    """A helper class for a single auto annotators.

    Parameters
    ----------
    prompt_templates : dict
        A dictionary of prompts that will be given to `fn_prompter`.

    outputs_to_match : dict
        A dictionary of outputs to match from the completions. The values should be a regex pattern that should
        be matched, the keys should be the corresponding preference value. For each completion, the number of patterns
        that are matched should be equal to the batch_size if not we set all the preferences in that batch to NaN.

    fn_decoder : callable or str
        Function in `decoders.py` to use for decoding the output.

    decoder_kwargs : dict
        kwargs for fn_decoder. E.g. model_name, max_tokens, temperature, top_p, top_k, stop_seq.

    is_randomize_output_order : bool
        Whether to randomize output_1, output_2 when formatting.

    is_shuffle : bool
        Whether to shuffle the order of the examples before making the prompt. Useful if batch_size > 1.

    seed : int
        Seed for randomization.

    batch_size : int
        Number of examples that will be added in a single prompt.
    """

    def __init__(
        self,
        prompt_templates: dict[str, str],
        outputs_to_match: dict[Any, str],
        fn_decoder: Union[Callable, str],
        decoder_kwargs: dict[str, Any],
        is_randomize_output_order: bool = True,
        is_shuffle: bool = True,
        seed: Optional[int] = None,
        batch_size: int = 1,
    ):
        self.prompt_templates = {
            k: ann_utils.read_or_return(CURRENT_DIR / prompt) for k, prompt in prompt_templates.items()
        }
        self.outputs_to_match = {k: re.compile(v) for k, v in outputs_to_match.items()}
        self.is_randomize_output_order = is_randomize_output_order
        self.fn_decoder = getattr(decoders, fn_decoder, fn_decoder)
        self.decoder_kwargs = decoder_kwargs
        self.seed = seed
        self.is_shuffle = is_shuffle
        self.batch_size = batch_size

    def __call__(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """Annotates the given examples.

        Parameters
        ----------
        df_to_annotate : pd.DataFrame
            Examples to annotate
        """
        df_to_annotate = df_to_annotate.copy()  # avoid in place modifications

        if df_to_annotate.empty:
            df_to_annotate["preference"] = []
            return df_to_annotate

        df_to_annotate = self.preprocess(df_to_annotate)

        # prompts and completions here will not be the same length as the dataframe due to batching
        prompts, df_to_annotate = self.make_prompts(df_to_annotate=df_to_annotate)

        completions = self.fn_decoder(prompts=prompts, **self.decoder_kwargs)

        df_to_annotate["preference"] = self.parse_completions(completions=completions)

        df_annotated = self.postprocess(df_to_annotate)

        return df_annotated

    def preprocess(self, df_to_annotate: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the examples before annotating. In particular, takes care of all the randomization."""

        if self.is_randomize_output_order:
            # randomize order of output_1, output_2 base on inputs
            df_to_annotate["is_switched_outputs"] = df_to_annotate.apply(
                # we add "is_switched_outputs" at the beginning to not use the same seed for all tasks
                lambda x: ann_utils.random_seeded_choice(
                    seed="is_switched_outputs" + x["instruction"] + x["input"], choices=[False, True]
                ),
                axis=1,
            )
            df_to_annotate = ann_utils.shuffle_pairwise_preferences(
                df_to_annotate, df_to_annotate["is_switched_outputs"]
            )

        if self.is_shuffle:
            df_to_annotate = df_to_annotate.sample(frac=1, random_state=self.seed)

        return df_to_annotate

    def make_prompts(self, df_to_annotate: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
        """Make all the prompts for the given examples.

        Parameters
        ----------
        df_to_annotate : pd.DataFrame
            Examples to annotate

        Returns
        -------
        prompts : list[str]
            Formatted prompts for the given examples.

        df_to_annotate : pd.DataFrame
            Examples to annotate in the same order as prompts.
        """
        arr_is_inputs = (df_to_annotate["input"] != "") & (df_to_annotate["input"].notnull())
        df_with_inputs = df_to_annotate[arr_is_inputs]
        df_without_inputs = df_to_annotate[~arr_is_inputs]

        prompts, df = ann_utils.make_prompts(
            df_without_inputs, self.prompt_templates["without_inputs"], batch_size=self.batch_size
        )
        if arr_is_inputs.any():
            prompts_i, df_i = ann_utils.make_prompts(
                df_with_inputs, self.prompt_templates["with_inputs"], batch_size=self.batch_size
            )
            prompts += prompts_i
            df = pd.concat([df, df_i], axis=0, ignore_index=True)

        return prompts, df

    def parse_completions(self, completions: list[str]) -> list[int]:
        """Converts the completions into annotations."""
        all_preferences = []
        for completion in completions:
            # use a regex to match all outputs on a line. Assumes that there is at most one output to match per line
            batch_preferences = ann_utils.parse_batched_completion(completion, self.outputs_to_match)
            if len(batch_preferences) != self.batch_size:
                logging.warning(
                    f"""Found {len(batch_preferences)} preferences in {completion} but expected {self.batch_size}.
                    We are setting all preferences to np.nan."""
                )
                batch_preferences = [np.nan] * self.batch_size
            all_preferences += batch_preferences
        return all_preferences

    def postprocess(self, df_annotated: pd.DataFrame) -> pd.DataFrame:
        """Postprocess the annotated examples."""

        # remove padding examples when using batch_size > 1
        df_annotated = df_annotated.query("is_padding == False").drop(columns=["is_padding"])

        arr_is_na = df_annotated["preference"].isna()
        if arr_is_na.any():
            logging.warning(
                f"{arr_is_na.sum().item()} samples had no auto annotation. We are filtering them for now. "
                f"If you are using chain of thought it might be that max_tokens limit is too low. "
            )
            df_annotated = df_annotated[~arr_is_na]

        assert set(df_annotated["preference"].unique().tolist()) <= {1, 2}

        if self.is_randomize_output_order:
            # unshuffles output 1 and output 2. For binary preference, unshuffling is equivalent to reshuffling
            df_annotated = ann_utils.shuffle_pairwise_preferences(df_annotated, df_annotated["is_switched_outputs"])
            df_annotated = df_annotated.drop(columns=["is_switched_outputs"])

        return df_annotated
