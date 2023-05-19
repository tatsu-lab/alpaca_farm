import logging

import numpy as np
import pandas as pd
from typing import Any, Callable, Optional, Sequence, Union

import yaml

from ..types import AnyPath
from . import prompters, decoders, parsers, utils as ann_utils


class PairwiseAutoAnnotator:
    """Class for a pool of annotators. This should be the user

    Parameters
    ----------
    annotator_config : Path or list of dict
        A dictionary or path to a yaml file containing the configuration for the pool of annotators. The keys in the
        fist dictionary should be the annotator's name, and the value should be a dictionary of the annotator's
        configuration which should have the following keys:
        - prompt_templates (dict): a dictionary of prompts or path to the prompts. The dictionary of (loaded) prompts
            will be given to `fn_prompter`.
        - fn_prompter (str): function in `prompters.py` to use for formatting the prompt.
        - fn_decoder (str): function in `decoders.py` to use for decoding the output.
        - fn_parser (str): function in `parsers.py` to use for parsing the output. E.g.
        - prompter_kwargs (dict): kwargs for the formatter. E.g. prompt_with_inputs, prompt_without_inputs, is_randomize_output_order
        - decoder_kwargs (dict): kwargs for fn_decode. E.g. model_name, max_completions_tokens
        - parser_kwargs (dict): kwargs for fn_parse. E.g. batch size
        - other secondary kwargs to SinglePairwiseAutoAnnotator

    seed : int, optional
        Seed for the random number generator.

    is_avoid_reannotations : bool, optional
        Whether to avoid re-annotating examples that have already been annotated by the annotator. This will decrease
        cost but can be slightly slower if there are no annotations that can be reused.

    saving_path : Path, optional
        Path to save the annotations to. If None, will not save the annotations. If the path already exists it will load
        annotations from there.

    keys : tuple of str, optional
        Keys use to find examples that were already annotated. Only if `is_avoid_reannotations`.
    """

    def __init__(
        self,
        annotators_config: Union[AnyPath, list[dict[str, Any]]],
        seed: Optional[int] = None,
        is_avoid_reannotations: bool = False,
        saving_path: Optional[AnyPath] = None,
        keys: Sequence[str] = (
            "instruction",
            "input",
            "annotator",
            "output_1",
            "output_2",
        ),
    ):
        self.seed = seed
        self.is_avoid_reannotations = is_avoid_reannotations
        self.keys = keys

        self.annotators = self.initialize_annotators(annotators_config)
        self.saving_path = saving_path
        self.df_annotations = self.load_annotations()

    def initialize_annotators(
        self, annotators_config: Union[AnyPath, dict[str, dict[str, Any]]]
    ) -> dict[str, dict[str, Any]]:
        """Load all the configs and prompts if necessary."""
        if not isinstance(annotators_config, dict):
            with open(annotators_config, "r") as stream:
                try:
                    annotators_config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    logging.exception(exc)

        return {
            name: SinglePairwiseAutoAnnotator(**annotator_config)
            for name, annotator_config in annotators_config.items()
        }

    def __call__(
        self, to_annotate: list[dict[str, Any]], is_save: bool = True, np=None
    ) -> list[dict[str, Any]]:
        """Annotates the given examples.

        Parameters
        ----------
        to_annotate : list of dict
            Examples to annotate

        is_save : bool, optional
            Whether to save the annotations to file. This is especially useful if you are using is_avoid_reannotations
            as it will ensure that the annotations are saved and can be reused.

        Returns
        -------
        out : list of dict
            The annotated examples.
        """
        # 1. PREPROCESSING
        df = pd.from_dict(to_annotate)

        if self.is_avoid_reannotations:
            df = self.remove_already_annotated(df)

        idcs_is_same_outputs = df["output_1"] == df["output_2"]
        df.loc[idcs_is_same_outputs, "preference"] = 0

        # 2. ANNOTATING
        df["annotator"] = df.apply(
            lambda x: ann_utils.random_seeded_choice(
                seed=x["instruction"] + x["input"], choices=self.annotators.keys()
            ),
            axis=1,
        )
        for annotator in self.annotators.keys():
            curr_idcs = (df["annotator"] == annotator) & ~idcs_is_same_outputs
            df.loc[curr_idcs, "preference"] = self.annotators[annotator](df[curr_idcs])

        # 3. STORING
        self.df_annotations = pd.concat([self.df_annotations, df], axis=0)

        if is_save:
            self.save_annotations()

        # POSTPROCESSING
        if self.is_avoid_reannotations:
            df = self.add_already_annotated(df, to_annotate)

        return df.to_dict(orient="records")

    def save_annotations(self):
        """Save all the annotations to json."""
        self.df_annotations.to_json(self.saving_path)

    def load_annotations(self) -> list[dict[str, Any]]:
        """Load all the annotations from json."""
        self.df_annotations = pd.read_json(self.saving_path)

    def remove_already_annotated(
        self, to_annotate: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove all examples that are already annotated to avoid unnecessary cost."""
        pass

    def add_already_annotated(
        self, annotated: list[dict[str, Any]], to_annotate: list[dict[str, Any]]
    ):
        """Return all the annotations including those that were already annotated."""
        pass


class SinglePairwiseAutoAnnotator:
    """A helper class for a single auto annotators.

    Parameters
    ----------
    annotator : str
        The name of the annotator.

    prompt_templates : dict
        A dictionary of prompts that will be given to `fn_prompter`.

    fn_prompter : callable or str
        Function in `prompters.py` to use for formatting the prompt.

    fn_decoder : callable or str
        Function in `decoders.py` to use for decoding the output.

    fn_parser : callable or str
        Function in `parsers.py` to use for parsing the output.

    prompter_kwargs : dict
        kwargs for fn_prompter.

    decoder_kwargs : dict
        kwargs for fn_decoder. E.g. model_name, max_tokens, temperature, top_p, top_k, stop_seq.

    parser_kwargs : dict
        kwargs for fn_parser. E.g. is_remove_input, is_remove_output, is_remove_prefix, is_remove_suffix.

    is_randomize_output_order : bool
        Whether to randomize output_1, output_2 when formatting.
    """

    def __init__(
        self,
        annotator: str,
        prompt_templates: dict[str, str],
        fn_prompter: Union[Callable, str],
        fn_decoder: Union[Callable, str],
        fn_parser: Union[Callable, str],
        prompter_kwargs: dict[str, Any],
        decoder_kwargs: dict[str, Any],
        parser_kwargs: dict[str, Any],
        is_randomize_output_order: bool = True,
    ):
        self.prompt_templates = {
            k: ann_utils.read_or_return(prompt)
            for k, prompt in prompt_templates.items()
        }
        self.is_randomize_output_order = is_randomize_output_order
        self.fn_prompter = getattr(prompters, fn_prompter, fn_prompter)
        self.fn_decoder = getattr(decoders, fn_decoder, fn_decoder)
        self.fn_parser = getattr(parsers, fn_parser, fn_parser)
        self.prompter_kwargs = prompter_kwargs
        self.decoder_kwargs = decoder_kwargs
        self.parser_kwargs = parser_kwargs
        self.annotator = annotator

    def __call__(self, df_to_annotate: pd.DataFrame) -> list[int]:
        """Annotates the given examples.

        Parameters
        ----------
        df_to_annotate : pd.DataFrame
            Examples to annotate
        """
        if df_to_annotate.empty:
            return []

        if self.is_randomize_output_order:
            # randomize order of output_1, output_2 base on inputs
            arr_is_shuffle = df_to_annotate.apply(
                lambda x: ann_utils.random_seeded_choice(
                    seed=x["instruction"] + x["input"], choices=[False, True]
                ),
            )
            col_1 = df_to_annotate["output_1"].copy()
            col_2 = df_to_annotate["output_2"].copy()
            df_to_annotate["output_1"] = np.where(arr_is_shuffle, col_2, col_1)
            df_to_annotate["output_2"] = np.where(arr_is_shuffle, col_1, col_2)

        prompts = self.fn_prompter(
            to_annotate=df_to_annotate.to_dict(),
            **self.prompt_templates, **self.prompter_kwargs
        )

        completions = self.fn_decoder(prompts=prompts, **self.decoder_kwargs)

        annotations = self.fn_parser(completions=completions, **self.parser_kwargs)

        # unshuffles output 1 and output 2. For binary preference, unshuffling is equivalent to reshuffling
        if self.is_randomize_output_order:
            annotations = [3-a if is_shuffle else a
                           for a, is_shuffle in zip(annotations, arr_is_shuffle)]

        return annotations
