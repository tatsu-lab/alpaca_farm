
import numpy as np

import alpaca_eval.annotators as eval_annotators
import alpaca_eval.utils as ae_utils
import alpaca_eval.main as ae_main

__all__ = ["alpaca_leaderboard", "PairwiseAutoAnnotator"]

#! Important:
# The leaderboard is different from teh paper because Davinci003 is depreciated. We now use AlpacaEval1 to
# evaluate the models.AlpacaEval2 is cheaper and has will have more evaluated models, but the baseline is too stong
# => models from AlpacaFarm will have very low scores.
def alpaca_leaderboard(
        *args,
        **kwargs,
):
    return ae_main.evaluate(*args,
                           leaderboard_mode_to_print=["alpaca-farm-ppo-human", "alpaca-7b", "text_davinci_001",
                                                      "gpt35_turbo_instruct", "alpaca-farm-ppo-sim-gpt4-20k"],
                           **kwargs)



class PairwiseAutoAnnotator(eval_annotators.PairwiseAnnotator):
    def __init__(self, *args, input_keys=("input", "instruction"), **kwargs):
        super().__init__(*args, input_keys=input_keys, **kwargs)
    def __call__(self, to_annotate, **decoding_kwargs):
        df_to_annotate = ae_utils.convert_to_dataframe(to_annotate)
        # merge input and instruction column into one. but only if input is not empty
        merged_col = df_to_annotate["instruction"] + "\n\n" + df_to_annotate["input"]
        df_to_annotate["instruction"] = np.where(df_to_annotate["input"] != "",
                                                 merged_col,
                                                 df_to_annotate["instruction"])
        return super().__call__(df_to_annotate, **decoding_kwargs)
