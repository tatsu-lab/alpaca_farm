"""
All functions that are used to convert between different types of annotations. There are 4 main types of annotations:
- reward: absolute annotation that is a continuous score between 0 and 1, where 1 is max.
- likert: absolute annotation that is an integer score between 1 and K.
- preference: relative annotation that chooses the best between N options, it is an integer between 1 and N.
- ranking: relative annotation that ranks the N options, it an ordered sequence of integers between 0 and N-1.

Every conversion function should take as input a dataframe of annotations and return a dataframe of annotations.
The input dataframe is expected to have all the important columns of corresponding table in the database. In particular:
- reward: contain a `reward` column
- likert: contain a `likert_score` column and K `likert_score_{i}_logprob` columns, where K number of likert options.
- ordinal: contain a `preference` column, N*K `preference_{i}_logprob` columns, and N `output_{i}` columns,
  where N is the number of outputs and `K` is the number of ordinal classes per outputs (eg if preference vs slight
  preference => K=2).
- preference: contain a `preference` column, N `preference_{i}_logprob` columns, and N `output_{i}` columns, where N
is the number of outputs.
- ranking: contain N `ranking_output_{i}` columns, and N `output_{i}` columns, where N
is the number of outputs.
"""
import copy
import dataclasses
import logging
import math

import numpy as np
import pandas as pd
import scipy

# TODO: @yann simplify converters now that you know it's wither 2 or 4 choices


@dataclasses.dataclass
class LogprobsColumnPatch(object):
    num_categories: int = 2

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        for i in range(self.num_categories):
            df[f"preference_{i}_logprob"] = np.nan
        return df


# TODO in terms of naming in the DB, maybe this should be "ordinal_preference" instead of "preference"
#  and the table should be called "ordinal_preference" instead of "ordered_preference"
def convert_ordinal_to_preference(df_ordinal: pd.DataFrame) -> pd.DataFrame:
    """Convert ordinal preference annotations to preference annotations. By merging multiple subcategories together,
    eg A/a/b/B into A/B, or AA/A/a/b/B/BB into A/B.

    Parameters
    ----------
    df_ordinal : pd.DataFrame
        Dataframe of ordinal preference annotations.

    Returns
    -------
    pd.DataFrame
        Dataframe of preference annotations.

    Examples
    --------
    >>> import pandas as pd
    >>> df_ordinal = pd.DataFrame(dict(preference=[1,2,3,1],
    ...     output_1=["a", "b", "c", "d"],
    ...     output_2=["b", "c", "d", "a"],
    ...     preference_1_logprob=[-0, -0.5, -2.0, np.nan],
    ...     preference_2_logprob=[-0.5, -0, -1.5, np.nan],
    ...     preference_3_logprob=[-2, -3, -0.5, np.nan],
    ...     preference_4_logprob=[-float("Inf"), -1.1, -1.1, np.nan]))
    >>> convert_ordinal_to_preference(df_ordinal)
       preference output_1 output_2  preference_1_logprob  preference_2_logprob
    0           1        a        b              -0.21907             -0.991734
    1           1        b        c              -0.21907             -0.644560
    2           2        c        d              -1.71907             -0.879885
    3           1        d        a                   NaN                   NaN
    """
    df_ordinal = df_ordinal.copy()  # ensure not in place

    df_ordinal, n_outputs, k_ordinal = _prepare_df_ordinal(df_ordinal)

    # aggregate ordinal preference
    non_ordinal_preference = (df_ordinal["preference"] - 1) // k_ordinal
    df_ordinal["preference"] = non_ordinal_preference + 1  # preference is 1 indexed

    # aggregate log prob
    for i in range(1, n_outputs + 1):
        df_ordinal[f"preference_{i}_logprob"] = log_mean_exp(
            [df_ordinal[f"preference_{i + j}_logprob"] for j in range(k_ordinal)], axis=0
        )

    # removes unused columns
    df_preference = df_ordinal.drop(
        columns=[f"preference_{i}_logprob" for i in range(n_outputs + 1, 1 + (n_outputs * k_ordinal))]
    )

    _prepare_df_preference(df_preference)  # validates that it is now a preference table

    return df_preference


def convert_likert_to_reward(
    df_likert: pd.DataFrame,
    is_min_max_scale: bool = True,
    reward_key: str = "reward",
    reward_model_name_key: str = "reward_model",
    reward_model_name: str = "converted_likert",
) -> pd.DataFrame:
    """Convert likert annotations to reward annotations.

    Parameters
    ----------
    df_likert : pd.DataFrame
        Dataframe of likert annotations.

    is_min_max_scale : bool
        Whether to min max scale reward to [0,1] or not.

    reward_key: str
        Name for the reward column.

    reward_model_name_key: str
        Name for the reward model column.

    reward_model_name : str
        Name of the reward model that we want to add to the columns.

    Returns
    -------
    pd.DataFrame
        Dataframe of reward annotations.

    Examples
    --------
    >>> import pandas as pd
    >>> df_likert = pd.DataFrame(dict(likert_score=[1,1,2],
    ...     likert_score_1_logprob=[-0, -0.5, -2.0],
    ...     likert_score_2_logprob=[-float("Inf"), -1.1, -1.1]))
    >>> convert_likert_to_reward(df_likert)
       likert_score  likert_score_1_logprob  likert_score_2_logprob    reward
    0             1                     0.0                    -inf  0.500000
    1             1                    -0.5                    -1.1  0.677172
    2             2                    -2.0                    -1.1  0.855475
    """
    df_likert = df_likert.copy()  # ensure not in place
    df_likert, K = _prepare_df_likert(df_likert)

    # computed expected value from log probabilities
    log_probs = df_likert[[f"likert_score_{i}_logprob" for i in range(1, K + 1)]].values
    probs = scipy.special.softmax(log_probs, axis=1)
    expected_likert = probs @ np.arange(1, K + 1)

    df_likert[reward_key] = expected_likert

    if is_min_max_scale:
        # reward should be between 0 and 1 => min max scaling
        df_likert[reward_key] = (df_likert[reward_key] - 1) / (K - 1)

    df_likert[reward_model_name_key] = reward_model_name

    return df_likert


def convert_reward_to_preference(df_reward: pd.DataFrame, df_preference_to_annotate: pd.DataFrame) -> pd.DataFrame:
    """Convert reward annotations to preference annotations.

    Parameters
    ----------
    df_reward : pd.DataFrame
        Dataframe of reward annotations. Should contain input_id, output, and reward_model column. The reward_model
        column should contain a single value.

    df_preference_to_annotate : pd.DataFrame
        Dataframe of empty preference annotations that should be filled. Should contain input_id.

    Returns
    -------
    pd.DataFrame
        df_preference with preference annotations filled.

    Examples
    --------
    >>> import pandas as pd; import numpy as np
    >>> df_reward = pd.DataFrame(dict(input_id=[1,1,1], reward=[0.1, 0.2, 0.3],
    ...                               output=["(a)","(b)","(c)"], reward_model=["r", "r", "r"]))
    >>> df_reward
       input_id  reward output reward_model
    0         1     0.1    (a)            r
    1         1     0.2    (b)            r
    2         1     0.3    (c)            r
    >>> df_preference = pd.DataFrame(dict(input_id=[1,1,1], output_1=["(a)","(b)","(a)"], output_2=["(c)","(a)","(d)"],
    ...                 preference_1_logprob=[np.nan]*3, preference_2_logprob=[np.nan]*3, preference=[np.nan]*3))
    >>> df_preference
       input_id output_1 output_2  preference_1_logprob  preference_2_logprob  preference
    0         1      (a)      (c)                   NaN                   NaN        NaN
    1         1      (b)      (a)                   NaN                   NaN        NaN
    2         1      (a)      (d)                   NaN                   NaN        NaN
    >>> convert_reward_to_preference(df_reward, df_preference)
       input_id output_1 output_2  preference_1_logprob  preference_2_logprob  preference  annotator
    0         1      (a)      (c)                   NaN                   NaN           2          r
    1         1      (b)      (a)                   NaN                   NaN           1          r
    2         1      (a)      (d)                   NaN                   NaN         NaN          r
    """
    df_reward = df_reward.copy()  # ensure not in place
    df_preference_to_annotate = df_preference_to_annotate.copy()  # ensure not in place

    df_reward = _prepare_df_reward(df_reward)
    df_preference, n_options = _prepare_df_preference(df_preference_to_annotate)

    reward_models = df_reward["reward_model"].unique()
    assert len(reward_models) == 1, "df_reward should contain only one reward model"
    rewards = df_reward.set_index(["input_id", "output", "reward_model"])["reward"]

    all_rewards = []
    for i in range(1, n_options + 1):
        idcs = df_preference[["input_id", f"output_{i}"]].copy()
        idcs["reward_model"] = reward_models[0]
        idcs = pd.MultiIndex.from_frame(idcs)
        missing_idcs = idcs.difference(rewards.index)
        # allow for missing values => will get nan reward => nan preference
        rewards = pd.concat([rewards, pd.Series(np.nan, index=missing_idcs)])
        all_rewards.append(rewards.loc[idcs].values)

    all_rewards = np.stack(all_rewards, axis=1)
    any_nan = np.isnan(all_rewards).any(axis=1)
    # adding 1 because preference is 1-indexed
    df_preference["preference"] = np.where(any_nan, np.nan, np.argmax(all_rewards, axis=1) + 1)
    df_preference["annotator"] = reward_models[0]

    return df_preference


###### HELPER FUNCTIONS ######


def log_mean_exp(arrays, **kwargs):
    """Computes mean in log space."""
    return scipy.special.logsumexp(arrays, **kwargs) - np.log(len(arrays))


def _prepare_df_ordinal(df_ordinal: pd.DataFrame, is_allow_nan: bool = True) -> tuple[pd.DataFrame, int, int]:
    """Validate ordinal annotations and returns the number of outputs and categories.

    Parameters
    ----------
    df_ordinal : pd.DataFrame
        Dataframe of ordinal annotations.

    is_allow_nan : bool, optional
        Whether to allow NaN values in the ordinal annotations.

    Returns
    -------
    df_ordinal : pd.DataFrame
        Validated dataframe of ordinal annotations.

    n_outputs : int
        Number of outputs that you are comparing.

    k_ordinal : int
        Number of ordinal categories per outputs that you are comparing.

    Raises
    ------
    ValueError
        If the preference annotations are not valid.

    """
    # find all columns that match preference_{i}_logprob and extract {i} using regex
    columns_log_prob = df_ordinal.filter(regex="preference_\d+_logprob").columns
    preference_options = set([int(col.split("_")[-2]) for col in columns_log_prob])

    # convert to float in the case where all None
    df_ordinal[columns_log_prob] = df_ordinal[columns_log_prob].astype(float)

    # find all outputs
    columns_outputs = df_ordinal.filter(regex="output_\d+").columns
    output_options = set([int(col.split("_")[-1]) for col in columns_outputs])

    n_outputs = len(output_options)

    if n_outputs == 0:
        raise ValueError("No output options found.")

    k_ordinal = len(preference_options) // n_outputs

    if preference_options != set(range(1, (n_outputs * k_ordinal) + 1)):
        raise ValueError(f"Output options {preference_options} are not consecutive integers from 1 to {n_outputs}.")

    preference_options.add(-1)  # add -1 for the no preference option
    _assert_log_prob_df(df_ordinal[columns_log_prob], is_allow_nan)
    _assert_options_df(df_ordinal, is_allow_nan, "preference", preference_options)

    if k_ordinal == 1:
        logging.warning("k_ordinal=1, you are likely using non ordinal preference annotations.")

    return df_ordinal, n_outputs, k_ordinal


def _prepare_df_likert(df_likert: pd.DataFrame, is_allow_nan: bool = True) -> tuple[pd.DataFrame, int]:
    """Validate likert annotations.

    Parameters
    ----------
    df_likert : pd.DataFrame
        Dataframe of likert annotations.

    is_allow_nan : bool, optional
        Whether to allow NaN values in the likert annotations.

    Returns
    -------
    df_likert : pd.DataFrame
        Validated dataframe of likert annotations.

    K : int
        Number of likert options.

    Raises
    ------
    ValueError
        If the likert annotations are not valid.
    """

    # find all columns that match likert_{i}_logprob and extract {i} using regex
    columns_log_prob = df_likert.filter(regex="likert_score_\d+_logprob").columns
    likert_options = set([int(col.split("_")[-2]) for col in columns_log_prob])
    K = len(likert_options)

    # convert to float in the case where all None
    df_likert[columns_log_prob] = df_likert[columns_log_prob].astype(float)

    if K == 0:
        raise ValueError("No likert options found.")

    if likert_options != set(range(1, K + 1)):
        raise ValueError(f"Likert options {likert_options} are not consecutive integers from 1 to {K}.")

    _assert_options_df(df_likert, is_allow_nan, "likert_score", likert_options)
    _assert_log_prob_df(df_likert[columns_log_prob], is_allow_nan)

    return df_likert, K


def _prepare_df_reward(df_reward: pd.DataFrame, is_allow_nan: bool = True) -> pd.DataFrame:
    """Validate reward annotations.

    Parameters
    ----------
    df_reward : pd.DataFrame
        Dataframe of reward annotations.

    is_allow_nan : bool, optional
        Whether to allow NaN values in the reward annotations.

    Returns
    -------
    df_reward : pd.DataFrame
        Dataframe of reward annotations.

    Raises
    ------
    ValueError
        If the reward annotations are not valid.
    """

    is_reward_valid = 0 <= df_reward["reward"]
    is_reward_valid &= df_reward["reward"] <= 1
    if is_allow_nan:
        is_reward_valid |= df_reward["reward"].isna()

    if not is_reward_valid.all():
        raise ValueError("Reward not between 0 and 1.")

    return df_reward


def _prepare_df_preference(df_preference: pd.DataFrame, is_allow_nan: bool = True) -> tuple[pd.DataFrame, int]:
    """Validate preference annotations.

    Parameters
    ----------
    df_preference : pd.DataFrame
        Dataframe of preference annotations.

    is_allow_nan : bool, optional
        Whether to allow NaN values in the preference annotations.

    Returns
    -------
    df_preference : pd.DataFrame
        Dataframe of preference annotations.

    N : int
        Number of preference options.

    Raises
    ------
    ValueError
        If the preference annotations are not valid.
    """

    # find all columns that match preference_{i}_logprob and extract {i} using regex
    columns_log_prob = df_preference.filter(regex="preference_\d+_logprob").columns
    preference_options = set([int(col.split("_")[-2]) for col in columns_log_prob])

    # find all outputs
    columns_outputs = df_preference.filter(regex="output_\d+").columns
    output_options = set([int(col.split("_")[-1]) for col in columns_outputs])

    if preference_options != output_options:
        raise ValueError(f"Log prob preferences {preference_options} and outputs {output_options} are not the same.")

    N = len(preference_options)

    if N == 0:
        raise ValueError("No output options found.")

    if preference_options != set(range(1, N + 1)):
        raise ValueError(f"Output options {preference_options} are not consecutive integers from 1 to {N}.")

    output_options.add(-1)  # add -1 for the no preference option
    _assert_log_prob_df(df_preference[columns_log_prob], is_allow_nan)
    _assert_options_df(df_preference, is_allow_nan, "preference", output_options)

    return df_preference, N


def _assert_log_prob_df(df: pd.DataFrame, is_allow_nan: bool):
    """Assert that the log probs in dataframe are between -inf and 0."""
    is_log_prob_valid = -float("inf") <= df
    is_log_prob_valid &= df <= 0
    if is_allow_nan:
        is_log_prob_valid |= df.isna()

    if not is_log_prob_valid.all(None):
        raise ValueError("Log probs are not between -inf and 0.")


def _assert_options_df(df: pd.DataFrame, is_allow_nan: bool, col: str, options: set):
    """Assert that values in a column are in `options`."""

    # check that preference is in [1, K]
    if is_allow_nan:
        options = copy.deepcopy(options)
        options.add(float("nan"))

    values = set(df[col].unique())
    if not _issubset_nansafe(values, options):
        raise ValueError(f"{col} takes values {values} but restricted to {options}.")


def _issubset_nansafe(s1: set, s2: set) -> bool:
    # need to have same nan for issubset to work
    s1 = {np.nan if math.isnan(s) else s for s in s1}
    s2 = {np.nan if math.isnan(s) else s for s in s2}
    return s1.issubset(s2)
