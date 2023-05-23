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

import logging
from typing import Sequence, Union

import pandas as pd


def head2head_to_metrics(preferences: Union[pd.Series, Sequence]) -> dict[str, int]:
    """Extract head2head metrics (n_wins, n_counts, win_rate) from a sequence preference.
    This assumes that the preference is encoded as 0 for draw, 1 for base win, 2 when the model to compare wins.
    """
    if not isinstance(preferences, pd.Series):
        series_preferences = pd.Series(preferences)
    else:
        series_preferences = preferences.copy()

    is_preference = series_preferences.isin([0, 1, 2])
    n_not_pair = sum(~is_preference)
    if n_not_pair > 0:
        logging.info(f"drop {n_not_pair} outputs that are not[0, 1, 2]")
    series_preferences = series_preferences[is_preference].astype(int).copy()

    n_draws = (series_preferences == 0).sum()
    n_wins_base = (series_preferences == 1).sum()
    n_wins = (series_preferences == 2).sum()
    n_total = len(series_preferences)
    series_preferences[series_preferences == 0] = 1.5
    series_preferences -= 1
    win_rate = series_preferences.mean()

    return dict(
        win_rate=win_rate * 100,
        standard_error=series_preferences.sem() * 100,
        n_wins=n_wins,
        n_wins_base=n_wins_base,
        n_draws=n_draws,
        n_total=n_total,
    )
