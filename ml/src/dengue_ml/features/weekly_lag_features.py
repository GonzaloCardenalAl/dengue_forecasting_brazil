import numpy as np
import pandas as pd

from dengue_ml.config import CITY_COL


def add_weekly_lag_features(
    df: pd.DataFrame,
    value_col: str,
    prefix: str,
    n_weeks: int,
    log_transform: bool = False,
    group_col: str = CITY_COL,
) -> pd.DataFrame:
    """
    Attach the last `n_weeks` values of value_col strictly before each row
    (t-1 = most recent, t-n_weeks = oldest), plus week-over-week
    growth/acceleration, via direct groupby shifts on whatever df is passed
    in. During the autoregressive forecast loop, df grows each iteration to
    include the model's own just-predicted weeks, so these lags automatically
    track the model's evolving trajectory instead of a frozen historical
    snapshot -- this replaces an earlier cross-reference mechanism that read
    a separate, static weekly table and could never see those predicted
    weeks. Assumes df is already sorted by date within each group, same
    convention as target_lag_features.py.
    """
    df = df.copy()
    s = np.log1p(df[value_col]) if log_transform else df[value_col]
    g = s.groupby(df[group_col], sort=False)

    for i in range(1, n_weeks + 1):
        df[f"{prefix}_week_t-{i}"] = g.shift(i)

    t1, t4, t8 = g.shift(1), g.shift(4), g.shift(8)
    growth_4w = (t1 - t4) / 3
    growth_8w = (t1 - t8) / 7
    df[f"{prefix}_weekly_growth_avg_4w"] = growth_4w
    df[f"{prefix}_weekly_growth_avg_8w"] = growth_8w
    df[f"{prefix}_weekly_growth_accel"]  = growth_4w - growth_8w

    return df
