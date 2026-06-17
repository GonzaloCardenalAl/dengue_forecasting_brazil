from functools import lru_cache

import numpy as np
import pandas as pd

from dengue_ml.config import CITY_COL, DATE_COL
from dengue_ml.preprocessing import get_weekly_table


@lru_cache(maxsize=None)
def _city_weekly_arrays(city: str, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Sorted (dates, values) for one city/column. Cached — weekly_df never changes."""
    weekly_df = get_weekly_table()
    city_weekly = weekly_df[weekly_df[CITY_COL] == city].sort_values(DATE_COL)
    return city_weekly[DATE_COL].to_numpy(), city_weekly[value_col].to_numpy(dtype=float)


@lru_cache(maxsize=None)
def _week_window_features(
    city: str,
    quarter_start: pd.Timestamp,
    value_col: str,
    prefix: str,
    n_weeks: int,
    log_transform: bool,
) -> dict:
    """
    The window for a given (city, quarter, column) is identical no matter which
    quarterly_df subset/fold calls into it, so memoizing here makes repeated
    build_features() calls during nested CV / hyperparameter search nearly free.
    """
    dates, vals_all = _city_weekly_arrays(city, value_col)
    qstart = np.datetime64(quarter_start)
    vals = vals_all[dates < qstart][-n_weeks:]
    if log_transform:
        vals = np.log1p(vals)

    window = np.full(n_weeks, np.nan)
    if len(vals) > 0:
        window[-len(vals):] = vals

    record = {f"{prefix}_week_t-{i}": window[-i] for i in range(1, n_weeks + 1)}

    last4, last8 = window[-4:], window[-8:]
    growth_4w = np.diff(last4).mean() if not np.isnan(last4).any() else np.nan
    growth_8w = np.diff(last8).mean() if not np.isnan(last8).any() else np.nan
    record[f"{prefix}_weekly_growth_avg_4w"] = growth_4w
    record[f"{prefix}_weekly_growth_avg_8w"] = growth_8w
    record[f"{prefix}_weekly_growth_accel"]  = growth_4w - growth_8w
    return record


def add_weekly_lag_features(
    quarterly_df: pd.DataFrame,
    value_col: str,
    prefix: str,
    n_weeks: int,
    log_transform: bool = False,
    group_col: str = CITY_COL,
) -> pd.DataFrame:
    """
    For each city-quarter row, pull the last `n_weeks` raw weekly values strictly
    before the quarter starts (t-1 = most recent week, t-n = oldest), plus
    week-over-week growth/acceleration. Gives the model the actual shape of the
    recent trajectory instead of a single quarterly aggregate.

    Multi-step-ahead forecasts (quarters beyond the next one) have no real future
    weekly data, so they reuse the same last-known weekly window — an inherent
    limitation of week-level features under a quarterly forecast horizon.
    """
    df = quarterly_df.reset_index(drop=True).copy()
    records = [
        _week_window_features(row[group_col], row["quarter_start"], value_col, prefix, n_weeks, log_transform)
        for _, row in df.iterrows()
    ]
    feat_df = pd.DataFrame(records, index=df.index)
    return df.join(feat_df)
