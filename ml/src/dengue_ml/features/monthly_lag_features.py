from functools import lru_cache

import numpy as np
import pandas as pd

from dengue_ml.preprocessing import get_monthly_sst_table


@lru_cache(maxsize=None)
def _global_monthly_arrays(value_col: str) -> tuple[np.ndarray, np.ndarray]:
    monthly_df = get_monthly_sst_table().sort_values("date")
    return monthly_df["date"].to_numpy(), monthly_df[value_col].to_numpy(dtype=float)


@lru_cache(maxsize=None)
def _month_window_features(
    month_start: pd.Timestamp,
    value_col: str,
    prefix: str,
    n_months: int,
    log_transform: bool,
) -> dict:
    """
    SST/ENSO is one global series (same value for every city in a given month),
    so this only needs to be cached per month — not per (city, month) like
    the weekly cases/climate windows — making it even cheaper.
    """
    dates, vals_all = _global_monthly_arrays(value_col)
    qstart = np.datetime64(month_start)
    vals = vals_all[dates < qstart][-n_months:]
    if log_transform:
        vals = np.log1p(vals)

    window = np.full(n_months, np.nan)
    if len(vals) > 0:
        window[-len(vals):] = vals

    record = {f"{prefix}_month_t-{i}": window[-i] for i in range(1, n_months + 1)}

    last3, last6 = window[-3:], window[-6:]
    growth_3m = np.diff(last3).mean() if not np.isnan(last3).any() else np.nan
    growth_6m = np.diff(last6).mean() if not np.isnan(last6).any() else np.nan
    record[f"{prefix}_monthly_growth_avg_3m"] = growth_3m
    record[f"{prefix}_monthly_growth_avg_6m"] = growth_6m
    record[f"{prefix}_monthly_growth_accel"]  = growth_3m - growth_6m
    return record


def add_monthly_lag_features(
    monthly_df: pd.DataFrame,
    value_col: str,
    prefix: str,
    n_months: int,
    log_transform: bool = False,
) -> pd.DataFrame:
    """
    Last-N-months raw values before each row's own month (t-1 = most recent
    month), plus month-over-month growth/acceleration.

    Gives the model SST's actual recent trajectory at its native monthly
    resolution, rather than relying solely on the coarser 3m/6m/12m
    aggregate lags in sst_features.py.
    """
    df = monthly_df.reset_index(drop=True).copy()
    records = [
        _month_window_features(row["month_start"], value_col, prefix, n_months, log_transform)
        for _, row in df.iterrows()
    ]
    feat_df = pd.DataFrame(records, index=df.index)
    return df.join(feat_df)
