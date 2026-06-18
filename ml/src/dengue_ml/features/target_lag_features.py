import numpy as np
import pandas as pd

from dengue_ml.config import TARGET, CITY_COL


def add_target_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET,
    group_col: str = CITY_COL,
) -> pd.DataFrame:
    """Compute lag/rolling features on log1p(target), grouped by city."""
    df = df.copy()
    log_target = np.log1p(df[target_col])

    g = df.groupby(group_col, sort=False)

    def _within_city(series: pd.Series) -> pd.Series:
        return series

    log_s = df.assign(_log=log_target).groupby(group_col, sort=False)["_log"]

    df["cases_lag_3m"]  = log_s.shift(3).values
    df["cases_lag_6m"]  = log_s.shift(6).values
    df["cases_lag_12m"] = log_s.shift(12).values

    df["cases_rolling_mean_6m"] = log_s.shift(1).transform(
        lambda x: x.rolling(6, min_periods=1).mean()
    ).values
    df["cases_rolling_mean_12m"] = log_s.shift(1).transform(
        lambda x: x.rolling(12, min_periods=1).mean()
    ).values
    df["cases_rolling_std_12m"] = log_s.shift(1).transform(
        lambda x: x.rolling(12, min_periods=2).std()
    ).values

    df["cases_growth_3m"]  = df["cases_lag_3m"] - df["cases_lag_6m"]
    df["cases_growth_12m"] = df["cases_lag_3m"] - log_s.shift(15).values

    return df
