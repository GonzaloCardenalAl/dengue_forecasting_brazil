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

    df["cases_lag_1q"] = log_s.shift(1).values
    df["cases_lag_2q"] = log_s.shift(2).values
    df["cases_lag_4q"] = log_s.shift(4).values

    df["cases_rolling_mean_2q"] = log_s.shift(1).transform(
        lambda x: x.rolling(2, min_periods=1).mean()
    ).values
    df["cases_rolling_mean_4q"] = log_s.shift(1).transform(
        lambda x: x.rolling(4, min_periods=1).mean()
    ).values
    df["cases_rolling_std_4q"] = log_s.shift(1).transform(
        lambda x: x.rolling(4, min_periods=2).std()
    ).values

    df["cases_growth_1q"] = df["cases_lag_1q"] - df["cases_lag_2q"]
    df["cases_growth_4q"] = df["cases_lag_1q"] - log_s.shift(5).values

    return df
