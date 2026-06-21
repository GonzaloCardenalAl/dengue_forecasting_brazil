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

    df["cases_lag_13w"] = log_s.shift(13).values
    df["cases_lag_26w"] = log_s.shift(26).values
    df["cases_lag_52w"] = log_s.shift(52).values

    df["cases_rolling_mean_26w"] = log_s.shift(1).transform(
        lambda x: x.rolling(26, min_periods=1).mean()
    ).values
    df["cases_rolling_mean_52w"] = log_s.shift(1).transform(
        lambda x: x.rolling(52, min_periods=1).mean()
    ).values
    df["cases_rolling_std_52w"] = log_s.shift(1).transform(
        lambda x: x.rolling(52, min_periods=2).std()
    ).values

    df["cases_growth_13w"] = df["cases_lag_13w"] - df["cases_lag_26w"]
    df["cases_growth_52w"] = df["cases_lag_13w"] - log_s.shift(65).values

    return df
