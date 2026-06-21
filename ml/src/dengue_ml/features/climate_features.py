import pandas as pd
from typing import Optional

from dengue_ml.config import CITY_COL


def add_climate_features(
    df: pd.DataFrame,
    group_col: str = CITY_COL,
    fit_stats: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add local climate lag/rolling/anomaly features.

    Parameters
    ----------
    df : DataFrame with columns tempmed, humidmed (weekly resolution)
    group_col : city column for within-city operations
    fit_stats : pre-computed city-level means for anomaly calculation.
                If None, compute from df (fit mode). Pass training-set stats
                at inference to avoid leakage.

    Returns
    -------
    (enriched_df, fit_stats)
    """
    df = df.copy()

    g_temp  = df.groupby(group_col, sort=False)["tempmed"]
    g_humid = df.groupby(group_col, sort=False)["humidmed"]

    # Lags (within city)
    df["temp_lag_13w"]  = g_temp.shift(13).values
    df["temp_lag_26w"]  = g_temp.shift(26).values
    df["humid_lag_13w"] = g_humid.shift(13).values
    df["humid_lag_26w"] = g_humid.shift(26).values

    # Rolling means (26-week, based on lag-1 to avoid leakage)
    df["temp_rolling_mean_26w"] = g_temp.shift(1).transform(
        lambda x: x.rolling(26, min_periods=1).mean()
    ).values
    df["humid_rolling_mean_26w"] = g_humid.shift(1).transform(
        lambda x: x.rolling(26, min_periods=1).mean()
    ).values

    # Anomaly = value − city long-run mean (fit on training data only)
    if fit_stats is None:
        fit_stats = (
            df.groupby(group_col, sort=False)[["tempmed", "humidmed"]]
            .mean()
            .rename(columns={"tempmed": "temp_mean", "humidmed": "humid_mean"})
        )

    df = df.merge(fit_stats, on=group_col, how="left")
    df["temp_anomaly"]  = df["tempmed"]  - df["temp_mean"]
    df["humid_anomaly"] = df["humidmed"] - df["humid_mean"]
    df = df.drop(columns=["temp_mean", "humid_mean"])

    return df, fit_stats
