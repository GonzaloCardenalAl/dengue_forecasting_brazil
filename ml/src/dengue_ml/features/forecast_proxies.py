import numpy as np
import pandas as pd

from dengue_ml.config import CITY_COL


def last_val(df: pd.DataFrame, city: str | None, col: str):
    """Most recent non-null value of `col` in `df`, restricted to `city` if
    given (None for a global, non-city-keyed series e.g. nino34_anom)."""
    if col not in df.columns:
        return np.nan
    sub = df[df[CITY_COL] == city] if city is not None else df
    if sub.empty or col not in sub.columns:
        return np.nan
    val = sub.sort_values("week_start")[col].dropna()
    return val.iloc[-1] if not val.empty else np.nan


def climatological_val(df: pd.DataFrame, city: str, col: str, target_week: pd.Timestamp):
    """City + ISO-week-of-year historical mean of `col` -- the right stand-in
    for a seasonal exogenous variable whose true value at `target_week` isn't
    known yet (climate is seasonal, so the week-of-year mean beats flat
    carry-forward of the last observed value). Falls back to `last_val` for
    ISO week 53, which most years lack."""
    if col not in df.columns:
        return np.nan
    sub = df[df[CITY_COL] == city]
    if sub.empty:
        return np.nan
    target_woy = target_week.isocalendar().week
    woy = sub["week_start"].dt.isocalendar().week
    vals = sub.loc[woy == target_woy, col].dropna()
    if not vals.empty:
        return vals.mean()
    return last_val(df, city, col)
