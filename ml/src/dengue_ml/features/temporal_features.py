import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ISO week-of-year cyclical encoding. The raw data's weeks are
    predominantly Sunday-start (InfoDengue's own epi-week convention), not
    ISO Monday-start, so `.dt.isocalendar()` is an accepted approximation of
    the "true" epi-week number, not an exact match -- same spirit as the
    week-53 approximation below.
    """
    df = df.copy()
    iso = df["week_start"].dt.isocalendar()
    df["year"] = iso["year"]   # ISO year, not .dt.year -- correctly reassigns
                                # late-Dec/early-Jan boundary weeks
    df["week"] = iso["week"]   # 1..52 or 1..53

    min_w = df["week_start"].min()
    df["time_index"] = (df["week_start"] - min_w).dt.days // 7  # 0-based global week index

    # Seasonal encoding. Week 53 deliberately maps past week 52's point on
    # the unit circle (53/52 of a full revolution) rather than wrapping to
    # week 1 -- week 53 falls at the very end of December, not the start of
    # January, so wrapping it to week 1's angle would be wrong.
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    return df
