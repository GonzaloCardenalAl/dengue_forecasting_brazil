import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]  = df["month_start"].dt.year
    df["month"] = df["month_start"].dt.month

    # Integer time index (0-based, global across all cities and months)
    min_m = df["month_start"].min()
    df["time_index"] = (
        (df["month_start"].dt.year - min_m.year) * 12
        + (df["month_start"].dt.month - min_m.month)
    )

    # Seasonal encoding
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df
