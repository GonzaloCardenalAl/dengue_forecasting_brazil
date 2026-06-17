import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"]    = df["quarter_start"].dt.year
    df["quarter"] = df["quarter_start"].dt.quarter

    # Integer time index (0-based, global across all cities and quarters)
    min_q = df["quarter_start"].min()
    df["time_index"] = (
        (df["quarter_start"].dt.year - min_q.year) * 4
        + (df["quarter_start"].dt.quarter - min_q.quarter)
    )

    # Seasonal encoding
    df["quarter_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)
    return df
