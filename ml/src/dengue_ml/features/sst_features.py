import pandas as pd

from dengue_ml.config import EL_NINO_THRESHOLD, LA_NINA_THRESHOLD


def add_sst_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Niño 3.4 lag/rolling/categorical features.
    SST is a global signal: no groupby needed.
    All .shift() operations are on the already-monthly merged nino34_anom column.
    """
    df = df.copy()
    s = df["nino34_anom"]

    df["nino34_lag_3m"]  = s.shift(3)
    df["nino34_lag_6m"]  = s.shift(6)
    df["nino34_lag_12m"] = s.shift(12)

    df["nino34_rolling_mean_6m"]  = s.shift(1).rolling(6, min_periods=1).mean()
    df["nino34_rolling_mean_12m"] = s.shift(1).rolling(12, min_periods=1).mean()

    df["nino34_change_3m"] = s - s.shift(3)

    df["is_el_nino"] = (s > EL_NINO_THRESHOLD).astype(int)
    df["is_la_nina"] = (s < LA_NINA_THRESHOLD).astype(int)
    df["is_neutral"]  = (~(df["is_el_nino"].astype(bool) | df["is_la_nina"].astype(bool))).astype(int)

    return df
