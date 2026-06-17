import pandas as pd

from dengue_ml.config import EL_NINO_THRESHOLD, LA_NINA_THRESHOLD


def add_sst_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Niño 3.4 lag/rolling/categorical features.
    SST is a global signal: no groupby needed.
    All .shift() operations are on the already-quarterly merged nino34_anom column.
    """
    df = df.copy()
    s = df["nino34_anom"]

    df["nino34_lag_1q"] = s.shift(1)
    df["nino34_lag_2q"] = s.shift(2)
    df["nino34_lag_4q"] = s.shift(4)

    df["nino34_rolling_mean_2q"] = s.shift(1).rolling(2, min_periods=1).mean()
    df["nino34_rolling_mean_4q"] = s.shift(1).rolling(4, min_periods=1).mean()

    df["nino34_change_1q"] = s - s.shift(1)

    df["is_el_nino"] = (s > EL_NINO_THRESHOLD).astype(int)
    df["is_la_nina"] = (s < LA_NINA_THRESHOLD).astype(int)
    df["is_neutral"]  = (~(df["is_el_nino"].astype(bool) | df["is_la_nina"].astype(bool))).astype(int)

    return df
