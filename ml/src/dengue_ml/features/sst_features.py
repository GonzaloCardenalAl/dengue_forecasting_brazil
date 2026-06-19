import pandas as pd

from dengue_ml.config import EL_NINO_THRESHOLD, LA_NINA_THRESHOLD


def add_sst_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Niño 3.4 lag/rolling/categorical features.
    SST is a global signal: no groupby needed.
    All .shift() operations are on the already-merged nino34_anom column
    (bridged onto the weekly model table's own grain in preprocessing.py).

    Note: the lag/rolling/change columns below are not currently included in
    any feature_pipeline.py FEATURE_COLS list (confirmed dead/unused) --
    add_monthly_lag_features' nino34_month_t-i columns are used instead,
    since they correctly pull true calendar months back from the native
    monthly series rather than approximating via shifts on this weekly-grain
    (but month-duplicated) column.
    """
    df = df.copy()
    s = df["nino34_anom"]

    df["nino34_lag_13w"] = s.shift(13)
    df["nino34_lag_26w"] = s.shift(26)
    df["nino34_lag_52w"] = s.shift(52)

    df["nino34_rolling_mean_26w"] = s.shift(1).rolling(26, min_periods=1).mean()
    df["nino34_rolling_mean_52w"] = s.shift(1).rolling(52, min_periods=1).mean()

    df["nino34_change_13w"] = s - s.shift(13)

    df["is_el_nino"] = (s > EL_NINO_THRESHOLD).astype(int)
    df["is_la_nina"] = (s < LA_NINA_THRESHOLD).astype(int)
    df["is_neutral"]  = (~(df["is_el_nino"].astype(bool) | df["is_la_nina"].astype(bool))).astype(int)

    return df
