import numpy as np
import pandas as pd
from typing import Optional

from dengue_ml.config import TARGET, CITY_COL
from dengue_ml.features.temporal_features import add_temporal_features
from dengue_ml.features.target_lag_features import add_target_lag_features
from dengue_ml.features.climate_features import add_climate_features
from dengue_ml.features.sst_features import add_sst_features
from dengue_ml.features.weekly_lag_features import add_weekly_lag_features
from dengue_ml.features.monthly_lag_features import add_monthly_lag_features

# Lookback windows for week-level lag features (replaces quarter-level raw lags
# cases_lag_1q/2q/4q and temp/humid_lag_1q/2q with the actual weekly trajectory).
N_WEEKS_CASES   = 12  # ~1 quarter of weekly resolution
N_WEEKS_CLIMATE = 8

# SST is published monthly natively; quarter-bucketed lags (lag_1q=3mo, lag_2q=6mo)
# straddle past the EDA-identified peak ENSO->dengue correlation at lag 4 months.
N_MONTHS_SST = 6

# InfoDengue's own surveillance status fields (transmission/receptivity/alert
# level/Rt), at week-level resolution -- same rationale as the cases/climate
# windows above: the actual recent trajectory, not a single quarterly summary.
N_WEEKS_ALERT = 8  # matches N_WEEKS_CLIMATE
_ALERT_VALUE_COLS = [
    ("nivel_inc", "nivel_inc"), ("transmissao", "transmissao"),
    ("receptivo", "receptivo"), ("Rt", "rt"), ("p_rt1", "p_rt1"),
    ("sustained_rt", "sustained_rt"),
]

# Feature columns for each set (populated at end of module)
FEATURE_COLS: dict[str, list[str]] = {}

_TEMPORAL_COLS = [
    "year", "quarter", "time_index", "quarter_sin", "quarter_cos",
]
_TARGET_LAG_COLS = [
    "cases_rolling_mean_2q", "cases_rolling_mean_4q", "cases_rolling_std_4q",
    "cases_growth_1q", "cases_growth_4q",
]
_WEEKLY_CASES_COLS = (
    [f"cases_week_t-{i}" for i in range(1, N_WEEKS_CASES + 1)]
    + ["cases_weekly_growth_avg_4w", "cases_weekly_growth_avg_8w", "cases_weekly_growth_accel"]
)
_CLIMATE_COLS = [
    "tempmed", "humidmed",
    "temp_rolling_mean_2q", "humid_rolling_mean_2q",
    "temp_anomaly", "humid_anomaly",
]
_WEEKLY_CLIMATE_COLS = (
    [f"temp_week_t-{i}" for i in range(1, N_WEEKS_CLIMATE + 1)]
    + [f"humid_week_t-{i}" for i in range(1, N_WEEKS_CLIMATE + 1)]
    + ["temp_weekly_growth_avg_4w", "temp_weekly_growth_avg_8w", "temp_weekly_growth_accel"]
    + ["humid_weekly_growth_avg_4w", "humid_weekly_growth_avg_8w", "humid_weekly_growth_accel"]
)
_SST_COLS = (
    ["nino34_anom", "is_el_nino", "is_la_nina", "is_neutral"]
    + [f"nino34_month_t-{i}" for i in range(1, N_MONTHS_SST + 1)]
    + ["nino34_monthly_growth_avg_3m", "nino34_monthly_growth_avg_6m", "nino34_monthly_growth_accel"]
)
_WEEKLY_ALERT_COLS = [
    col
    for _, prefix in _ALERT_VALUE_COLS
    for col in (
        [f"{prefix}_week_t-{i}" for i in range(1, N_WEEKS_ALERT + 1)]
        + [f"{prefix}_weekly_growth_avg_4w", f"{prefix}_weekly_growth_avg_8w", f"{prefix}_weekly_growth_accel"]
    )
]

FEATURE_COLS["cases_only"] = (
    _TEMPORAL_COLS + _TARGET_LAG_COLS + _WEEKLY_CASES_COLS + _WEEKLY_ALERT_COLS
)
FEATURE_COLS["cases_climate"] = (
    FEATURE_COLS["cases_only"] + _CLIMATE_COLS + _WEEKLY_CLIMATE_COLS
)
FEATURE_COLS["cases_climate_sst"] = FEATURE_COLS["cases_climate"] + _SST_COLS


def build_features(
    df: pd.DataFrame,
    feature_set: str,
    climate_fit_stats: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Build model-ready feature matrix.

    Parameters
    ----------
    df : quarterly model table (output of prepare_model_table)
    feature_set : one of 'cases_only', 'cases_climate', 'cases_climate_sst'
    climate_fit_stats : pre-computed city means for climate anomaly.
                        None → compute from df (training mode).

    Returns
    -------
    X            : feature DataFrame
    y            : log1p(casos_est) Series
    meta         : city + quarter_start columns (for reporting)
    climate_stats: fit_stats to reuse at inference (None for cases_only)
    """
    if feature_set not in FEATURE_COLS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Choose from {list(FEATURE_COLS)}")

    df = add_temporal_features(df)
    df = add_target_lag_features(df)
    df = add_weekly_lag_features(
        df, value_col="casos_est", prefix="cases",
        n_weeks=N_WEEKS_CASES, log_transform=True,
    )
    for value_col, prefix in _ALERT_VALUE_COLS:
        df = add_weekly_lag_features(df, value_col=value_col, prefix=prefix, n_weeks=N_WEEKS_ALERT)

    returned_climate_stats = None
    if feature_set in ("cases_climate", "cases_climate_sst"):
        df, returned_climate_stats = add_climate_features(df, fit_stats=climate_fit_stats)
        df = add_weekly_lag_features(
            df, value_col="tempmed", prefix="temp", n_weeks=N_WEEKS_CLIMATE,
        )
        df = add_weekly_lag_features(
            df, value_col="umidmed", prefix="humid", n_weeks=N_WEEKS_CLIMATE,
        )

    if feature_set == "cases_climate_sst":
        df = add_sst_features(df)
        df = add_monthly_lag_features(
            df, value_col="nino34_anom", prefix="nino34", n_months=N_MONTHS_SST,
        )

    cols = FEATURE_COLS[feature_set]
    # Drop rows with NaNs in feature columns (first quarters lack lag history)
    df_clean = df.dropna(subset=cols).copy()

    X    = df_clean[cols]
    y    = np.log1p(df_clean[TARGET])
    meta = df_clean[[CITY_COL, "quarter_start"]].reset_index(drop=True)

    return X.reset_index(drop=True), y.reset_index(drop=True), meta, returned_climate_stats


def build_features_for_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Build (X_train, y_train, X_eval, y_eval, meta_eval, climate_stats) for a
    train/eval pair, where eval_df's lag features are computed using combined
    history (train_df + eval_df) so lags remain valid, then masked back down
    to only eval_df's quarters. climate_fit_stats are always fit on train_df
    only (no leakage from eval_df's climate distribution).
    """
    X_tr, y_tr, _, climate_stats = build_features(train_df, feature_set)

    combined = pd.concat([train_df, eval_df], ignore_index=True).sort_values(
        [CITY_COL, "quarter_start"]
    )
    X_all, y_all, meta_all, _ = build_features(
        combined, feature_set, climate_fit_stats=climate_stats
    )
    eval_quarters = set(eval_df["quarter_start"].unique())
    eval_mask = meta_all["quarter_start"].isin(eval_quarters)

    X_eval    = X_all[eval_mask].reset_index(drop=True)
    y_eval    = y_all[eval_mask].reset_index(drop=True)
    meta_eval = meta_all[eval_mask].reset_index(drop=True)

    return X_tr, y_tr, X_eval, y_eval, meta_eval, climate_stats
