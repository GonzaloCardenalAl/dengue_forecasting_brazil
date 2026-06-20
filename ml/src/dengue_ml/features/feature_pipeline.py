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

# Lookback windows for week-level lag features (supplements the coarser raw
# aggregate lags cases_lag_13w/26w/52w and temp/humid_lag_13w/26w in
# target_lag_features.py/climate_features.py with the actual weekly trajectory).
N_WEEKS_CASES   = 12  # ~3 months of weekly resolution
N_WEEKS_CLIMATE = 8

# SST is published monthly natively; this gives the model the raw recent
# trajectory rather than relying solely on the coarser 3m/6m/12m aggregate
# lags in sst_features.py.
N_MONTHS_SST = 6

# InfoDengue's own Rt-derived surveillance fields, at week-level resolution
# -- same rationale as the cases/climate windows above: the actual recent
# trajectory, not a single monthly summary.
#
# transmissao/receptivo/nivel_inc are deliberately NOT here, even though
# they're in the raw InfoDengue feed: they're outputs of InfoDengue's own
# internal alert classifier with no documented formula, so there is no way
# to compute them for the forecast horizon (unlike Rt/p_rt1, which we can
# estimate forward via features/rt_estimation.py). Carrying them forward flat
# for a year would be indefensible, so they're excluded from feature
# engineering entirely. nivel_inc is still kept (just not as a lag feature
# here) as the epidemic classifier's label -- see build_classification_features.
N_WEEKS_ALERT = 8  # matches N_WEEKS_CLIMATE
_ALERT_VALUE_COLS = [
    ("Rt", "rt"), ("p_rt1", "p_rt1"), ("sustained_rt", "sustained_rt"),
]

# Feature columns for each set (populated at end of module)
FEATURE_COLS: dict[str, list[str]] = {}

_TEMPORAL_COLS = [
    "year", "week", "time_index", "week_sin", "week_cos",
]
_TARGET_LAG_COLS = [
    "cases_rolling_mean_26w", "cases_rolling_mean_52w", "cases_rolling_std_52w",
    "cases_growth_13w", "cases_growth_52w",
]
_WEEKLY_CASES_COLS = (
    [f"cases_week_t-{i}" for i in range(1, N_WEEKS_CASES + 1)]
    + ["cases_weekly_growth_avg_4w", "cases_weekly_growth_avg_8w", "cases_weekly_growth_accel"]
)
_CLIMATE_COLS = [
    "tempmed", "humidmed",
    "temp_rolling_mean_26w", "humid_rolling_mean_26w",
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


def _build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str,
    climate_fit_stats: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Shared feature-construction core for both the regression (`build_features`)
    and classification (`build_classification_features`) builders -- identical
    lag/climate/SST columns either way, only the attached `y` differs.

    Returns
    -------
    df_clean      : feature-engineered rows with NaN-lag rows dropped (still has
                     all original columns, e.g. `nivel_inc`, for the caller to
                     derive its own `y` from)
    X             : feature DataFrame (cols restricted to `feature_set`)
    meta          : city + week_start columns (for reporting)
    climate_stats : fit_stats to reuse at inference (None for cases_only)
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

    # nivel_inc_week_t-1 is kept as a side channel (NOT added to FEATURE_COLS,
    # so it never reaches the cases-forecasting or classifier models as an
    # input) purely for the nivel_inc_rule benchmark comparison in
    # nested_cv_classifier.py / proxy_comparison_table -- both only need the
    # last *observed* value, which this lag already is.
    if "nivel_inc" in df.columns:
        df["nivel_inc_week_t-1"] = df.groupby(CITY_COL, sort=False)["nivel_inc"].shift(1)

    returned_climate_stats = None
    if feature_set in ("cases_climate", "cases_climate_sst"):
        df, returned_climate_stats = add_climate_features(df, fit_stats=climate_fit_stats)
        df = add_weekly_lag_features(
            df, value_col="tempmed", prefix="temp", n_weeks=N_WEEKS_CLIMATE,
        )
        df = add_weekly_lag_features(
            df, value_col="humidmed", prefix="humid", n_weeks=N_WEEKS_CLIMATE,
        )

    if feature_set == "cases_climate_sst":
        df = add_sst_features(df)
        df = add_monthly_lag_features(
            df, value_col="nino34_anom", prefix="nino34", n_months=N_MONTHS_SST,
        )

    cols = FEATURE_COLS[feature_set]
    # Drop rows with NaNs in feature columns (first months lack lag history)
    df_clean = df.dropna(subset=cols).copy()

    X    = df_clean[cols]
    meta_cols = [CITY_COL, "week_start"]
    if "nivel_inc_week_t-1" in df_clean.columns:
        meta_cols.append("nivel_inc_week_t-1")
    meta = df_clean[meta_cols].reset_index(drop=True)

    return df_clean.reset_index(drop=True), X.reset_index(drop=True), meta, returned_climate_stats


def build_features(
    df: pd.DataFrame,
    feature_set: str,
    climate_fit_stats: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Build model-ready feature matrix.

    Parameters
    ----------
    df : weekly model table (output of prepare_model_table)
    feature_set : one of 'cases_only', 'cases_climate', 'cases_climate_sst'
    climate_fit_stats : pre-computed city means for climate anomaly.
                        None → compute from df (training mode).

    Returns
    -------
    X            : feature DataFrame
    y            : log1p(casos_est) Series
    meta         : city + week_start columns (for reporting)
    climate_stats: fit_stats to reuse at inference (None for cases_only)
    """
    df_clean, X, meta, returned_climate_stats = _build_feature_matrix(df, feature_set, climate_fit_stats)
    y = np.log1p(df_clean[TARGET])
    return X, y.reset_index(drop=True), meta, returned_climate_stats


def build_classification_features(
    df: pd.DataFrame,
    feature_set: str,
    climate_fit_stats: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Same leak-free feature columns as `build_features`, but with the binary
    "epidemic this week" label (`nivel_inc == 2`, InfoDengue's own
    week-level alert level -- valid as a label, never used as a feature)
    instead of
    log1p(casos_est).

    Returns
    -------
    X            : feature DataFrame (identical columns to build_features)
    y            : binary is_epidemic Series (1 = nivel_inc==2 this week)
    meta         : city + week_start columns (for reporting)
    climate_stats: fit_stats to reuse at inference (None for cases_only)
    """
    df_clean, X, meta, returned_climate_stats = _build_feature_matrix(df, feature_set, climate_fit_stats)
    y = (df_clean["nivel_inc"] == 2).astype(int)
    return X, y.reset_index(drop=True), meta, returned_climate_stats


def build_features_for_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Build (X_train, y_train, X_eval, y_eval, meta_eval, climate_stats) for a
    train/eval pair, where eval_df's lag features are computed using combined
    history (train_df + eval_df) so lags remain valid, then masked back down
    to only eval_df's weeks. climate_fit_stats are always fit on train_df
    only (no leakage from eval_df's climate distribution).
    """
    X_tr, y_tr, _, climate_stats = build_features(train_df, feature_set)

    combined = pd.concat([train_df, eval_df], ignore_index=True).sort_values(
        [CITY_COL, "week_start"]
    )
    X_all, y_all, meta_all, _ = build_features(
        combined, feature_set, climate_fit_stats=climate_stats
    )
    eval_weeks = set(eval_df["week_start"].unique())
    eval_mask = meta_all["week_start"].isin(eval_weeks)

    X_eval    = X_all[eval_mask].reset_index(drop=True)
    y_eval    = y_all[eval_mask].reset_index(drop=True)
    meta_eval = meta_all[eval_mask].reset_index(drop=True)

    return X_tr, y_tr, X_eval, y_eval, meta_eval, climate_stats


def build_classification_features_for_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Classification counterpart of `build_features_for_split` -- identical
    train/eval feature construction (combined-history lags, train-only climate
    stats), but `y` is the binary is_epidemic label from
    `build_classification_features` instead of log1p(casos_est).
    """
    X_tr, y_tr, _, climate_stats = build_classification_features(train_df, feature_set)

    combined = pd.concat([train_df, eval_df], ignore_index=True).sort_values(
        [CITY_COL, "week_start"]
    )
    X_all, y_all, meta_all, _ = build_classification_features(
        combined, feature_set, climate_fit_stats=climate_stats
    )
    eval_weeks = set(eval_df["week_start"].unique())
    eval_mask = meta_all["week_start"].isin(eval_weeks)

    X_eval    = X_all[eval_mask].reset_index(drop=True)
    y_eval    = y_all[eval_mask].reset_index(drop=True)
    meta_eval = meta_all[eval_mask].reset_index(drop=True)

    return X_tr, y_tr, X_eval, y_eval, meta_eval, climate_stats
