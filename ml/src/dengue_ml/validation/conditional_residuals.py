import numpy as np
import pandas as pd

from dengue_ml.config import TARGET, CITY_COL, FORECAST_HORIZON
from dengue_ml.training_config import load_training_config

# growth_proxy = the trained epidemic classifier's predicted probability of
# "epidemic this week" (build_classification_features' nivel_inc==2 label),
# read for each (city, week, fold) row -- "how likely does our own model
# think we're already in confirmed epidemic territory, given only data
# available as of the most recently known week." Used to condition the
# residual distribution instead of a hard outbreak/non-outbreak label on the
# target month itself, which would require knowing the outcome we're trying
# to bound.
#
# This replaces an earlier version of this proxy that read InfoDengue's own
# nivel_inc_week_t-1 alert level directly (empirically ~92.5-93.1% LOFO
# coverage, vs. 90.6% for an earlier cases_growth_1q median-split, and 73.1%
# for the original independent-quantile-models approach). nivel_inc can't be
# computed for the forecast horizon (it's an InfoDengue-internal alert-
# classifier output with no documented formula), so it was replaced with our
# own classifier's prediction, which *can* run forward on forecasted weeks.
# See features/feature_pipeline.py's nivel_inc_week_t-1 side-channel and
# nested_cv_classifier.py's nivel_inc_rule for the retained benchmark
# comparison against this old rule.
PROXY_SOURCE_FEATURE = "classifier_predicted_proba"
# Standard probability decision threshold (growth_proxy is now a predicted
# probability in [0, 1], not nivel_inc's ordinal 0/1/2 scale).
REGIME_THRESHOLD = 0.5
# Label this proxy is stored under in fold_predictions/preds_df.
PROXY_COL = "growth_proxy"

# Known data-collection outages to exclude from residual-quantile
# CALIBRATION only -- the rows stay in fold_predictions/plots as real
# history, they just aren't valid evidence of "how wrong the model normally
# is," since the gap is a reporting failure, not a forecasting error.
#
# Vitória's InfoDengue feed reported a flat 0 (casos_est == casos_est_min ==
# casos_est_max == raw casos, every single week) for 49 straight weeks --
# confirmed against ml/data/raw/infodengue_capitals_subsetBR.csv. No real
# city goes a full year with zero dengue cases across every column while the
# model (trained on climate + lag features) predicted 50-115/week throughout;
# this is a surveillance outage (plausibly COVID-era), not a quiet season.
# Left uncalibrated-against, this single anomaly was setting the low-regime,
# low-magnitude quantile tail for *every* city, not just Vitória.
KNOWN_DATA_GAPS = [
    {CITY_COL: "Vitória", "start": "2021-01-03", "end": "2021-12-05"},
]


def is_known_data_gap(df: pd.DataFrame) -> pd.Series:
    """Boolean mask, True for rows inside a KNOWN_DATA_GAPS window. Auto-
    detects whichever per-row date column the caller's frame carries
    (weekly fold_predictions has week_start; aggregate_oof_to_monthly's
    output has month_start); frames with neither (e.g. synthetic test
    fixtures) have nothing to mask, so just return all-False."""
    date_col = next((c for c in ("week_start", "month_start") if c in df.columns), None)
    if date_col is None:
        return pd.Series(False, index=df.index)
    dates = pd.to_datetime(df[date_col])
    mask = pd.Series(False, index=df.index)
    for gap in KNOWN_DATA_GAPS:
        mask |= (
            (df[CITY_COL] == gap[CITY_COL])
            & (dates >= pd.Timestamp(gap["start"]))
            & (dates <= pd.Timestamp(gap["end"]))
        )
    return mask


def quantile_bounds() -> tuple[float, float]:
    q = load_training_config()["xgboost"]["quantiles"]
    return q["lower"], q["upper"]


def assign_loFo_conditional_ci(
    fold_predictions: pd.DataFrame,
    model_name: str,
    high_regime_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Leave-one-fold-out conditional residual quantile intervals for OOF reporting.

    Splits rows at the fixed REGIME_THRESHOLD into "confirmed epidemic" /
    "not yet" regimes, takes the empirical residual quantiles within each
    regime from all *other* folds, and applies them as an offset to that
    fold's own point predictions. A fold never uses its own residuals to
    calibrate its own band.

    high_regime_mask : optional precomputed boolean Series, indexed like
        `sub` (the model's rows), to use instead of `PROXY_COL >=
        REGIME_THRESHOLD` -- lets the exact same LOFO math be reused to
        evaluate alternate regime proxies (e.g. sustained_rt, or a trained
        classifier's thresholded probability) without touching the default
        path. None (default) reproduces today's production behavior exactly.

    Rows inside a KNOWN_DATA_GAPS window are dropped from every fold's
    calibration pool (they're known-bad signal, not model error) but still
    receive their own lower_95/upper_95 like any other held-out row, so
    plots built on the result don't show a hole.
    """
    lower_q, upper_q = quantile_bounds()
    sub = fold_predictions[fold_predictions["model"] == model_name].copy()
    log_resid = np.log1p(sub[TARGET]) - np.log1p(sub["predicted"])
    sub["_log_resid"] = log_resid

    if high_regime_mask is None:
        sub["_is_high"] = sub[PROXY_COL] >= REGIME_THRESHOLD
    else:
        sub["_is_high"] = high_regime_mask.reindex(sub.index)

    q_lower = pd.Series(np.nan, index=sub.index)
    q_upper = pd.Series(np.nan, index=sub.index)

    is_gap = is_known_data_gap(sub)

    for fold in sub["fold"].unique():
        calib = sub[(sub["fold"] != fold) & ~is_gap]

        for is_high in (True, False):
            calib_mask = calib["_is_high"] == is_high
            bin_resid = calib.loc[calib_mask, "_log_resid"]
            held_out_mask = (sub["fold"] == fold) & (sub["_is_high"] == is_high)
            q_lower.loc[held_out_mask] = bin_resid.quantile(lower_q)
            q_upper.loc[held_out_mask] = bin_resid.quantile(upper_q)

    log_pred = np.log1p(sub["predicted"])
    lower = np.expm1(log_pred + q_lower)
    upper = np.expm1(log_pred + q_upper)
    # Band is built as an offset around the point estimate, so it always
    # brackets it by construction — the min/max are just a safety net for
    # floating-point edge cases at q_lower/q_upper == 0.
    lower = np.minimum(lower, sub["predicted"])
    upper = np.maximum(upper, sub["predicted"])

    if high_regime_mask is not None:
        return sub.assign(lower_95=lower, upper_95=upper)

    fold_predictions = fold_predictions.copy()
    fold_predictions.loc[sub.index, "lower_95"] = lower
    fold_predictions.loc[sub.index, "upper_95"] = upper
    return fold_predictions


def aggregate_oof_to_monthly(
    fold_predictions: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Monthly-aggregated counterpart of one model's weekly OOF fold_predictions,
    for a validation plot that doesn't inherit the weekly band's heavy
    near-zero-count noise (see compute_quarterly_residual_quantile_table's
    docstring for the same rationale at quarterly grain).

    Sums TARGET/predicted to (city, month, fold) and keeps the first week's
    growth_proxy ("value available as of the start of the period"), then
    re-tags the result as a single-model frame shaped exactly like
    fold_predictions so assign_loFo_conditional_ci can be reused unmodified to
    compute monthly-grain LOFO conditional CI bands on it.
    """
    sub = fold_predictions[fold_predictions["model"] == model_name].copy()
    sub["week_start"] = pd.to_datetime(sub["week_start"])
    sub["_month"] = sub["week_start"].values.astype("datetime64[M]")

    grouped = sub.groupby([CITY_COL, "_month", "fold"]).agg(
        **{TARGET: (TARGET, "sum")},
        predicted=("predicted", "sum"),
        **{PROXY_COL: (PROXY_COL, "first")},
    ).reset_index()
    grouped = grouped.rename(columns={"_month": "month_start"})
    grouped["model"] = model_name
    return grouped


def aggregate_oof_to_quarterly(
    fold_predictions: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Quarterly twin of `aggregate_oof_to_monthly` -- same shape (keeps `fold`
    so `assign_loFo_conditional_ci` can be reused unmodified), just grouped
    by calendar quarter instead of month. Used to build the "{prev_year}
    Estimated Cases" line's own LOFO conditional CI for the redesigned
    year-over-year plot.
    """
    sub = fold_predictions[fold_predictions["model"] == model_name].copy()
    sub["week_start"] = pd.to_datetime(sub["week_start"])
    sub["_quarter"] = pd.PeriodIndex(sub["week_start"], freq="Q").to_timestamp()

    grouped = sub.groupby([CITY_COL, "_quarter", "fold"]).agg(
        **{TARGET: (TARGET, "sum")},
        predicted=("predicted", "sum"),
        **{PROXY_COL: (PROXY_COL, "first")},
    ).reset_index()
    grouped = grouped.rename(columns={"_quarter": "quarter_start"})
    grouped["model"] = model_name
    return grouped


def compute_regime_coverage(
    fold_predictions: pd.DataFrame,
    model_name: str,
    high_regime_mask: pd.Series | None = None,
) -> dict:
    """
    LOFO coverage + regime bin sizes for an arbitrary regime-split candidate
    (the existing nivel_inc rule, the sustained_rt rule, or a trained
    classifier's thresholded probability), reusing assign_loFo_conditional_ci's
    exact math so results are directly comparable to the production proxy's
    ~92.5-93% baseline. Does not mutate fold_predictions or PROXY_SOURCE_FEATURE.
    high_regime_mask=None reproduces the production nivel_inc proxy.
    """
    result = assign_loFo_conditional_ci(fold_predictions, model_name, high_regime_mask=high_regime_mask)
    sub = result[result["model"] == model_name]
    is_high = sub[PROXY_COL] >= REGIME_THRESHOLD if high_regime_mask is None else sub["_is_high"]
    in_band = (sub[TARGET] >= sub["lower_95"]) & (sub[TARGET] <= sub["upper_95"])
    return {
        "coverage": float(in_band.mean()),
        "n_high": int(is_high.sum()),
        "n_low": int((~is_high).sum()),
    }


def compute_residual_quantile_table(
    fold_predictions: pd.DataFrame,
    model_name: str,
) -> dict:
    """
    Calibration table for the production forecast: the fixed REGIME_THRESHOLD
    plus one pair of residual quantiles per side, computed from all OOF
    residuals for this model across the full nested CV run (none of which the
    model trained on its own label for). Rows inside a KNOWN_DATA_GAPS window
    are dropped first -- see that constant's docstring.
    """
    lower_q, upper_q = quantile_bounds()
    sub = fold_predictions[fold_predictions["model"] == model_name]
    sub = sub[~is_known_data_gap(sub)]
    log_resid = np.log1p(sub[TARGET]) - np.log1p(sub["predicted"])

    low_resid  = log_resid[sub[PROXY_COL] <  REGIME_THRESHOLD]
    high_resid = log_resid[sub[PROXY_COL] >= REGIME_THRESHOLD]

    return {
        "proxy_col": PROXY_SOURCE_FEATURE,
        "threshold": REGIME_THRESHOLD,
        "low":  {"q_lower": float(low_resid.quantile(lower_q)),  "q_upper": float(low_resid.quantile(upper_q))},
        "high": {"q_lower": float(high_resid.quantile(lower_q)), "q_upper": float(high_resid.quantile(upper_q))},
    }


def compute_quarterly_residual_quantile_table(
    fold_predictions: pd.DataFrame,
    model_name: str,
) -> dict:
    """
    Quarterly counterpart of `compute_residual_quantile_table`, for the final
    forecast deliverable (weekly model, predictions summed to quarters).

    Naively summing weekly lower_95/upper_95 bounds to get a quarterly band
    is not statistically valid (overstates width if treated as fully
    correlated, wrong if treated as independent without proper convolution).
    Instead: aggregate this model's weekly OOF actual/predicted values to
    (city, quarter, fold) sums, compute the quarterly residual
    log1p(actual_sum) - log1p(predicted_sum), and run the exact same
    regime-conditional quantile calibration as the weekly table -- keyed off
    the same growth_proxy, read from the first week of each quarter (the
    proxy is "classifier's predicted epidemic probability as of the most
    recently known week before the period starts", so the first week's value
    is the one that would actually be available at quarterly-forecast time).

    Rows inside a KNOWN_DATA_GAPS window are dropped before aggregating to
    quarters -- see that constant's docstring.
    """
    lower_q, upper_q = quantile_bounds()
    sub = fold_predictions[fold_predictions["model"] == model_name].copy()
    sub = sub[~is_known_data_gap(sub)]
    sub = sub.sort_values("week_start")
    sub["_quarter"] = pd.PeriodIndex(sub["week_start"], freq="Q").to_timestamp()

    grouped = sub.groupby([CITY_COL, "_quarter", "fold"]).agg(
        actual_sum=(TARGET, "sum"),
        predicted_sum=("predicted", "sum"),
        **{PROXY_COL: (PROXY_COL, "first")},
    ).reset_index()

    log_resid = np.log1p(grouped["actual_sum"]) - np.log1p(grouped["predicted_sum"])

    low_resid  = log_resid[grouped[PROXY_COL] <  REGIME_THRESHOLD]
    high_resid = log_resid[grouped[PROXY_COL] >= REGIME_THRESHOLD]

    return {
        "proxy_col": PROXY_SOURCE_FEATURE,
        "threshold": REGIME_THRESHOLD,
        "low":  {"q_lower": float(low_resid.quantile(lower_q)),  "q_upper": float(low_resid.quantile(upper_q))},
        "high": {"q_lower": float(high_resid.quantile(lower_q)), "q_upper": float(high_resid.quantile(upper_q))},
    }


def compute_horizon_bucketed_quarterly_residual_quantile_table(
    fold_predictions_ar: pd.DataFrame,
    model_name: str,
) -> dict:
    """
    Horizon-aware counterpart of `compute_quarterly_residual_quantile_table`,
    built from `validation/autoregressive_cv.run_autoregressive_cv`'s OOF
    residuals (a real autoregressive multi-step rollout within each outer CV
    fold's test window) instead of 1-step-ahead residuals built from real
    historical lags. Bucketed by `quarter_position` (1st/2nd/3rd/4th quarter
    of the forecast horizon) rather than calendar quarter, so later
    quarters' wider compounding error shows up as a wider calibrated band --
    the flat 1-step table can't capture this since every quarter's residuals
    there come from predictions made with perfect historical lags.

    Returns {1: {...}, 2: {...}, 3: {...}, 4: {...}}, each shaped like
    `compute_quarterly_residual_quantile_table`'s single-table return.

    Rows inside a KNOWN_DATA_GAPS window are dropped first -- see that
    constant's docstring.
    """
    lower_q, upper_q = quantile_bounds()
    sub = fold_predictions_ar[fold_predictions_ar["model"] == model_name].copy()
    sub = sub[~is_known_data_gap(sub)]

    grouped = sub.groupby([CITY_COL, "fold", "quarter_position"]).agg(
        actual_sum=(TARGET, "sum"),
        predicted_sum=("predicted", "sum"),
        **{PROXY_COL: (PROXY_COL, "first")},
    ).reset_index()

    log_resid = np.log1p(grouped["actual_sum"]) - np.log1p(grouped["predicted_sum"])

    table = {}
    for qpos in (1, 2, 3, 4):
        bucket_mask  = grouped["quarter_position"] == qpos
        bucket_resid = log_resid[bucket_mask]
        bucket_proxy = grouped.loc[bucket_mask, PROXY_COL]

        low_resid  = bucket_resid[bucket_proxy <  REGIME_THRESHOLD]
        high_resid = bucket_resid[bucket_proxy >= REGIME_THRESHOLD]

        table[qpos] = {
            "proxy_col": PROXY_SOURCE_FEATURE,
            "threshold": REGIME_THRESHOLD,
            "low":  {"q_lower": float(low_resid.quantile(lower_q)),  "q_upper": float(low_resid.quantile(upper_q))},
            "high": {"q_lower": float(high_resid.quantile(lower_q)), "q_upper": float(high_resid.quantile(upper_q))},
        }
    return table


# Used only by compute_horizon_bucketed_monthly_residual_quantile_table.
# Monthly buckets only have ~40 (city, fold) samples each (one summed value
# per outer-CV fold-year per city, regardless of bucket width); splitting
# that further into low/high growth-proxy regime starves late-season "high"
# buckets to n=3-6, where the global quantile_bounds() (1.25%/98.75%) tail is
# just the single most extreme historical residual rather than a real
# percentile estimate. Pooling both regimes and using a less extreme width
# keeps the calibration from being dominated by one outlier fold-year.
_MONTHLY_AR_QUANTILES = (0.05, 0.95)


def compute_horizon_bucketed_monthly_residual_quantile_table(
    fold_predictions_ar: pd.DataFrame,
    model_name: str,
) -> dict:
    """
    Monthly twin of `compute_horizon_bucketed_quarterly_residual_quantile_table`
    -- same autoregressive-rollout residuals, bucketed by `month_position`
    (1st..12th month of the forecast horizon) instead of `quarter_position`.
    Unlike the quarterly/weekly tables, this one does NOT split by growth-proxy
    regime (see _MONTHLY_AR_QUANTILES) -- "low"/"high" are identical so
    `apply_residual_quantile_table`'s regime dispatch is a no-op here.
    Returns {1: {...}, ..., 12: {...}}.
    """
    lower_q, upper_q = _MONTHLY_AR_QUANTILES
    sub = fold_predictions_ar[fold_predictions_ar["model"] == model_name].copy()
    sub = sub[~is_known_data_gap(sub)]

    grouped = sub.groupby([CITY_COL, "fold", "month_position"]).agg(
        actual_sum=(TARGET, "sum"),
        predicted_sum=("predicted", "sum"),
    ).reset_index()

    log_resid = np.log1p(grouped["actual_sum"]) - np.log1p(grouped["predicted_sum"])

    table = {}
    for mpos in range(1, 13):
        bucket_resid = log_resid[grouped["month_position"] == mpos]
        band = {"q_lower": float(bucket_resid.quantile(lower_q)), "q_upper": float(bucket_resid.quantile(upper_q))}
        table[mpos] = {
            "proxy_col": PROXY_SOURCE_FEATURE,
            "threshold": REGIME_THRESHOLD,
            "low": band,
            "high": band,
        }
    return table


def compute_horizon_bucketed_weekly_residual_quantile_table(
    fold_predictions_ar: pd.DataFrame,
    model_name: str,
) -> dict:
    """
    Weekly twin of `compute_horizon_bucketed_quarterly_residual_quantile_table`
    -- bucketed by `week_position` (1st..FORECAST_HORIZON-th week of the
    rollout). Grain already equals the bucket here (one row per week), so
    there's no quarterly/monthly-style summing step before taking residuals,
    but the `actual_sum`/`predicted_sum` naming is kept for symmetry with the
    other two horizon-bucketed tables and so `apply_horizon_bucketed_quantile_table`
    doesn't need a special case.
    Returns {1: {...}, ..., FORECAST_HORIZON: {...}}.
    """
    lower_q, upper_q = quantile_bounds()
    sub = fold_predictions_ar[fold_predictions_ar["model"] == model_name].copy()
    sub = sub[~is_known_data_gap(sub)]

    grouped = sub.groupby([CITY_COL, "fold", "week_position"]).agg(
        actual_sum=(TARGET, "sum"),
        predicted_sum=("predicted", "sum"),
        **{PROXY_COL: (PROXY_COL, "first")},
    ).reset_index()

    log_resid = np.log1p(grouped["actual_sum"]) - np.log1p(grouped["predicted_sum"])

    table = {}
    for wpos in range(1, FORECAST_HORIZON + 1):
        bucket_mask  = grouped["week_position"] == wpos
        bucket_resid = log_resid[bucket_mask]
        bucket_proxy = grouped.loc[bucket_mask, PROXY_COL]

        low_resid  = bucket_resid[bucket_proxy <  REGIME_THRESHOLD]
        high_resid = bucket_resid[bucket_proxy >= REGIME_THRESHOLD]

        table[wpos] = {
            "proxy_col": PROXY_SOURCE_FEATURE,
            "threshold": REGIME_THRESHOLD,
            "low":  {"q_lower": float(low_resid.quantile(lower_q)),  "q_upper": float(low_resid.quantile(upper_q))},
            "high": {"q_lower": float(high_resid.quantile(lower_q)), "q_upper": float(high_resid.quantile(upper_q))},
        }
    return table


def apply_horizon_bucketed_quantile_table(
    predicted: np.ndarray,
    proxy: np.ndarray,
    position: np.ndarray,
    bucketed_table: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch each row to `apply_residual_quantile_table` for its bucket
    position (week/month/quarter -- whichever grain `bucketed_table` was
    built at, by compute_horizon_bucketed_{weekly,monthly,quarterly}_
    residual_quantile_table)."""
    predicted = np.asarray(predicted, dtype=float)
    proxy = np.asarray(proxy, dtype=float)
    position = np.asarray(position)

    lower = np.full(len(predicted), np.nan)
    upper = np.full(len(predicted), np.nan)
    for pos, table in bucketed_table.items():
        mask = position == pos
        if not mask.any():
            continue
        lower[mask], upper[mask] = apply_residual_quantile_table(predicted[mask], proxy[mask], table)
    return lower, upper


def attach_classifier_proxy(
    fold_predictions_reg: pd.DataFrame,
    fold_predictions_clf: pd.DataFrame,
    classifier_model_name: str,
) -> pd.DataFrame:
    """
    Join the epidemic classifier's OOF predicted_proba onto the regression
    pipeline's fold_predictions as growth_proxy, then run the regime-
    conditional CI assignment now that growth_proxy actually exists.

    Both nested CV runs (regression in nested_cv.py, classifier in
    nested_cv_classifier.py) use the identical make_outer_splits/
    make_inner_splits protocol on the same df, so every (city_name,
    week_start) pair lands in the same outer fold in both -- joining on
    (city_name, week_start, fold) is therefore exact, not approximate.
    """
    clf = fold_predictions_clf[fold_predictions_clf["model"] == classifier_model_name]
    proxy_map = clf.set_index([CITY_COL, "week_start", "fold"])["predicted_proba"]

    fold_predictions_reg = fold_predictions_reg.copy()
    keys = pd.MultiIndex.from_arrays(
        [fold_predictions_reg[CITY_COL], fold_predictions_reg["week_start"], fold_predictions_reg["fold"]]
    )
    fold_predictions_reg[PROXY_COL] = proxy_map.reindex(keys).values

    for model_name in fold_predictions_reg.loc[fold_predictions_reg[PROXY_COL].notna(), "model"].unique():
        fold_predictions_reg = assign_loFo_conditional_ci(fold_predictions_reg, model_name)

    return fold_predictions_reg


def apply_residual_quantile_table(
    predicted: np.ndarray,
    proxy: np.ndarray,
    table: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a calibration table (from compute_residual_quantile_table) to new point predictions."""
    predicted = np.asarray(predicted, dtype=float)
    proxy = np.asarray(proxy, dtype=float)
    is_high = proxy >= table["threshold"]

    q_lower = np.where(is_high, table["high"]["q_lower"], table["low"]["q_lower"])
    q_upper = np.where(is_high, table["high"]["q_upper"], table["low"]["q_upper"])

    log_pred = np.log1p(predicted)
    lower = np.minimum(np.expm1(log_pred + q_lower), predicted)
    upper = np.maximum(np.expm1(log_pred + q_upper), predicted)
    return lower, upper
