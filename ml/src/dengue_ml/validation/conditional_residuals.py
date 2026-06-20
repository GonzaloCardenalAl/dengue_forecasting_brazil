import numpy as np
import pandas as pd

from dengue_ml.config import TARGET, CITY_COL
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


def _quantile_bounds() -> tuple[float, float]:
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
    """
    lower_q, upper_q = _quantile_bounds()
    sub = fold_predictions[fold_predictions["model"] == model_name].copy()
    log_resid = np.log1p(sub[TARGET]) - np.log1p(sub["predicted"])
    sub["_log_resid"] = log_resid

    if high_regime_mask is None:
        sub["_is_high"] = sub[PROXY_COL] >= REGIME_THRESHOLD
    else:
        sub["_is_high"] = high_regime_mask.reindex(sub.index)

    q_lower = pd.Series(np.nan, index=sub.index)
    q_upper = pd.Series(np.nan, index=sub.index)

    for fold in sub["fold"].unique():
        calib = sub[sub["fold"] != fold]

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
    model trained on its own label for).
    """
    lower_q, upper_q = _quantile_bounds()
    sub = fold_predictions[fold_predictions["model"] == model_name]
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
    """
    lower_q, upper_q = _quantile_bounds()
    sub = fold_predictions[fold_predictions["model"] == model_name].copy()
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
