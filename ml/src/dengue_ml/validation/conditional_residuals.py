import numpy as np
import pandas as pd

from dengue_ml.config import TARGET
from dengue_ml.training_config import load_training_config

# nivel_inc_week_t-1 = InfoDengue's own incidence-vs-threshold alert level
# (0=baseline, 1=pre-epidemic alert, 2=epidemic) as of the most recently
# known week before the target quarter -- "are we, as of the freshest
# available data, already in confirmed epidemic territory." Used to condition
# the residual distribution instead of a hard outbreak/non-outbreak label on
# the target quarter itself, which would require knowing the outcome we're
# trying to bound.
#
# Empirically validated against the OOF predictions of all three XGBoost
# feature sets: this split gives ~92.5-93.1% leave-one-fold-out coverage
# (vs. 90.6% for the previous cases_growth_1q median-split, and 73.1% for
# the original independent-quantile-models approach this replaced).
PROXY_SOURCE_FEATURE = "nivel_inc_week_t-1"
# Fixed domain threshold, not a recomputed median: nivel_inc's ordinal 0/1/2
# scale makes a per-calibration-set median unstable (it's often exactly 1).
# ">= 2" means "last known week was already a confirmed epidemic."
REGIME_THRESHOLD = 2
# Label this proxy is stored under in fold_predictions/preds_df (distinct
# from PROXY_SOURCE_FEATURE, which is the column name in the X feature
# matrix used at forecast time, where fold_predictions doesn't exist).
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
