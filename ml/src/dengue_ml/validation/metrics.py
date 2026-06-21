import numpy as np
import pandas as pd

from dengue_ml.config import CITY_COL, TARGET


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0) -> float:
    """Safe MAPE: skip rows where y_true < eps to avoid division by near-zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true >= eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae":  calculate_mae(y_true, y_pred),
        "rmse": calculate_rmse(y_true, y_pred),
        "mape": calculate_mape(y_true, y_pred),
    }


def compute_fold_metrics(fold_predictions: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (fold, model, city) with mae/rmse/mape -- same shape as
    run_nested_cv's fold_metrics, but as a reusable function instead of an
    inline loop, so it also works unmodified on validation/autoregressive_cv.
    run_autoregressive_cv's fold_predictions_ar (same fold/model/city_name/
    casos_est/predicted columns, just built from the autoregressive rollout
    instead of 1-step-ahead predictions).
    """
    rows = []
    for (fold, model, city), group in fold_predictions.groupby(["fold", "model", CITY_COL]):
        m = calculate_all_metrics(group[TARGET].values, group["predicted"].values)
        rows.append({"fold": fold, "model": model, "city": city, **m})
    return pd.DataFrame(rows)
