import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
)


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(precision_score(y_true, y_pred, zero_division=0))


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, zero_division=0))


def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, zero_division=0))


def calculate_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """NaN (not 0.5) if y_true has only one class -- sklearn can't define AUC then."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def calculate_all_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """y_pred is the thresholded binary label; y_proba (optional) enables AUC
    for probabilistic predictors -- rule-based proxies have no probability,
    so AUC is left NaN for those."""
    metrics = {
        "precision": calculate_precision(y_true, y_pred),
        "recall":    calculate_recall(y_true, y_pred),
        "f1":        calculate_f1(y_true, y_pred),
    }
    metrics["auc"] = calculate_roc_auc(y_true, y_proba) if y_proba is not None else float("nan")
    return metrics
