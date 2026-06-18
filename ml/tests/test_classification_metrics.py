import numpy as np

from dengue_ml.validation.classification_metrics import (
    calculate_precision,
    calculate_recall,
    calculate_f1,
    calculate_roc_auc,
    calculate_all_classification_metrics,
)


def test_calculate_precision_and_recall():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert calculate_precision(y_true, y_pred) == 0.5
    assert calculate_recall(y_true, y_pred) == 0.5


def test_calculate_precision_zero_division_returns_zero():
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    assert calculate_precision(y_true, y_pred) == 0.0
    assert calculate_recall(y_true, y_pred) == 0.0


def test_calculate_f1_perfect_prediction():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    assert calculate_f1(y_true, y_pred) == 1.0


def test_calculate_roc_auc_nan_when_single_class():
    y_true = np.array([1, 1, 1])
    y_proba = np.array([0.2, 0.6, 0.9])
    assert np.isnan(calculate_roc_auc(y_true, y_proba))


def test_calculate_roc_auc_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])
    assert calculate_roc_auc(y_true, y_proba) == 1.0


def test_calculate_all_classification_metrics_returns_expected_keys():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    result = calculate_all_classification_metrics(y_true, y_pred, y_proba=np.array([0.9, 0.1, 0.8, 0.2]))
    assert set(result.keys()) == {"precision", "recall", "f1", "auc"}
    assert result["auc"] == 1.0


def test_calculate_all_classification_metrics_auc_nan_without_proba():
    y_true = np.array([1, 0])
    y_pred = np.array([1, 0])
    result = calculate_all_classification_metrics(y_true, y_pred)
    assert np.isnan(result["auc"])
