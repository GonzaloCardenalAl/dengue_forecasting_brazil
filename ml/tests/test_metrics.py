import numpy as np

from dengue_ml.validation.metrics import (
    calculate_mae,
    calculate_mape,
    calculate_rmse,
    calculate_all_metrics,
)


def test_calculate_mae():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    assert calculate_mae(y_true, y_pred) == 7 / 3


def test_calculate_rmse():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([3.0, 4.0])
    assert calculate_rmse(y_true, y_pred) == np.sqrt(12.5)


def test_calculate_mape_skips_near_zero_true_values():
    y_true = np.array([0.5, 100.0])
    y_pred = np.array([1.0, 110.0])
    # row 0 has y_true < eps (default 1.0) and is excluded
    assert calculate_mape(y_true, y_pred) == 10.0


def test_calculate_mape_returns_nan_when_all_rows_excluded():
    y_true = np.array([0.1, 0.2])
    y_pred = np.array([0.1, 0.2])
    assert np.isnan(calculate_mape(y_true, y_pred))


def test_calculate_all_metrics_returns_expected_keys():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([10.0, 20.0])
    result = calculate_all_metrics(y_true, y_pred)
    assert set(result.keys()) == {"mae", "rmse", "mape"}
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
