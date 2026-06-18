import numpy as np
import pandas as pd

from dengue_ml.validation.conditional_residuals import (
    assign_loFo_conditional_ci, compute_regime_coverage, PROXY_COL,
)


def _fake_fold_predictions(n_folds: int = 3, rows_per_fold: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for fold in range(1, n_folds + 1):
        for i in range(rows_per_fold):
            predicted = float(rng.uniform(10, 100))
            casos_est = float(np.expm1(np.log1p(predicted) + rng.normal(0, 0.3)))
            rows.append({
                "fold": fold,
                "model": "xgb_cases_only",
                "casos_est": casos_est,
                "predicted": predicted,
                PROXY_COL: 2 if i % 4 == 0 else 0,  # ~25% high regime, like production
            })
    return pd.DataFrame(rows)


def test_assign_lofo_default_matches_production_proxy():
    fold_predictions = _fake_fold_predictions()
    result = assign_loFo_conditional_ci(fold_predictions, "xgb_cases_only")
    assert {"lower_95", "upper_95"}.issubset(result.columns)
    assert result["lower_95"].notna().all()
    assert (result["lower_95"] <= result["predicted"]).all()
    assert (result["upper_95"] >= result["predicted"]).all()


def test_assign_lofo_with_override_mask_changes_calibration_pool():
    # NOTE: a mask that's a complete inversion of the default split is an
    # observational no-op (every row's "same-label calib pool" maps onto the
    # same physical rows, just renamed) -- not a meaningful contrast. Use an
    # asymmetric override (force everything into one bucket) instead, which
    # genuinely changes which rows calibrate each held-out row's quantile.
    fold_predictions = _fake_fold_predictions()
    sub = fold_predictions[fold_predictions["model"] == "xgb_cases_only"]
    all_high_mask = pd.Series(True, index=sub.index)

    default_result  = assign_loFo_conditional_ci(fold_predictions, "xgb_cases_only")
    override_result = assign_loFo_conditional_ci(fold_predictions, "xgb_cases_only", high_regime_mask=all_high_mask)

    high_idx = sub.index[(sub[PROXY_COL] >= 2).values]
    assert not np.allclose(
        default_result.loc[high_idx, "lower_95"].values,
        override_result.set_index(sub.index).loc[high_idx, "lower_95"].values,
    )


def test_compute_regime_coverage_default_proxy():
    fold_predictions = _fake_fold_predictions()
    result = compute_regime_coverage(fold_predictions, "xgb_cases_only")
    assert 0.0 <= result["coverage"] <= 1.0
    assert result["n_high"] + result["n_low"] == len(fold_predictions)


def test_compute_regime_coverage_with_custom_mask():
    fold_predictions = _fake_fold_predictions()
    sub = fold_predictions[fold_predictions["model"] == "xgb_cases_only"]
    all_low_mask = pd.Series(False, index=sub.index)

    result = compute_regime_coverage(fold_predictions, "xgb_cases_only", high_regime_mask=all_low_mask)
    assert result["n_high"] == 0
    assert result["n_low"] == len(sub)
