import numpy as np
import pandas as pd
import pytest

import dengue_ml.validation.nested_cv_classifier as nested_cv_classifier
from dengue_ml.validation.nested_cv_classifier import _run_one_classifier
from dengue_ml.validation.conditional_residuals import PROXY_SOURCE_FEATURE


def _fake_split(*_args, **_kwargs):
    """Stand-in for build_classification_features_for_split: avoids touching
    the real weekly CSV (feature_pipeline.py's lag features are keyed off
    preprocessing.get_weekly_table(), not the df passed in -- not mockable
    cheaply at the unit-test level), while still exercising the real rule-
    derivation/orchestration logic in _run_one_classifier."""
    n_tr, n_te = 20, 4
    rng = np.random.default_rng(0)
    X_tr = pd.DataFrame({
        "year": rng.integers(2015, 2024, n_tr),
        PROXY_SOURCE_FEATURE: [0, 1, 2, 0] * (n_tr // 4),
        "sustained_rt_week_t-1": [0.0, 0.0, 1.0, 1.0] * (n_tr // 4),
    })
    y_tr = pd.Series([0, 0, 1, 0] * (n_tr // 4))
    X_te = pd.DataFrame({
        "year": [2024] * n_te,
        PROXY_SOURCE_FEATURE: [0, 1, 2, 2],
        "sustained_rt_week_t-1": [0.0, 0.0, 1.0, 0.0],
    })
    y_te = pd.Series([0, 0, 1, 1])
    meta_te = pd.DataFrame({
        "city_name": ["A", "A", "B", "B"],
        "month_start": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]),
    })
    return X_tr, y_tr, X_te, y_te, meta_te, None


def test_run_one_classifier_attaches_rule_columns_from_features(monkeypatch):
    monkeypatch.setattr(
        nested_cv_classifier, "build_classification_features_for_split", _fake_split,
    )

    preds_df = _run_one_classifier("logreg", pd.DataFrame(), pd.DataFrame(), [], "cases_only")

    assert list(preds_df["nivel_inc_rule"])    == [0, 0, 1, 1]   # >= REGIME_THRESHOLD (2)
    assert list(preds_df["sustained_rt_rule"]) == [0, 0, 1, 0]   # >= 0.5
    assert list(preds_df["is_epidemic"])       == [0, 0, 1, 1]
    assert preds_df["predicted_proba"].between(0, 1).all()
    assert set(preds_df.columns) >= {
        "city_name", "month_start", "predicted_proba", "is_epidemic",
        "nivel_inc_rule", "sustained_rt_rule",
    }


def test_run_one_classifier_unknown_model_name_raises(monkeypatch):
    monkeypatch.setattr(
        nested_cv_classifier, "build_classification_features_for_split", _fake_split,
    )
    with pytest.raises(ValueError):
        _run_one_classifier("not_a_model", pd.DataFrame(), pd.DataFrame(), [], "cases_only")
