import numpy as np
import pandas as pd

from dengue_ml.config import (
    CITY_COL, CLASSIFIER_MODEL_NAMES, CLASSIFICATION_FEATURE_SET,
)
from dengue_ml.features.feature_pipeline import build_classification_features_for_split
from dengue_ml.models.classifier_models import (
    train_logreg, predict_proba_logreg, train_xgb_clf, predict_proba_xgb_clf,
)
from dengue_ml.training.hyperparameter_search import random_search_logreg, random_search_xgb_clf
from dengue_ml.validation.time_splits import make_outer_splits, make_inner_splits
from dengue_ml.validation.classification_metrics import calculate_all_classification_metrics
from dengue_ml.validation.conditional_residuals import PROXY_SOURCE_FEATURE, REGIME_THRESHOLD

# Same alert-lag column the regression pipeline's CI-regime proxy reads
# (PROXY_SOURCE_FEATURE), plus its sustained_rt counterpart -- both are
# already present in X_te (part of the cases_only feature set's weekly alert
# lags), so they're pulled straight off the classifier's own rows rather than
# recomputed. This is what makes the precision/recall/F1 comparison across
# {nivel_inc rule, sustained_rt rule, trained classifier} fair: identical
# (city, quarter, fold) population for every candidate.
SUSTAINED_RT_FEATURE = "sustained_rt_week_t-1"


def run_nested_cv_classifier(
    df: pd.DataFrame,
    model_names: list[str] = CLASSIFIER_MODEL_NAMES,
    feature_set: str = CLASSIFICATION_FEATURE_SET,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Nested rolling CV for the binary "epidemic this quarter" classifier --
    structurally separate from run_nested_cv() (regression point models), but
    reusing the identical outer/inner fold protocol (make_outer_splits /
    make_inner_splits on the same df) so results are directly comparable and
    leak-free in the same way.

    Returns
    -------
    fold_metrics     : one row per (fold, model) -- precision/recall/f1/auc
                        plus the two rule-based proxies' precision/recall/f1
                        on the SAME rows, for a fair side-by-side comparison.
    fold_predictions : one row per (fold, model, city, quarter) with
                        predicted_proba, is_epidemic (true label),
                        nivel_inc_rule, sustained_rt_rule.
    """
    outer_splits = make_outer_splits(df)
    all_metrics = []
    all_preds   = []

    for fold_idx, (outer_train, outer_test) in enumerate(outer_splits):
        print(f"\n=== [classifier] Outer fold {fold_idx + 1}/{len(outer_splits)} "
              f"(test: {outer_test['quarter_start'].min().date()} – "
              f"{outer_test['quarter_start'].max().date()}) ===")

        inner_splits = make_inner_splits(outer_train)

        for model_name in model_names:
            print(f"  [{model_name}] ... ", end="", flush=True)
            try:
                preds_df = _run_one_classifier(model_name, outer_train, outer_test, inner_splits, feature_set)
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            preds_df["fold"]  = fold_idx + 1
            preds_df["model"] = model_name
            all_preds.append(preds_df)

            m = calculate_all_classification_metrics(
                preds_df["is_epidemic"].values,
                (preds_df["predicted_proba"].values >= 0.5).astype(int),
                preds_df["predicted_proba"].values,
            )
            rule_metrics = {}
            for rule_col in ("nivel_inc_rule", "sustained_rt_rule"):
                rm = calculate_all_classification_metrics(
                    preds_df["is_epidemic"].values, preds_df[rule_col].values,
                )
                rule_metrics.update({f"{rule_col}_{k}": v for k, v in rm.items() if k != "auc"})

            all_metrics.append({"fold": fold_idx + 1, "model": model_name, **m, **rule_metrics})
            print("done")

    fold_metrics     = pd.DataFrame(all_metrics)
    fold_predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    return fold_metrics, fold_predictions


def _run_one_classifier(
    model_name: str,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    inner_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_set: str,
) -> pd.DataFrame:
    X_tr, y_tr, X_te, y_te, meta_te, _ = build_classification_features_for_split(
        outer_train, outer_test, feature_set
    )

    if model_name == "logreg":
        best_params = random_search_logreg(outer_train, feature_set, inner_splits)
        model = train_logreg(X_tr, y_tr, params=best_params)
        proba = predict_proba_logreg(model, X_te)
    elif model_name == "xgb_clf":
        best_params = random_search_xgb_clf(outer_train, feature_set, inner_splits)
        model = train_xgb_clf(X_tr, y_tr, params=best_params)
        proba = predict_proba_xgb_clf(model, X_te)
    else:
        raise ValueError(f"Unknown classifier model_name '{model_name}'.")

    preds_df = meta_te.copy()
    preds_df["predicted_proba"]    = proba
    preds_df["is_epidemic"]        = y_te.values
    preds_df["nivel_inc_rule"]     = (X_te[PROXY_SOURCE_FEATURE] >= REGIME_THRESHOLD).astype(int).values
    preds_df["sustained_rt_rule"]  = (X_te[SUSTAINED_RT_FEATURE] >= 0.5).astype(int).values
    return preds_df
