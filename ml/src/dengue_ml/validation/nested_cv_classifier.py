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

# Benchmark-only rules, scored on the same rows as the trained classifier for
# a fair comparison -- NOT used as model inputs (nivel_inc isn't in
# FEATURE_COLS; see feature_pipeline.py) and NOT the production CI-regime
# proxy anymore (that's now the trained classifier's predicted_proba itself,
# see conditional_residuals.py). nivel_inc_week_t-1 comes from meta_te (a
# side channel feature_pipeline.py attaches outside of X specifically so it
# stays available for this comparison without being a model input).
# sustained_rt_week_t-1 IS still a model input (sustained_rt wasn't removed),
# so it's read from X_te as before.
NIVEL_INC_RULE_FEATURE   = "nivel_inc_week_t-1"
NIVEL_INC_RULE_THRESHOLD = 2
SUSTAINED_RT_FEATURE = "sustained_rt_week_t-1"


def run_nested_cv_classifier(
    df: pd.DataFrame,
    model_names: list[str] = CLASSIFIER_MODEL_NAMES,
    feature_set: str = CLASSIFICATION_FEATURE_SET,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Nested rolling CV for the binary "epidemic this week" classifier --
    structurally separate from run_nested_cv() (regression point models), but
    reusing the identical outer/inner fold protocol (make_outer_splits /
    make_inner_splits on the same df) so results are directly comparable and
    leak-free in the same way.

    Returns
    -------
    fold_metrics     : one row per (fold, model) -- precision/recall/f1/auc
                        plus the two rule-based proxies' precision/recall/f1
                        on the SAME rows, for a fair side-by-side comparison.
    fold_predictions : one row per (fold, model, city, week) with
                        predicted_proba, is_epidemic (true label),
                        nivel_inc_rule, sustained_rt_rule.
    best_hyperparams : one row per (fold, model) -- mirrors run_nested_cv's
                        best_hyperparams, so a fold's already-tuned
                        classifier can be retrained later (e.g. by
                        validation/autoregressive_cv.py) without re-running
                        the random search.
    """
    outer_splits = make_outer_splits(df)
    all_metrics = []
    all_preds   = []
    all_hparams = []

    for fold_idx, (outer_train, outer_test) in enumerate(outer_splits):
        print(f"\n=== [classifier] Outer fold {fold_idx + 1}/{len(outer_splits)} "
              f"(test: {outer_test['week_start'].min().date()} – "
              f"{outer_test['week_start'].max().date()}) ===")

        inner_splits = make_inner_splits(outer_train)

        for model_name in model_names:
            print(f"  [{model_name}] ... ", end="", flush=True)
            try:
                preds_df, best_params = _run_one_classifier(model_name, outer_train, outer_test, inner_splits, feature_set)
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
            all_hparams.append({"fold": fold_idx + 1, "model": model_name, **best_params})
            print("done")

    fold_metrics     = pd.DataFrame(all_metrics)
    fold_predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    best_hyperparams = pd.DataFrame(all_hparams)
    return fold_metrics, fold_predictions, best_hyperparams


def _run_one_classifier(
    model_name: str,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    inner_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    feature_set: str,
) -> tuple[pd.DataFrame, dict]:
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
    preds_df["nivel_inc_rule"]     = (meta_te[NIVEL_INC_RULE_FEATURE] >= NIVEL_INC_RULE_THRESHOLD).astype(int).values
    preds_df["sustained_rt_rule"]  = (X_te[SUSTAINED_RT_FEATURE] >= 0.5).astype(int).values
    return preds_df, best_params
