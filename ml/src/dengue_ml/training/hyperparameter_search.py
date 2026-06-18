import numpy as np
import pandas as pd

from dengue_ml.config import RANDOM_SEED, XGB_N_TRIALS, XRFM_N_TRIALS, CLASSIFICATION_N_TRIALS
from dengue_ml.features.feature_pipeline import build_features_for_split, build_classification_features_for_split
from dengue_ml.models.xgboost_models import train_xgb, predict_xgb
from dengue_ml.models.xrfm_models import train_xrfm, predict_xrfm
from dengue_ml.models.classifier_models import (
    train_logreg, predict_proba_logreg, train_xgb_clf, predict_proba_xgb_clf,
)
from dengue_ml.validation.metrics import calculate_mae
from dengue_ml.validation.classification_metrics import calculate_f1
from dengue_ml.training_config import load_training_config


def random_search_xgb(
    df_train: pd.DataFrame,
    feature_set: str,
    inner_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    param_distributions: dict | None = None,
    n_trials: int = XGB_N_TRIALS,
) -> dict:
    """
    Random search for XGBoost hyperparameters using inner rolling splits.
    Returns the best parameter dict (by mean MAE across inner folds, original scale).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    dists = param_distributions or load_training_config()["xgboost"]["param_distributions"]

    best_mae    = float("inf")
    best_params = {}

    for _ in range(n_trials):
        trial_params = {k: rng.choice(v).item() for k, v in dists.items()}
        fold_maes = []

        for inner_train_df, inner_val_df in inner_splits:
            try:
                X_tr, y_tr, X_val, y_val, _, _ = build_features_for_split(
                    inner_train_df, inner_val_df, feature_set
                )
                if X_tr.empty or X_val.empty:
                    continue
                model = train_xgb(X_tr, y_tr, params=trial_params)
                preds = np.expm1(predict_xgb(model, X_val))
                true  = np.expm1(y_val.values)
                fold_maes.append(calculate_mae(true, preds))
            except Exception:
                pass

        if fold_maes:
            mean_mae = float(np.mean(fold_maes))
            if mean_mae < best_mae:
                best_mae    = mean_mae
                best_params = trial_params

    return best_params


def random_search_xrfm(
    df_train: pd.DataFrame,
    feature_set: str,
    inner_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    param_distributions: dict | None = None,
    n_trials: int = XRFM_N_TRIALS,
) -> dict:
    """
    Random search for xRFM hyperparameters using inner rolling splits.
    Each inner (train, val) pair maps directly onto xRFM's required
    X_train/y_train/X_val/y_val fit() signature.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    dists = param_distributions or load_training_config()["xrfm"]["param_distributions"]

    best_mae    = float("inf")
    best_params = {}

    for _ in range(n_trials):
        trial_params = {k: rng.choice(v).item() for k, v in dists.items()}
        fold_maes = []

        for inner_train_df, inner_val_df in inner_splits:
            try:
                X_tr, y_tr, X_val, y_val, _, _ = build_features_for_split(
                    inner_train_df, inner_val_df, feature_set
                )
                if X_tr.empty or X_val.empty:
                    continue
                model = train_xrfm(X_tr, y_tr, X_val, y_val, params=trial_params)
                preds = np.expm1(predict_xrfm(model, X_val))
                true  = np.expm1(y_val.values)
                fold_maes.append(calculate_mae(true, preds))
            except Exception:
                pass

        if fold_maes:
            mean_mae = float(np.mean(fold_maes))
            if mean_mae < best_mae:
                best_mae    = mean_mae
                best_params = trial_params

    return best_params


def random_search_logreg(
    df_train: pd.DataFrame,
    feature_set: str,
    inner_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    param_distributions: dict | None = None,
    n_trials: int = CLASSIFICATION_N_TRIALS,
) -> dict:
    """Random search for logistic-regression hyperparameters using inner
    rolling splits. Returns the best parameter dict (by mean F1 across inner
    folds, thresholded at 0.5)."""
    rng = np.random.default_rng(RANDOM_SEED)
    dists = param_distributions or load_training_config()["logreg"]["param_distributions"]

    best_f1     = -1.0
    best_params = {}

    for _ in range(n_trials):
        trial_params = {k: rng.choice(v).item() for k, v in dists.items()}
        fold_f1s = []

        for inner_train_df, inner_val_df in inner_splits:
            try:
                X_tr, y_tr, X_val, y_val, _, _ = build_classification_features_for_split(
                    inner_train_df, inner_val_df, feature_set
                )
                if X_tr.empty or X_val.empty or y_tr.nunique() < 2:
                    continue
                model = train_logreg(X_tr, y_tr, params=trial_params)
                proba = predict_proba_logreg(model, X_val)
                fold_f1s.append(calculate_f1(y_val.values, (proba >= 0.5).astype(int)))
            except Exception:
                pass

        if fold_f1s:
            mean_f1 = float(np.mean(fold_f1s))
            if mean_f1 > best_f1:
                best_f1     = mean_f1
                best_params = trial_params

    return best_params


def random_search_xgb_clf(
    df_train: pd.DataFrame,
    feature_set: str,
    inner_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    param_distributions: dict | None = None,
    n_trials: int = CLASSIFICATION_N_TRIALS,
) -> dict:
    """Random search for XGBClassifier hyperparameters using inner rolling
    splits. Returns the best parameter dict (by mean F1 across inner folds,
    thresholded at 0.5)."""
    rng = np.random.default_rng(RANDOM_SEED)
    dists = param_distributions or load_training_config()["xgb_classifier"]["param_distributions"]

    best_f1     = -1.0
    best_params = {}

    for _ in range(n_trials):
        trial_params = {k: rng.choice(v).item() for k, v in dists.items()}
        fold_f1s = []

        for inner_train_df, inner_val_df in inner_splits:
            try:
                X_tr, y_tr, X_val, y_val, _, _ = build_classification_features_for_split(
                    inner_train_df, inner_val_df, feature_set
                )
                if X_tr.empty or X_val.empty or y_tr.nunique() < 2:
                    continue
                model = train_xgb_clf(X_tr, y_tr, params=trial_params)
                proba = predict_proba_xgb_clf(model, X_val)
                fold_f1s.append(calculate_f1(y_val.values, (proba >= 0.5).astype(int)))
            except Exception:
                pass

        if fold_f1s:
            mean_f1 = float(np.mean(fold_f1s))
            if mean_f1 > best_f1:
                best_f1     = mean_f1
                best_params = trial_params

    return best_params
