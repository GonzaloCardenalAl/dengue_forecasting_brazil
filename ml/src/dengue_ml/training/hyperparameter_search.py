import numpy as np
import pandas as pd

from dengue_ml.config import RANDOM_SEED, XGB_N_TRIALS, CITY_COL
from dengue_ml.features.feature_pipeline import build_features
from dengue_ml.models.xgboost_models import train_xgb, predict_xgb
from dengue_ml.validation.metrics import calculate_mae
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
                # Fit features on inner train only (climate stats, etc.)
                X_tr, y_tr, _, climate_stats = build_features(inner_train_df, feature_set)

                # Build val features using combined history so lag features are valid
                combined = pd.concat(
                    [inner_train_df, inner_val_df], ignore_index=True
                ).sort_values([CITY_COL, "quarter_start"])
                X_all, y_all, meta_all, _ = build_features(
                    combined, feature_set, climate_fit_stats=climate_stats
                )
                val_qs  = set(inner_val_df["quarter_start"].unique())
                val_mask = meta_all["quarter_start"].isin(val_qs)
                X_val  = X_all[val_mask]
                y_val  = y_all[val_mask]

                if X_tr.empty or X_val.empty:
                    continue
                model = train_xgb(X_tr, y_tr, params=trial_params)
                preds_log = predict_xgb(model, X_val)
                preds = np.expm1(preds_log)
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
