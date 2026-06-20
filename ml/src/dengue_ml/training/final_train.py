import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from dengue_ml.config import CITY_COL, TARGET, CITIES, CLASSIFICATION_FEATURE_SET
from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.features.feature_pipeline import (
    build_features, build_features_for_split, build_classification_features, FEATURE_COLS,
)
from dengue_ml.models.xgboost_models import train_xgb
from dengue_ml.models.xrfm_models import train_xrfm
from dengue_ml.models.sarima import fit_sarima, tune_sarima
from dengue_ml.models.classifier_models import (
    train_logreg, predict_proba_logreg, train_xgb_clf, predict_proba_xgb_clf,
)
from dengue_ml.validation.time_splits import make_inner_splits
from dengue_ml.validation.conditional_residuals import compute_residual_quantile_table, PROXY_COL


def train_final_model(
    selected_model_name: str,
    selected_params: dict | None = None,
    outputs_dir: Path | None = None,
    df: pd.DataFrame | None = None,
    fold_predictions: pd.DataFrame | None = None,
) -> dict:
    """
    Train the final selected model on all available historical data.

    Returns dict with 'model' (or 'models' for SARIMA per-city dict),
    'feature_set', and optionally quantile models for XGBoost.
    """
    if outputs_dir is None:
        from dengue_ml.run_dir import get_latest_run_dir
        outputs_dir = get_latest_run_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = prepare_model_table()

    artifact: dict = {"model_name": selected_model_name}

    if selected_model_name == "baseline":
        # Baseline needs no training — store the training data for lookup
        artifact["train_df"] = df
        joblib.dump(artifact, outputs_dir / "final_model.pkl")
        return artifact

    if selected_model_name == "sarima":
        inner_splits = make_inner_splits(df)
        city_models  = {}
        for city in CITIES:
            city_train = df[df[CITY_COL] == city].copy()
            order, sorder = tune_sarima(city_train, inner_splits, city=city)
            train_s  = city_train.set_index("week_start")[TARGET].sort_index()
            city_models[city] = fit_sarima(np.log1p(train_s), order, sorder)
        artifact["models"] = city_models
        joblib.dump(artifact, outputs_dir / "final_model.pkl")
        return artifact

    from dengue_ml.config import FEATURE_SET_FOR_MODEL
    feature_set = FEATURE_SET_FOR_MODEL[selected_model_name]

    if selected_model_name.startswith("xgb"):
        X_tr, y_tr, _, climate_stats = build_features(df, feature_set)
        feature_cols = list(X_tr.columns)
        model = train_xgb(X_tr, y_tr, params=selected_params)
    else:
        # xRFM requires a held-out X_val/y_val for fit() (used internally for
        # early stopping) -- carve the most-recent-horizon-weeks tail off
        # of df the same way nested_cv.py does. climate_stats for the
        # production artifact must still reflect the FULL df (used later at
        # forecast time), not just train_tail, so it's fit separately here.
        inner_splits = make_inner_splits(df)
        if not inner_splits:
            raise ValueError(
                f"Not enough history to carve an xRFM val split for {selected_model_name}."
            )
        train_tail, val_tail = inner_splits[-1]
        X_tr, y_tr, X_val, y_val, _, _ = build_features_for_split(train_tail, val_tail, feature_set)
        _, _, _, climate_stats = build_features(df, feature_set)
        feature_cols = list(X_tr.columns)
        model = train_xrfm(X_tr, y_tr, X_val, y_val, params=selected_params)

    residual_quantiles = None
    if fold_predictions is not None:
        model_oof = fold_predictions[fold_predictions["model"] == selected_model_name]
        if not model_oof.empty and PROXY_COL in model_oof.columns and model_oof[PROXY_COL].notna().any():
            residual_quantiles = compute_residual_quantile_table(fold_predictions, selected_model_name)

    artifact.update({
        "model":              model,
        "residual_quantiles": residual_quantiles,
        "feature_set":        feature_set,
        "feature_cols":       feature_cols,
        "climate_stats":      climate_stats,
    })

    joblib.dump(artifact, outputs_dir / "final_model.pkl")

    # Save feature column list separately for inspection
    with open(outputs_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"Final model saved → {outputs_dir / 'final_model.pkl'}")
    return artifact


def select_best_classifier(fold_metrics_clf: pd.DataFrame) -> tuple[str, float]:
    """
    Return (best_model_name, mean_auc) for the epidemic classifier -- ranked
    by mean AUC across outer folds. AUC (not F1/precision/recall at the fixed
    0.5 threshold) is the right selection criterion here because the
    classifier's output is consumed as a continuous probability
    (`growth_proxy`, see conditional_residuals.py), not just a thresholded
    label -- AUC measures how well-ordered those probabilities are across
    the full range, which is what the downstream regime split actually needs.
    """
    mean_auc = fold_metrics_clf.groupby("model")["auc"].mean()
    best = mean_auc.idxmax()
    return best, float(mean_auc.loc[best])


def train_final_classifier(
    selected_model_name: str,
    df: pd.DataFrame,
    selected_params: dict | None = None,
    feature_set: str = CLASSIFICATION_FEATURE_SET,
    outputs_dir: Path | None = None,
) -> dict:
    """
    Train the final epidemic classifier on all available historical data and
    persist it as its own artifact (final_classifier.pkl, parallel to
    final_model.pkl) -- this is the model whose predicted probability becomes
    the production CI-regime proxy (growth_proxy) both for the calibration
    table (via OOF predictions, see conditional_residuals.py) and for live
    forecast-time inference (forecasting/forecast_next_52w.py).
    """
    if outputs_dir is None:
        from dengue_ml.run_dir import get_latest_run_dir
        outputs_dir = get_latest_run_dir()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, _, _ = build_classification_features(df, feature_set)

    if selected_model_name == "logreg":
        model = train_logreg(X_tr, y_tr, params=selected_params)
    elif selected_model_name == "xgb_clf":
        model = train_xgb_clf(X_tr, y_tr, params=selected_params)
    else:
        raise ValueError(f"Unknown classifier model_name '{selected_model_name}'.")

    artifact = {
        "model_name":   selected_model_name,
        "model":        model,
        "feature_set":  feature_set,
        "feature_cols": list(X_tr.columns),
    }
    joblib.dump(artifact, outputs_dir / "final_classifier.pkl")
    print(f"Final classifier saved → {outputs_dir / 'final_classifier.pkl'}")
    return artifact


def predict_proba_classifier(artifact: dict, X: pd.DataFrame) -> np.ndarray:
    """Dispatch to the right predict_proba_* for a final_classifier.pkl artifact."""
    if artifact["model_name"] == "logreg":
        return predict_proba_logreg(artifact["model"], X)
    if artifact["model_name"] == "xgb_clf":
        return predict_proba_xgb_clf(artifact["model"], X)
    raise ValueError(f"Unknown classifier model_name '{artifact['model_name']}'.")


def select_best_model(fold_metrics: pd.DataFrame) -> tuple[str, float]:
    """
    Return (best_model_name, median_mae) based on average rank of median MAE
    and std MAE across outer folds and cities (lowest average rank wins,
    ties broken by lowest median MAE). Median MAE alone picks whichever
    model handles a typical week best, but mean MAE is dominated by the
    single 2024 outbreak fold (~7x any prior year), so among models with a
    comparable median, the steadier one (lower std) is preferable for
    production use. Ranking instead of a fixed tolerance band on the median
    avoids tuning a cutoff to one run's specific gap.
    """
    summary = fold_metrics.groupby("model")["mae"].agg(["median", "std"])
    avg_rank = (summary["median"].rank() + summary["std"].rank()) / 2
    tied = avg_rank[avg_rank == avg_rank.min()].index
    best = summary.loc[tied, "median"].idxmin()
    return best, float(summary.loc[best, "median"])
