import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from dengue_ml.config import CITY_COL, TARGET, CITIES
from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.features.feature_pipeline import build_features, build_features_for_split, FEATURE_COLS
from dengue_ml.models.xgboost_models import train_xgb
from dengue_ml.models.xrfm_models import train_xrfm
from dengue_ml.models.sarima import fit_sarima, tune_sarima
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
            train_s  = city_train.set_index("quarter_start")[TARGET].sort_index()
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
        # early stopping) -- carve the most-recent-horizon-quarters tail off
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


def select_best_model(fold_metrics: pd.DataFrame) -> tuple[str, float]:
    """Return (best_model_name, mean_mae) based on mean MAE across outer folds and cities."""
    summary = fold_metrics.groupby("model")["mae"].mean()
    best = summary.idxmin()
    return best, float(summary[best])
