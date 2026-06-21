"""Inference-only forecast generation: load an already-trained model artifact
and produce the 4-quarter deliverable. Does NOT fit/retrain anything -- model
weights are assumed to already exist in `run_dir` (e.g. produced by
train_final_model.py, or pushed in from an external training run).

Shared by ml/scripts/generate_forecasts.py (full run, with plots) and the
DORA API's /admin/refresh endpoint (data-refresh-triggered re-forecast, no
plots needed since the dashboard renders its own charts from these CSVs).
"""
from pathlib import Path

import joblib
import pandas as pd

from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.forecasting.forecast_next_52w import generate_next_52w_forecast
from dengue_ml.forecasting.quarterly_aggregation import aggregate_weekly_forecast_to_quarterly
from dengue_ml.validation.conditional_residuals import compute_quarterly_residual_quantile_table, PROXY_COL
from dengue_ml.reporting.results_tables import final_forecast_table


def run_inference(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load run_dir's final_model.pkl (+ optional final_classifier.pkl) and
    run inference only. Regenerates the 52-week forecast from whatever's
    currently in the raw CSV, aggregates it to the quarterly deliverable,
    writes final_weekly_forecast.csv + final_quarterly_forecast.csv (+ the
    _horizon_aware variant if available) back into run_dir, and returns
    (weekly_df, quarterly_df).
    """
    model_path = run_dir / "final_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found -- push trained model weights into this run dir first."
        )
    artifact = joblib.load(model_path)
    df = prepare_model_table()

    classifier_path = run_dir / "final_classifier.pkl"
    classifier_artifact = joblib.load(classifier_path) if classifier_path.exists() else None

    weekly_forecast_df = generate_next_52w_forecast(artifact, df, classifier_artifact=classifier_artifact)

    quarterly_residual_quantiles = None
    fold_predictions_path = run_dir / "fold_predictions.csv"
    if fold_predictions_path.exists():
        fold_predictions = pd.read_csv(fold_predictions_path, parse_dates=["week_start"])
        model_oof = fold_predictions[fold_predictions["model"] == artifact["model_name"]]
        if not model_oof.empty and PROXY_COL in model_oof.columns and model_oof[PROXY_COL].notna().any():
            quarterly_residual_quantiles = compute_quarterly_residual_quantile_table(
                fold_predictions, artifact["model_name"]
            )

    quarterly_forecast_df = aggregate_weekly_forecast_to_quarterly(
        weekly_forecast_df, quarterly_residual_quantiles
    )

    final_forecast_table(weekly_forecast_df, outputs_dir=run_dir)
    final_forecast_table(
        quarterly_forecast_df, outputs_dir=run_dir,
        period_col="forecast_quarter", filename="final_quarterly_forecast.csv",
    )

    horizon_quantiles = artifact.get("horizon_quantiles")
    if horizon_quantiles is not None:
        quarterly_forecast_df_ha = aggregate_weekly_forecast_to_quarterly(
            weekly_forecast_df, quarterly_residual_quantiles, horizon_bucketed_quantiles=horizon_quantiles,
        )
        final_forecast_table(
            quarterly_forecast_df_ha, outputs_dir=run_dir,
            period_col="forecast_quarter", filename="final_quarterly_forecast_horizon_aware.csv",
        )

    return weekly_forecast_df, quarterly_forecast_df
