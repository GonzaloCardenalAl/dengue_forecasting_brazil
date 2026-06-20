#!/usr/bin/env python
"""
Load the trained final model, generate a 52-week-ahead forecast with 95% CI,
then aggregate it to the quarterly deliverable (sum ~13 weeks -> 1 quarter,
with a separately-calibrated quarterly 95% CI -- see quarterly_aggregation.py
and conditional_residuals.compute_quarterly_residual_quantile_table).
"""
import joblib
import pandas as pd

from dengue_ml.run_dir import get_latest_run_dir
from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.forecasting.forecast_next_52w import generate_next_52w_forecast
from dengue_ml.forecasting.quarterly_aggregation import (
    aggregate_weekly_forecast_to_quarterly, aggregate_weekly_history_to_quarterly,
)
from dengue_ml.validation.conditional_residuals import compute_quarterly_residual_quantile_table, PROXY_COL
from dengue_ml.reporting.results_tables import final_forecast_table
from dengue_ml.reporting.plots import plot_final_forecast

if __name__ == "__main__":
    run_dir = get_latest_run_dir()
    print(f"Run directory: {run_dir}")

    model_path = run_dir / "final_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run train_final_model.py first."
        )

    artifact        = joblib.load(model_path)
    df              = prepare_model_table()

    classifier_path = run_dir / "final_classifier.pkl"
    classifier_artifact = joblib.load(classifier_path) if classifier_path.exists() else None
    if classifier_artifact is None:
        print(f"Warning: {classifier_path} not found — forecast CI will fall back "
              f"to NaN proxy_value/lower_95/upper_95 for the forecast horizon.")

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

    # Unfiltered history (includes the still-converging final week) so the
    # forecast plot can show its genuinely wide credible interval.
    full_history_df = prepare_model_table(apply_reliability_cutoff=False)
    quarterly_history_df = aggregate_weekly_history_to_quarterly(full_history_df)

    final_forecast_table(weekly_forecast_df, outputs_dir=run_dir)
    final_forecast_table(
        quarterly_forecast_df, outputs_dir=run_dir,
        period_col="forecast_quarter", filename="final_quarterly_forecast.csv",
    )
    plot_final_forecast(quarterly_forecast_df, quarterly_history_df, outputs_dir=run_dir)

    print("\n=== Quarterly forecast (deliverable) ===")
    print(quarterly_forecast_df.to_string(index=False))
    print(f"\nWeekly forecast saved to {run_dir / 'final_weekly_forecast.csv'}")
    print(f"Quarterly forecast saved to {run_dir / 'final_quarterly_forecast.csv'}")
