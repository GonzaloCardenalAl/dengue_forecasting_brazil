#!/usr/bin/env python
"""Train the final model on all data. Writes into the current run directory."""
import pandas as pd
from dengue_ml.run_dir import get_latest_run_dir
from dengue_ml.training.final_train import train_final_model, select_best_model
from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.reporting.plots import plot_feature_importance
from dengue_ml.reporting.results_tables import feature_importance_table

if __name__ == "__main__":
    run_dir = get_latest_run_dir()
    print(f"Run directory: {run_dir}")

    fold_metrics_path = run_dir / "fold_metrics.csv"
    fold_predictions_path = run_dir / "fold_predictions.csv"
    if not fold_metrics_path.exists():
        raise FileNotFoundError(
            f"{fold_metrics_path} not found — run run_nested_cv.py first."
        )

    fold_metrics = pd.read_csv(fold_metrics_path)
    fold_predictions = pd.read_csv(fold_predictions_path) if fold_predictions_path.exists() else None
    best_model, best_mae = select_best_model(fold_metrics)
    print(f"Best model: {best_model}  (mean MAE = {best_mae:.1f} cases)")

    df = prepare_model_table()
    artifact = train_final_model(
        selected_model_name=best_model, df=df, outputs_dir=run_dir,
        fold_predictions=fold_predictions,
    )

    if "model" in artifact and hasattr(artifact["model"], "feature_importances_"):
        feat_cols = artifact["feature_cols"]
        feature_importance_table(artifact["model"], feat_cols, outputs_dir=run_dir)
        plot_feature_importance(artifact["model"], feat_cols)
        print("Feature importance saved.")
