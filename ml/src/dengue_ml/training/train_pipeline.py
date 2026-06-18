import pandas as pd
from pathlib import Path

from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.validation.nested_cv import run_nested_cv
from dengue_ml.reporting.results_tables import (
    model_comparison_table, metrics_by_fold, baseline_improvement_table,
)
from dengue_ml.reporting.plots import plot_oof_predictions, plot_model_comparison
from dengue_ml.training.final_train import select_best_model


def run_training_pipeline(
    outputs_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end training pipeline:
    1. Load and preprocess data
    2. Run nested rolling CV
    3. Save outputs

    Returns (fold_metrics, fold_predictions, best_hyperparams)
    """
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing data ...")
    df = prepare_model_table()
    print(f"  Quarterly table: {df.shape[0]} rows × {df.shape[1]} cols "
          f"| cities: {df['city_name'].nunique()} "
          f"| quarters: {df['quarter_start'].nunique()}")

    print("\nRunning nested cross-validation ...")
    fold_metrics, fold_predictions, best_hparams = run_nested_cv(df)

    # Save outputs
    fold_metrics.to_csv(outputs_dir / "fold_metrics.csv", index=False)
    fold_predictions.to_csv(outputs_dir / "fold_predictions.csv", index=False)
    best_hparams.to_csv(outputs_dir / "best_hyperparameters.csv", index=False)

    # Print summary (median-sorted: mean/std alone are dominated by outlier
    # folds such as the 2024 outbreak, ~7x any prior year)
    print("\n=== Model Comparison (sorted by median fold MAE) ===")
    summary = model_comparison_table(fold_metrics, outputs_dir=outputs_dir)
    print(summary[["mae_mean", "mae_std", "mae_median", "mae_q25", "mae_q75"]].to_string())

    metrics_by_fold(fold_metrics, outputs_dir=outputs_dir)
    plot_model_comparison(fold_metrics, outputs_dir=outputs_dir, log_scale=False)
    plot_model_comparison(fold_metrics, outputs_dir=outputs_dir, log_scale=True)

    improvement = baseline_improvement_table(fold_metrics, outputs_dir=outputs_dir)
    print("\n=== % improvement in median MAE vs baseline ===")
    print(improvement.to_string(index=False))

    best_model, best_mae = select_best_model(fold_metrics)
    print(f"\nBest model (by median MAE): {best_model}")
    plot_oof_predictions(fold_predictions, best_model, outputs_dir=outputs_dir, log_scale=False)
    plot_oof_predictions(fold_predictions, best_model, outputs_dir=outputs_dir, log_scale=True)
    print(f"OOF predictions plots saved for '{best_model}' (log + linear)")

    print(f"\nOutputs saved to {outputs_dir}")

    return fold_metrics, fold_predictions, best_hparams
