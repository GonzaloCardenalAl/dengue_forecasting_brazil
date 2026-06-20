import pandas as pd
from pathlib import Path

from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.validation.nested_cv import run_nested_cv
from dengue_ml.validation.nested_cv_classifier import run_nested_cv_classifier
from dengue_ml.validation.conditional_residuals import attach_classifier_proxy
from dengue_ml.reporting.results_tables import (
    model_comparison_table, metrics_by_fold, baseline_improvement_table,
)
from dengue_ml.reporting.plots import plot_oof_predictions, plot_model_comparison
from dengue_ml.training.final_train import select_best_model, select_best_classifier


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
    print(f"  Weekly table: {df.shape[0]} rows × {df.shape[1]} cols "
          f"| cities: {df['city_name'].nunique()} "
          f"| weeks: {df['week_start'].nunique()}")

    print("\nRunning nested cross-validation ...")
    fold_metrics, fold_predictions, best_hparams = run_nested_cv(df)

    print("\nRunning classifier nested cross-validation (CI-regime proxy) ...")
    fold_metrics_clf, fold_predictions_clf = run_nested_cv_classifier(df)
    fold_metrics_clf.to_csv(outputs_dir / "fold_metrics_clf.csv", index=False)
    fold_predictions_clf.to_csv(outputs_dir / "fold_predictions_clf.csv", index=False)

    best_classifier, best_auc = select_best_classifier(fold_metrics_clf)
    print(f"Best classifier (by mean AUC): {best_classifier} (AUC = {best_auc:.3f})")
    # growth_proxy = this classifier's OOF predicted_proba, joined onto the
    # regression fold_predictions -- see conditional_residuals.py. Must run
    # before saving fold_predictions.csv so train_final_model.py's residual
    # quantile calibration (which reads growth_proxy from that CSV) sees it.
    fold_predictions = attach_classifier_proxy(fold_predictions, fold_predictions_clf, best_classifier)
    with open(outputs_dir / "selected_classifier.txt", "w") as f:
        f.write(best_classifier)

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
    print(f"\nBest model (by avg rank of median + std MAE): {best_model}")
    plot_oof_predictions(fold_predictions, best_model, outputs_dir=outputs_dir, log_scale=False)
    plot_oof_predictions(fold_predictions, best_model, outputs_dir=outputs_dir, log_scale=True)
    print(f"OOF predictions plots saved for '{best_model}' (log + linear)")

    print(f"\nOutputs saved to {outputs_dir}")

    return fold_metrics, fold_predictions, best_hparams
