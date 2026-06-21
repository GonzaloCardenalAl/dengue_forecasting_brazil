#!/usr/bin/env python
"""
Run the binary "epidemic this week" classifier's nested CV, evaluate it
against the existing nivel_inc/sustained_rt regime-proxy rules on the same
weekly-grain OOF rows, and report the comparison. Structurally independent
of run_nested_cv.py / train_pipeline.py (the point-regression pipeline) --
needs that run's fold_predictions.csv (in the same run dir) for the coverage
comparison step, so run run_nested_cv.py first.
"""
from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.run_dir import get_latest_run_dir
from dengue_ml.validation.nested_cv_classifier import run_nested_cv_classifier
from dengue_ml.reporting.results_tables import proxy_comparison_table
from dengue_ml.reporting.plots import plot_proxy_comparison

import pandas as pd

if __name__ == "__main__":
    outputs_dir = get_latest_run_dir()
    fold_predictions_reg = pd.read_csv(outputs_dir / "fold_predictions.csv", parse_dates=["week_start"])

    print("Loading and preprocessing data ...")
    df = prepare_model_table()

    print("\nRunning classifier nested cross-validation ...")
    fold_metrics_clf, fold_predictions_clf, best_hparams_clf = run_nested_cv_classifier(df)

    fold_metrics_clf.to_csv(outputs_dir / "fold_metrics_clf.csv", index=False)
    fold_predictions_clf.to_csv(outputs_dir / "fold_predictions_clf.csv", index=False)
    best_hparams_clf.to_csv(outputs_dir / "best_hyperparameters_clf.csv", index=False)

    print("\n=== Proxy comparison (fair, weekly grain) ===")
    comparison = proxy_comparison_table(fold_predictions_clf, fold_predictions_reg, outputs_dir=outputs_dir)
    print(comparison.to_string(index=False))

    plot_proxy_comparison(comparison, outputs_dir=outputs_dir)
    print(f"\nOutputs saved to {outputs_dir}")
