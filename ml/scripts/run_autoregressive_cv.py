#!/usr/bin/env python
"""
Second validation procedure: rolls the actual autoregressive forecast loop
forward within each of run_nested_cv.py's outer fold test windows (instead
of evaluating 1-step-ahead predictions built from real historical lags), to
get residuals that reflect real horizon-compounding error. Run this after
run_nested_cv.py (needs its fold_metrics.csv/best_hyperparameters.csv/
selected_classifier.txt/best_hyperparameters_clf.csv from the same run dir).

Produces fold_predictions_ar.csv (the horizon-bucketed calibration's raw
input -- consumed later by train_final_model.py/generate_forecasts.py for
the parallel horizon-aware deliverable) and a backtest figure for one
example year (fold 8 = test year 2023) showing the calibrated band actually
widening from Q1 to Q4 of the forecast horizon, against known 2023 actuals.
"""
import pandas as pd

from dengue_ml.run_dir import get_latest_run_dir
from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.training.final_train import select_best_model
from dengue_ml.validation.autoregressive_cv import run_autoregressive_cv, AR_CV_MODEL_NAMES
from dengue_ml.validation.conditional_residuals import compute_horizon_bucketed_quarterly_residual_quantile_table
from dengue_ml.reporting.plots import plot_horizon_widening_example

EXAMPLE_FOLD = 8  # train <= 2022, test 2023 -- see config.py's OUTER_CUTOFFS

if __name__ == "__main__":
    run_dir = get_latest_run_dir()
    print(f"Run directory: {run_dir}")

    required = ["fold_metrics.csv", "best_hyperparameters.csv", "best_hyperparameters_clf.csv", "selected_classifier.txt"]
    missing = [f for f in required if not (run_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} in {run_dir} — run run_nested_cv.py first."
        )

    fold_metrics = pd.read_csv(run_dir / "fold_metrics.csv")
    best_hyperparams = pd.read_csv(run_dir / "best_hyperparameters.csv")
    classifier_hyperparams = pd.read_csv(run_dir / "best_hyperparameters_clf.csv")
    classifier_model_name = (run_dir / "selected_classifier.txt").read_text().strip()

    print("Loading and preprocessing data ...")
    df = prepare_model_table()

    print(f"\nRunning autoregressive CV (classifier: {classifier_model_name}) ...")
    fold_predictions_ar = run_autoregressive_cv(
        df, best_hyperparams, classifier_model_name, classifier_hyperparams,
    )
    fold_predictions_ar.to_csv(run_dir / "fold_predictions_ar.csv", index=False)
    print(f"Saved {run_dir / 'fold_predictions_ar.csv'} ({len(fold_predictions_ar)} rows)")

    best_model, best_mae = select_best_model(fold_metrics)
    print(f"Best model (by avg rank of median + std MAE): {best_model}")

    if best_model not in AR_CV_MODEL_NAMES:
        print(f"'{best_model}' has no autoregressive feedback loop (baseline/sarima) "
              f"-- skipping horizon-widening example figure.")
    else:
        horizon_quantiles = compute_horizon_bucketed_quarterly_residual_quantile_table(
            fold_predictions_ar, best_model
        )
        plot_horizon_widening_example(
            fold_predictions_ar, best_model, horizon_quantiles,
            fold_label=EXAMPLE_FOLD, year_label="2023", outputs_dir=run_dir,
        )
        print(f"Horizon-widening example figure saved for '{best_model}', fold {EXAMPLE_FOLD} (2023).")
