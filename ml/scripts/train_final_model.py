#!/usr/bin/env python
"""Train the final model on all data. Writes into the current run directory."""
import pandas as pd
from dengue_ml.run_dir import get_latest_run_dir
from dengue_ml.training.final_train import (
    train_final_model, select_best_model_with_ar_stability, train_final_classifier, select_best_classifier,
)
from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.reporting.plots import (
    plot_feature_importance, plot_oof_predictions, plot_oof_predictions_monthly,
    plot_coverage_by_gap, plot_residual_distribution,
)
from dengue_ml.reporting.results_tables import feature_importance_table, coverage_by_gap_table

if __name__ == "__main__":
    run_dir = get_latest_run_dir()
    print(f"Run directory: {run_dir}")

    fold_metrics_path = run_dir / "fold_metrics.csv"
    fold_predictions_path = run_dir / "fold_predictions.csv"
    fold_predictions_ar_path = run_dir / "fold_predictions_ar.csv"
    fold_metrics_clf_path = run_dir / "fold_metrics_clf.csv"
    if not fold_metrics_path.exists():
        raise FileNotFoundError(
            f"{fold_metrics_path} not found — run run_nested_cv.py first."
        )

    fold_metrics = pd.read_csv(fold_metrics_path)
    fold_predictions = pd.read_csv(fold_predictions_path) if fold_predictions_path.exists() else None
    # Optional -- only present if run_autoregressive_cv.py has been run for
    # this run dir. None -> artifact["horizon_quantiles"] stays None and the
    # production forecast just skips the parallel horizon-aware deliverable.
    fold_predictions_ar = pd.read_csv(fold_predictions_ar_path) if fold_predictions_ar_path.exists() else None
    best_model, selection_info = select_best_model_with_ar_stability(fold_metrics, fold_predictions_ar)
    if "median_mae_ar" in selection_info:
        print(
            f"Best model: {best_model}  "
            f"(1-step median MAE = {selection_info['median_mae_1step']:.1f}, rank {selection_info['rank_1step']:.1f}; "
            f"AR median MAE = {selection_info['median_mae_ar']:.1f}, rank {selection_info['rank_ar']:.1f}; "
            f"combined rank = {selection_info['combined_rank']:.1f})"
        )
    else:
        print(f"Best model: {best_model}  (median MAE = {selection_info['median_mae_1step']:.1f} cases)")

    # run_nested_cv.py (Step 1) already plotted OOF/coverage/residual diagnostics,
    # but for select_best_model's 1-step-only pick -- which can differ from
    # best_model above once AR-rollout stability is folded in. Regenerate here
    # for the model that's actually about to be trained and shipped, so the
    # diagnostic figures on disk always match the production model.
    if fold_predictions is not None:
        plot_oof_predictions(fold_predictions, best_model, outputs_dir=run_dir, log_scale=False)
        plot_oof_predictions(fold_predictions, best_model, outputs_dir=run_dir, log_scale=True)
        plot_oof_predictions_monthly(fold_predictions, best_model, outputs_dir=run_dir, log_scale=False)
        plot_oof_predictions_monthly(fold_predictions, best_model, outputs_dir=run_dir, log_scale=True)
        coverage_gap = coverage_by_gap_table(fold_predictions, best_model, outputs_dir=run_dir)
        plot_coverage_by_gap(coverage_gap, outputs_dir=run_dir)
        plot_residual_distribution(fold_predictions, best_model, outputs_dir=run_dir)
        print(f"OOF/coverage/residual diagnostics regenerated for '{best_model}'.")

    df = prepare_model_table()
    artifact = train_final_model(
        selected_model_name=best_model, df=df, outputs_dir=run_dir,
        fold_predictions=fold_predictions, fold_predictions_ar=fold_predictions_ar,
    )

    if "model" in artifact and hasattr(artifact["model"], "feature_importances_"):
        feat_cols = artifact["feature_cols"]
        model_label = "xRFM" if best_model.startswith("xrfm") else "XGBoost"
        feature_importance_table(artifact["model"], feat_cols, outputs_dir=run_dir)
        plot_feature_importance(artifact["model"], feat_cols, model_label=model_label)
        print("Feature importance saved.")

    # Final epidemic classifier -- its predicted probability is the
    # CI-regime proxy applied at forecast time (forecasting/forecast_next_52w.py).
    if fold_metrics_clf_path.exists():
        fold_metrics_clf = pd.read_csv(fold_metrics_clf_path)
        best_classifier, best_auc = select_best_classifier(fold_metrics_clf)
        print(f"Best classifier: {best_classifier}  (mean AUC = {best_auc:.3f})")
        train_final_classifier(
            selected_model_name=best_classifier, df=df, outputs_dir=run_dir,
        )
    else:
        print(f"{fold_metrics_clf_path} not found — skipping final classifier "
              f"(run run_nested_cv.py with the classifier CV step first).")
