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
from dengue_ml.forecasting.pipeline import run_inference
from dengue_ml.forecasting.quarterly_aggregation import (
    aggregate_weekly_history_to_quarterly,
    aggregate_weekly_oof_predictions_to_quarterly,
)
from dengue_ml.reporting.plots import plot_final_forecast, plot_forecast_vs_previous_year

if __name__ == "__main__":
    run_dir = get_latest_run_dir()
    print(f"Run directory: {run_dir}")

    classifier_path = run_dir / "final_classifier.pkl"
    if not classifier_path.exists():
        print(f"Warning: {classifier_path} not found — forecast CI will fall back "
              f"to NaN proxy_value/lower_95/upper_95 for the forecast horizon.")

    weekly_forecast_df, quarterly_forecast_df = run_inference(run_dir)
    artifact = joblib.load(run_dir / "final_model.pkl")

    oof_quarterly_df = None
    fold_predictions_path = run_dir / "fold_predictions.csv"
    if fold_predictions_path.exists():
        fold_predictions = pd.read_csv(fold_predictions_path, parse_dates=["week_start"])
        model_oof = fold_predictions[fold_predictions["model"] == artifact["model_name"]]
        if not model_oof.empty:
            oof_quarterly_df = aggregate_weekly_oof_predictions_to_quarterly(
                fold_predictions, artifact["model_name"]
            )

    # Unfiltered history (includes the still-converging final week) so the
    # forecast plot can show its genuinely wide credible interval.
    full_history_df = prepare_model_table(apply_reliability_cutoff=False)
    quarterly_history_df = aggregate_weekly_history_to_quarterly(full_history_df)

    plot_final_forecast(
        quarterly_forecast_df, quarterly_history_df,
        oof_quarterly_df=oof_quarterly_df, outputs_dir=run_dir,
    )
    plot_forecast_vs_previous_year(quarterly_forecast_df, quarterly_history_df, outputs_dir=run_dir)

    print("\n=== Quarterly forecast (deliverable) ===")
    print(quarterly_forecast_df.to_string(index=False))
    print(f"\nWeekly forecast saved to {run_dir / 'final_weekly_forecast.csv'}")
    print(f"Quarterly forecast saved to {run_dir / 'final_quarterly_forecast.csv'}")

    # Parallel horizon-aware deliverable -- run_inference() already computed
    # and wrote final_quarterly_forecast_horizon_aware.csv above when
    # artifact["horizon_quantiles"] is populated (i.e. run_autoregressive_cv.py
    # has been run for this run dir); read it back here for plotting only, so
    # the residual-quantile/aggregation logic isn't duplicated.
    horizon_quantiles = artifact.get("horizon_quantiles")
    horizon_aware_path = run_dir / "final_quarterly_forecast_horizon_aware.csv"
    if horizon_quantiles is not None and horizon_aware_path.exists():
        quarterly_forecast_df_ha = pd.read_csv(horizon_aware_path, parse_dates=["forecast_quarter"])
        plot_final_forecast(
            quarterly_forecast_df_ha, quarterly_history_df,
            oof_quarterly_df=oof_quarterly_df, outputs_dir=run_dir,
            filename="final_forecast_horizon_aware.png",
        )
        plot_forecast_vs_previous_year(
            quarterly_forecast_df_ha, quarterly_history_df, outputs_dir=run_dir,
            filename="forecast_vs_previous_year_horizon_aware.png",
        )
        print(f"\nHorizon-aware quarterly forecast (Q1 narrower, Q4 wider CI) saved to "
              f"{horizon_aware_path}")
    else:
        print(f"\nNo horizon_quantiles in final_model.pkl -- skipping the parallel "
              f"horizon-aware deliverable (run run_autoregressive_cv.py then retrain to enable it).")

    # Year-over-year deliverable at week/month/quarter grain: a single
    # period-of-year axis (Q1-Q4 / Jan-Dec / W1-W52) overlaying last year's
    # actuals + last year's OOF model estimate (with its own CI) against this
    # year's forecast (with its own horizon-aware CI), plus a GIF revealing
    # the forecast one point at a time -- see reporting/plots.py's
    # plot_forecast_year_over_year. Needs fold_predictions.csv (OOF) and
    # fold_predictions_ar.csv (autoregressive-rollout CI calibration) in this
    # run dir; non-fatal, skips gracefully if either is missing.
    fold_predictions_ar_path = run_dir / "fold_predictions_ar.csv"
    if horizon_quantiles is None or not fold_predictions_ar_path.exists() or not fold_predictions_path.exists():
        print("\nSkipping year-over-year forecast deliverable (needs horizon_quantiles + "
              "fold_predictions.csv + fold_predictions_ar.csv).")
    else:
        from dengue_ml.config import CITY_COL, FORECAST_HORIZON
        from dengue_ml.validation.conditional_residuals import (
            assign_loFo_conditional_ci, aggregate_oof_to_monthly, aggregate_oof_to_quarterly,
            apply_horizon_bucketed_quantile_table,
            compute_horizon_bucketed_monthly_residual_quantile_table,
            compute_horizon_bucketed_weekly_residual_quantile_table,
        )
        from dengue_ml.forecasting.quarterly_aggregation import (
            aggregate_weekly_forecast_to_monthly, aggregate_weekly_history_to_monthly,
        )
        from dengue_ml.reporting.plots import plot_forecast_year_over_year, plot_forecast_year_over_year_frames

        model_name = artifact["model_name"]
        fold_predictions = pd.read_csv(fold_predictions_path, parse_dates=["week_start"])
        fold_predictions_ar = pd.read_csv(fold_predictions_ar_path, parse_dates=["week_start"])

        forecast_year = pd.Timestamp(weekly_forecast_df["forecast_week"].min()).year
        prev_year = forecast_year - 1

        # ── Quarterly ──
        quarterly_oof = assign_loFo_conditional_ci(
            aggregate_oof_to_quarterly(fold_predictions, model_name), model_name
        )
        quarterly_oof_prev = quarterly_oof[quarterly_oof["quarter_start"].dt.year == prev_year].copy()
        quarterly_oof_prev["x_pos"] = quarterly_oof_prev["quarter_start"].dt.quarter
        quarterly_oof_prev = quarterly_oof_prev.rename(columns={"predicted": "value"})

        quarterly_actual_prev = quarterly_history_df[
            quarterly_history_df["quarter_start"].dt.year == prev_year
        ].copy()
        quarterly_actual_prev["x_pos"] = quarterly_actual_prev["quarter_start"].dt.quarter
        quarterly_actual_prev = quarterly_actual_prev.rename(columns={"casos_est": "value"})

        quarterly_fc = quarterly_forecast_df_ha.rename(
            columns={"city": CITY_COL, "predicted_cases": "value"}
        ).copy()
        quarterly_fc["x_pos"] = quarterly_fc.groupby(CITY_COL)["forecast_quarter"].rank(method="first").astype(int)

        quarterly_dir = run_dir / "figures" / "forecast" / "quarterly"
        quarter_labels = ["Q1", "Q2", "Q3", "Q4"]
        plot_forecast_year_over_year(
            quarter_labels, quarterly_actual_prev, quarterly_oof_prev, quarterly_fc,
            prev_year, forecast_year, outputs_dir=quarterly_dir,
        )
        plot_forecast_year_over_year_frames(
            quarter_labels, quarterly_actual_prev, quarterly_oof_prev, quarterly_fc,
            prev_year, forecast_year, outputs_dir=quarterly_dir,
        )

        # ── Monthly ──
        monthly_horizon_quantiles = compute_horizon_bucketed_monthly_residual_quantile_table(
            fold_predictions_ar, model_name
        )
        monthly_oof = assign_loFo_conditional_ci(
            aggregate_oof_to_monthly(fold_predictions, model_name), model_name
        )
        monthly_oof_prev = monthly_oof[monthly_oof["month_start"].dt.year == prev_year].copy()
        monthly_oof_prev["x_pos"] = monthly_oof_prev["month_start"].dt.month
        monthly_oof_prev = monthly_oof_prev.rename(columns={"predicted": "value"})

        monthly_history_df = aggregate_weekly_history_to_monthly(full_history_df)
        monthly_actual_prev = monthly_history_df[monthly_history_df["month_start"].dt.year == prev_year].copy()
        monthly_actual_prev["x_pos"] = monthly_actual_prev["month_start"].dt.month
        monthly_actual_prev = monthly_actual_prev.rename(columns={"casos_est": "value"})

        monthly_fc = aggregate_weekly_forecast_to_monthly(weekly_forecast_df, monthly_horizon_quantiles)
        monthly_fc = monthly_fc.rename(columns={"city": CITY_COL, "predicted_cases": "value"})
        monthly_fc["x_pos"] = monthly_fc.groupby(CITY_COL)["forecast_month"].rank(method="first").astype(int)

        monthly_dir = run_dir / "figures" / "forecast" / "monthly"
        month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        plot_forecast_year_over_year(
            month_labels, monthly_actual_prev, monthly_oof_prev, monthly_fc,
            prev_year, forecast_year, outputs_dir=monthly_dir,
        )
        plot_forecast_year_over_year_frames(
            month_labels, monthly_actual_prev, monthly_oof_prev, monthly_fc,
            prev_year, forecast_year, outputs_dir=monthly_dir,
        )

        # ── Weekly ──
        weekly_horizon_quantiles = compute_horizon_bucketed_weekly_residual_quantile_table(
            fold_predictions_ar, model_name
        )
        weekly_oof = assign_loFo_conditional_ci(fold_predictions, model_name)
        weekly_oof_prev = weekly_oof[
            (weekly_oof["model"] == model_name) & (weekly_oof["week_start"].dt.year == prev_year)
        ].copy()
        weekly_oof_prev["x_pos"] = weekly_oof_prev["week_start"].dt.isocalendar().week.astype(int)
        weekly_oof_prev = weekly_oof_prev.rename(columns={"predicted": "value"})

        weekly_actual_prev = full_history_df[full_history_df["week_start"].dt.year == prev_year].copy()
        weekly_actual_prev["x_pos"] = weekly_actual_prev["week_start"].dt.isocalendar().week.astype(int)
        weekly_actual_prev = weekly_actual_prev.rename(columns={"casos_est": "value"})

        weekly_fc = weekly_forecast_df.rename(columns={"city": CITY_COL, "predicted_cases": "value"}).copy()
        weekly_fc["x_pos"] = weekly_fc.groupby(CITY_COL)["forecast_week"].rank(method="first").astype(int)
        fc_lower, fc_upper = apply_horizon_bucketed_quantile_table(
            weekly_fc["value"].values, weekly_fc["proxy_value"].values, weekly_fc["x_pos"].values,
            weekly_horizon_quantiles,
        )
        weekly_fc["lower_95"] = fc_lower
        weekly_fc["upper_95"] = fc_upper

        weekly_dir = run_dir / "figures" / "forecast" / "weekly"
        week_labels = [f"W{i}" for i in range(1, FORECAST_HORIZON + 1)]
        plot_forecast_year_over_year(
            week_labels, weekly_actual_prev, weekly_oof_prev, weekly_fc,
            prev_year, forecast_year, outputs_dir=weekly_dir,
        )
        plot_forecast_year_over_year_frames(
            week_labels, weekly_actual_prev, weekly_oof_prev, weekly_fc,
            prev_year, forecast_year, outputs_dir=weekly_dir,
        )

        print(f"\nYear-over-year forecast deliverable (week/month/quarter + GIF frames) saved under "
              f"{run_dir / 'figures' / 'forecast'}/")
