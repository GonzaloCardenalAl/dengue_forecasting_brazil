import numpy as np
import pandas as pd

from dengue_ml.config import CITY_COL
from dengue_ml.validation.conditional_residuals import apply_residual_quantile_table


def aggregate_weekly_forecast_to_quarterly(
    weekly_forecast_df: pd.DataFrame,
    quarterly_residual_quantiles: dict | None,
) -> pd.DataFrame:
    """
    Roll a 52-week-ahead forecast (output of `generate_next_52w_forecast`) up
    to the quarterly deliverable.

    Point forecast: sum of the ~13 weekly point forecasts per (city, quarter)
    -- valid since case counts are additive.

    95% CI: NOT a sum of the weekly lower_95/upper_95 (statistically invalid
    -- see `compute_quarterly_residual_quantile_table`'s docstring). Instead,
    apply the quarterly-residual-quantile calibration table (computed once
    from nested-CV OOF data) to the summed point forecast, keyed off the
    growth_proxy value at the first week of each quarter.

    Returns DataFrame: city, forecast_quarter, predicted_cases, lower_95,
    upper_95, model_name.
    """
    df = weekly_forecast_df.copy().sort_values("forecast_week")
    df["forecast_quarter"] = pd.PeriodIndex(df["forecast_week"], freq="Q").to_timestamp()

    grouped = df.groupby(["city", "forecast_quarter"]).agg(
        predicted_cases=("predicted_cases", "sum"),
        proxy_value=("proxy_value", "first"),
        model_name=("model_name", "first"),
    ).reset_index()

    if quarterly_residual_quantiles is not None and grouped["proxy_value"].notna().any():
        lower, upper = apply_residual_quantile_table(
            grouped["predicted_cases"].values,
            grouped["proxy_value"].values,
            quarterly_residual_quantiles,
        )
    else:
        lower = upper = np.full(len(grouped), np.nan)

    grouped["lower_95"] = lower
    grouped["upper_95"] = upper

    return grouped[["city", "forecast_quarter", "predicted_cases", "lower_95", "upper_95", "model_name"]]


def aggregate_weekly_oof_predictions_to_quarterly(
    fold_predictions: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """
    Roll up one model's out-of-fold weekly CV predictions (output of
    `run_nested_cv`) to quarterly sums, so the final-forecast plot can show
    the model's own historical track record (not just the raw actuals)
    leading into the forecast -- a single continuous "model prediction"
    line, past (backtested) and future (forecast).

    Returns DataFrame: city_name, quarter_start, predicted_cases.
    """
    df = fold_predictions[fold_predictions["model"] == model_name].copy()
    df["quarter_start"] = pd.PeriodIndex(df["week_start"], freq="Q").to_timestamp()

    agg = (
        df.groupby([CITY_COL, "quarter_start"], sort=True)["predicted"]
        .sum()
        .reset_index()
        .rename(columns={"predicted": "predicted_cases"})
    )
    return agg.sort_values([CITY_COL, "quarter_start"]).reset_index(drop=True)


def aggregate_weekly_history_to_quarterly(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up a weekly historical table (output of `prepare_model_table`) to
    quarterly sums of casos_est/casos_est_min/casos_est_max, for plotting the
    quarterly deliverable's forecast against quarterly-aggregated history.
    """
    df = weekly_df.copy()
    df["quarter_start"] = pd.PeriodIndex(df["week_start"], freq="Q").to_timestamp()

    agg = (
        df.groupby([CITY_COL, "quarter_start"], sort=True)
        .agg(
            casos_est=("casos_est", "sum"),
            casos_est_min=("casos_est_min", "sum"),
            casos_est_max=("casos_est_max", "sum"),
        )
        .reset_index()
    )
    return agg.sort_values([CITY_COL, "quarter_start"]).reset_index(drop=True)
