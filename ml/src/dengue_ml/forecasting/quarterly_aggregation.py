import numpy as np
import pandas as pd

from dengue_ml.config import CITY_COL
from dengue_ml.validation.conditional_residuals import apply_residual_quantile_table


def aggregate_monthly_forecast_to_quarterly(
    monthly_forecast_df: pd.DataFrame,
    quarterly_residual_quantiles: dict | None,
) -> pd.DataFrame:
    """
    Roll a 12-month-ahead forecast (output of `generate_next_12m_forecast`) up
    to the quarterly deliverable.

    Point forecast: sum of the 3 monthly point forecasts per (city, quarter)
    -- valid since case counts are additive.

    95% CI: NOT a sum of the monthly lower_95/upper_95 (statistically invalid
    -- see `compute_quarterly_residual_quantile_table`'s docstring). Instead,
    apply the quarterly-residual-quantile calibration table (computed once
    from nested-CV OOF data) to the summed point forecast, keyed off the
    growth_proxy value at the first month of each quarter.

    Returns DataFrame: city, forecast_quarter, predicted_cases, lower_95,
    upper_95, model_name.
    """
    df = monthly_forecast_df.copy().sort_values("forecast_month")
    df["forecast_quarter"] = pd.PeriodIndex(df["forecast_month"], freq="Q").to_timestamp()

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


def aggregate_monthly_history_to_quarterly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Roll up a monthly historical table (output of `prepare_model_table`) to
    quarterly sums of casos_est/casos_est_min/casos_est_max, for plotting the
    quarterly deliverable's forecast against quarterly-aggregated history
    (mirrors the sum-aggregation `aggregate_dengue_to_monthly` already does,
    one level up).
    """
    df = monthly_df.copy()
    df["quarter_start"] = pd.PeriodIndex(df["month_start"], freq="Q").to_timestamp()

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
