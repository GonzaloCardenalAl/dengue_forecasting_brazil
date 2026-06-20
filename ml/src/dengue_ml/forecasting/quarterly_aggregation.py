import numpy as np
import pandas as pd

from dengue_ml.config import CITY_COL, TARGET
from dengue_ml.validation.conditional_residuals import (
    apply_residual_quantile_table, apply_horizon_bucketed_quantile_table,
)


def _expected_weeks_in_quarter(quarter_start: pd.Timestamp) -> int:
    """Number of W-SUN week-starts that fall within the calendar quarter
    beginning at quarter_start -- computed from the calendar, not from
    whatever happens to be in a given forecast, so it's correct regardless
    of which specific quarter/year is being checked (13 most of the time,
    occasionally 12 or 14 depending on how Sundays line up that year)."""
    next_quarter_start = quarter_start + pd.DateOffset(months=3)
    return len(pd.date_range(quarter_start, next_quarter_start - pd.Timedelta(days=1), freq="W-SUN"))


def _expected_weeks_in_month(month_start: pd.Timestamp) -> int:
    """Same idea as `_expected_weeks_in_quarter`, for a calendar month (4 or
    5 W-SUN week-starts depending on alignment)."""
    next_month_start = month_start + pd.DateOffset(months=1)
    return len(pd.date_range(month_start, next_month_start - pd.Timedelta(days=1), freq="W-SUN"))


def aggregate_weekly_forecast_to_quarterly(
    weekly_forecast_df: pd.DataFrame,
    quarterly_residual_quantiles: dict | None,
    horizon_bucketed_quantiles: dict | None = None,
) -> pd.DataFrame:
    """
    Roll a 52-week-ahead forecast (output of `generate_next_52w_forecast`) up
    to the quarterly deliverable.

    Point forecast: sum of the ~13 weekly point forecasts per (city, quarter)
    -- valid since case counts are additive.

    Partial leading/trailing quarters are dropped: if the forecast horizon
    doesn't start exactly on a quarter boundary (e.g. MAX_RELIABLE_WEEK now
    rolls forward continuously rather than sitting at a fixed date -- see
    config.compute_max_reliable_week), the first and/or last quarter bucket
    can contain only a handful of forecasted weeks. Summing just those few
    weeks and labeling it a full quarter's forecast would understate that
    quarter (most of it is already-known actual data, not predicted) and
    would also throw off `quarter_position` below, which assumes position 1
    is genuinely the first FULL forecast quarter (that's what the
    horizon-bucketed calibration table was built against).

    95% CI: NOT a sum of the weekly lower_95/upper_95 (statistically invalid
    -- see `compute_quarterly_residual_quantile_table`'s docstring). Instead,
    apply a calibration table to the summed point forecast, keyed off the
    growth_proxy value at the first week of each quarter:

    - `horizon_bucketed_quantiles` (from
      `compute_horizon_bucketed_quarterly_residual_quantile_table`), if
      given, takes priority -- it's keyed by quarter_position (1st/2nd/3rd/
      4th FULL quarter *of this forecast*, derived here by ranking
      `forecast_quarter` per city after dropping partial quarters), giving a
      band that widens with horizon.
    - Otherwise `quarterly_residual_quantiles` (the flat, non-horizon-aware
      table) is used, exactly as before.

    Returns DataFrame: city, forecast_quarter, predicted_cases, lower_95,
    upper_95, proxy_value, model_name.
    """
    df = weekly_forecast_df.copy().sort_values("forecast_week")
    df["forecast_quarter"] = pd.PeriodIndex(df["forecast_week"], freq="Q").to_timestamp()

    grouped = df.groupby(["city", "forecast_quarter"]).agg(
        predicted_cases=("predicted_cases", "sum"),
        proxy_value=("proxy_value", "first"),
        model_name=("model_name", "first"),
        n_weeks=("forecast_week", "size"),
    ).reset_index()

    is_full_quarter = grouped["n_weeks"] == grouped["forecast_quarter"].map(_expected_weeks_in_quarter)
    grouped = grouped[is_full_quarter].drop(columns="n_weeks").reset_index(drop=True)

    if horizon_bucketed_quantiles is not None and grouped["proxy_value"].notna().any():
        quarter_position = grouped.groupby("city")["forecast_quarter"].rank(method="first").astype(int)
        lower, upper = apply_horizon_bucketed_quantile_table(
            grouped["predicted_cases"].values,
            grouped["proxy_value"].values,
            quarter_position.values,
            horizon_bucketed_quantiles,
        )
    elif quarterly_residual_quantiles is not None and grouped["proxy_value"].notna().any():
        lower, upper = apply_residual_quantile_table(
            grouped["predicted_cases"].values,
            grouped["proxy_value"].values,
            quarterly_residual_quantiles,
        )
    else:
        lower = upper = np.full(len(grouped), np.nan)

    grouped["lower_95"] = lower
    grouped["upper_95"] = upper

    return grouped[["city", "forecast_quarter", "predicted_cases", "lower_95", "upper_95", "proxy_value", "model_name"]]


def aggregate_weekly_forecast_to_monthly(
    weekly_forecast_df: pd.DataFrame,
    horizon_bucketed_quantiles: dict | None = None,
) -> pd.DataFrame:
    """
    Monthly twin of `aggregate_weekly_forecast_to_quarterly`, for the
    weekly/monthly forecast deliverable -- same horizon-bucketed CI
    approach, just keyed off `month_position` (1st..12th month of this
    forecast, ranked per city) instead of `quarter_position`. No flat
    (non-horizon-aware) fallback table here -- the monthly deliverable only
    exists once the autoregressive CV horizon-bucketed table is available.

    Partial leading/trailing months are dropped, same reasoning as
    `aggregate_weekly_forecast_to_quarterly`.

    Returns DataFrame: city, forecast_month, predicted_cases, lower_95,
    upper_95, proxy_value, model_name.
    """
    df = weekly_forecast_df.copy().sort_values("forecast_week")
    df["forecast_month"] = df["forecast_week"].values.astype("datetime64[M]")

    grouped = df.groupby(["city", "forecast_month"]).agg(
        predicted_cases=("predicted_cases", "sum"),
        proxy_value=("proxy_value", "first"),
        model_name=("model_name", "first"),
        n_weeks=("forecast_week", "size"),
    ).reset_index()

    is_full_month = grouped["n_weeks"] == grouped["forecast_month"].map(_expected_weeks_in_month)
    grouped = grouped[is_full_month].drop(columns="n_weeks").reset_index(drop=True)

    if horizon_bucketed_quantiles is not None and grouped["proxy_value"].notna().any():
        month_position = grouped.groupby("city")["forecast_month"].rank(method="first").astype(int)
        lower, upper = apply_horizon_bucketed_quantile_table(
            grouped["predicted_cases"].values,
            grouped["proxy_value"].values,
            month_position.values,
            horizon_bucketed_quantiles,
        )
    else:
        lower = upper = np.full(len(grouped), np.nan)

    grouped["lower_95"] = lower
    grouped["upper_95"] = upper

    return grouped[["city", "forecast_month", "predicted_cases", "lower_95", "upper_95", "proxy_value", "model_name"]]


def aggregate_weekly_history_to_monthly(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly twin of `aggregate_weekly_history_to_quarterly`, sums of casos_est
    only (the redesigned year-over-year plot doesn't show the historical
    casos_est_min/max reporting-uncertainty band, unlike plot_final_forecast).
    """
    df = weekly_df.copy()
    df["month_start"] = df["week_start"].values.astype("datetime64[M]")

    agg = (
        df.groupby([CITY_COL, "month_start"], sort=True)[TARGET]
        .sum()
        .reset_index()
    )
    return agg.sort_values([CITY_COL, "month_start"]).reset_index(drop=True)


def aggregate_weekly_classifier_to_quarterly(
    fold_predictions_clf: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """
    Roll up one classifier's out-of-fold weekly CV predictions (output of
    `run_nested_cv_classifier`) to quarterly epidemic status, for the
    dashboard's historical "was this an epidemic quarter" view.

    A quarter's `predicted_proba` is the mean of its weekly probabilities;
    `is_epidemic` (the true label) is True if any week within the quarter
    was flagged -- an epidemic that breaks out partway through a quarter
    still makes it an epidemic quarter.

    Returns DataFrame: city_name, quarter_start, predicted_proba, is_epidemic.
    """
    df = fold_predictions_clf[fold_predictions_clf["model"] == model_name].copy()
    df["quarter_start"] = pd.PeriodIndex(df["week_start"], freq="Q").to_timestamp()

    agg = (
        df.groupby([CITY_COL, "quarter_start"], sort=True)
        .agg(
            predicted_proba=("predicted_proba", "mean"),
            is_epidemic=("is_epidemic", "max"),
        )
        .reset_index()
    )
    agg["is_epidemic"] = agg["is_epidemic"].astype(bool)
    return agg.sort_values([CITY_COL, "quarter_start"]).reset_index(drop=True)


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
    quarterly sums of casos_est/casos_est_min/casos_est_max/p_inc100k, for
    plotting the quarterly deliverable's forecast against
    quarterly-aggregated history (p_inc100k -- incidence per 100k -- lets
    the dashboard compare cities of very different population sizes).
    """
    df = weekly_df.copy()
    df["quarter_start"] = pd.PeriodIndex(df["week_start"], freq="Q").to_timestamp()

    agg = (
        df.groupby([CITY_COL, "quarter_start"], sort=True)
        .agg(
            casos_est=("casos_est", "sum"),
            casos_est_min=("casos_est_min", "sum"),
            casos_est_max=("casos_est_max", "sum"),
            p_inc100k=("p_inc100k", "sum"),
        )
        .reset_index()
    )
    return agg.sort_values([CITY_COL, "quarter_start"]).reset_index(drop=True)
