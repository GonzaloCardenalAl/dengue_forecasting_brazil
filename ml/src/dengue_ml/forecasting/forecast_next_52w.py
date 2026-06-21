import numpy as np
import pandas as pd

from dengue_ml.config import CITY_COL, FORECAST_HORIZON, CITIES
from dengue_ml.models.xgboost_models import predict_xgb
from dengue_ml.models.xrfm_models import predict_xrfm
from dengue_ml.models.sarima import forecast_sarima, fourier_terms
from dengue_ml.forecasting.autoregressive import next_weeks, forecast_with_rt_estimation


def generate_next_52w_forecast(
    artifact: dict,
    latest_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
    classifier_artifact: dict | None = None,
) -> pd.DataFrame:
    """
    Generate a `horizon`-week ahead point forecast + 95% CI.

    Parameters
    ----------
    artifact   : dict returned by train_final_model (or loaded via joblib)
    latest_df  : full historical weekly table (all available data)
    horizon    : number of weeks to forecast (default 52)
    classifier_artifact : dict returned by train_final_classifier (or loaded
        via joblib) -- the epidemic classifier whose predicted probability
        supplies the CI-regime proxy (growth_proxy) for forecast weeks. None
        -> proxy_value/CI fall back to NaN, same as when no
        residual_quantiles are available.

    Returns
    -------
    DataFrame with columns:
        city, forecast_week, predicted_cases, lower_95, upper_95,
        proxy_value, model_name
    """
    model_name = artifact["model_name"]

    if model_name == "baseline":
        return _forecast_baseline(artifact, latest_df, horizon)

    if model_name == "sarima":
        return _forecast_sarima(artifact, latest_df, horizon)

    if model_name.startswith("xgb"):
        return _forecast_xgb(artifact, latest_df, horizon, classifier_artifact)

    if model_name.startswith("xrfm"):
        return _forecast_xrfm(artifact, latest_df, horizon, classifier_artifact)

    raise ValueError(f"Unknown model_name '{model_name}' for forecasting.")


# ── Baseline ──────────────────────────────────────────────────────────────────

def _forecast_baseline(
    artifact: dict, latest_df: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    from dengue_ml.models.baseline import seasonal_naive_forecast

    last_w = latest_df["week_start"].max()
    future_ws = next_weeks(last_w, horizon)
    rows = []
    for city in CITIES:
        for w in future_ws:
            # Build a fake test row
            fake = pd.DataFrame([{CITY_COL: city, "week_start": w}])
            result = seasonal_naive_forecast(latest_df, fake)
            pred = result["predicted"].iloc[0] if not result.empty else np.nan
            rows.append({
                "city": city, "forecast_week": w,
                "predicted_cases": pred,
                "lower_95": np.nan, "upper_95": np.nan,
                "proxy_value": np.nan,
                "model_name": "baseline",
            })
    return pd.DataFrame(rows)


# ── SARIMA ────────────────────────────────────────────────────────────────────

def _forecast_sarima(
    artifact: dict, latest_df: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    city_models = artifact["models"]
    last_w = latest_df["week_start"].max()
    future_ws = next_weeks(last_w, horizon)
    rows = []
    for city, info in city_models.items():
        fit_result = info["fit"]
        exog_future = fourier_terms(pd.DatetimeIndex(future_ws), info["fourier_order"])
        preds_log, lower_log, upper_log = forecast_sarima(fit_result, horizon=horizon, exog=exog_future)
        preds = np.expm1(preds_log)
        lower = np.expm1(lower_log)
        upper = np.expm1(upper_log)
        for i, w in enumerate(future_ws):
            rows.append({
                "city": city, "forecast_week": w,
                "predicted_cases": float(preds[i]),
                "lower_95": float(lower[i]),
                "upper_95": float(upper[i]),
                "proxy_value": np.nan,
                "model_name": "sarima",
            })
    return pd.DataFrame(rows)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def _forecast_xgb(
    artifact: dict, latest_df: pd.DataFrame, horizon: int,
    classifier_artifact: dict | None = None,
) -> pd.DataFrame:
    """Autoregressive multi-step forecast for XGBoost. See forecast_with_rt_estimation."""
    rows, _ = forecast_with_rt_estimation(artifact, latest_df, horizon, predict_xgb, classifier_artifact)
    return rows


# ── xRFM ──────────────────────────────────────────────────────────────────────

def _forecast_xrfm(
    artifact: dict, latest_df: pd.DataFrame, horizon: int,
    classifier_artifact: dict | None = None,
) -> pd.DataFrame:
    """
    Autoregressive multi-step forecast for xRFM -- identical to _forecast_xgb;
    xRFM's val-set requirement only applies at fit time (already satisfied
    when the model in `artifact` was trained), so this is a pure inference
    loop just like XGBoost's. See forecast_with_rt_estimation.
    """
    rows, _ = forecast_with_rt_estimation(artifact, latest_df, horizon, predict_xrfm, classifier_artifact)
    return rows
