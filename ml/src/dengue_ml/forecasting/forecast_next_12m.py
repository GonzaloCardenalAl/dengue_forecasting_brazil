import numpy as np
import pandas as pd
import joblib

from dengue_ml.config import CITY_COL, TARGET, FORECAST_HORIZON, CITIES
from dengue_ml.features.feature_pipeline import build_features
from dengue_ml.models.xgboost_models import predict_xgb
from dengue_ml.models.xrfm_models import predict_xrfm
from dengue_ml.models.sarima import forecast_sarima
from dengue_ml.validation.conditional_residuals import apply_residual_quantile_table


def generate_next_12m_forecast(
    artifact: dict,
    latest_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Generate a `horizon`-month ahead point forecast + 95% CI.

    Parameters
    ----------
    artifact   : dict returned by train_final_model (or loaded via joblib)
    latest_df  : full historical monthly table (all available data)
    horizon    : number of months to forecast (default 12)

    Returns
    -------
    DataFrame with columns:
        city, forecast_month, predicted_cases, lower_95, upper_95,
        proxy_value, model_name
    """
    model_name = artifact["model_name"]

    if model_name == "baseline":
        return _forecast_baseline(artifact, latest_df, horizon)

    if model_name == "sarima":
        return _forecast_sarima(artifact, latest_df, horizon)

    if model_name.startswith("xgb"):
        return _forecast_xgb(artifact, latest_df, horizon)

    if model_name.startswith("xrfm"):
        return _forecast_xrfm(artifact, latest_df, horizon)

    raise ValueError(f"Unknown model_name '{model_name}' for forecasting.")


# ── Baseline ──────────────────────────────────────────────────────────────────

def _forecast_baseline(
    artifact: dict, latest_df: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    from dengue_ml.models.baseline import seasonal_naive_forecast

    last_m = latest_df["month_start"].max()
    future_ms = _next_months(last_m, horizon)
    rows = []
    for city in CITIES:
        for m in future_ms:
            # Build a fake test row
            fake = pd.DataFrame([{CITY_COL: city, "month_start": m}])
            result = seasonal_naive_forecast(latest_df, fake)
            pred = result["predicted"].iloc[0] if not result.empty else np.nan
            rows.append({
                "city": city, "forecast_month": m,
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
    last_m = latest_df["month_start"].max()
    future_ms = _next_months(last_m, horizon)
    rows = []
    for city, fit_result in city_models.items():
        preds_log, lower_log, upper_log = forecast_sarima(fit_result, horizon=horizon)
        preds = np.expm1(preds_log)
        lower = np.expm1(lower_log)
        upper = np.expm1(upper_log)
        for i, m in enumerate(future_ms):
            rows.append({
                "city": city, "forecast_month": m,
                "predicted_cases": float(preds[i]),
                "lower_95": float(lower[i]),
                "upper_95": float(upper[i]),
                "proxy_value": np.nan,
                "model_name": "sarima",
            })
    return pd.DataFrame(rows)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def _forecast_xgb(
    artifact: dict, latest_df: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """
    Autoregressive multi-step forecast: predict M+1, append to history,
    predict M+2, etc. Uses quantile models for 95% CI.
    """
    feature_set  = artifact["feature_set"]
    climate_stats = artifact.get("climate_stats")
    model        = artifact["model"]
    residual_quantiles = artifact.get("residual_quantiles")

    history = latest_df.copy()
    last_m  = history["month_start"].max()
    future_ms = _next_months(last_m, horizon)

    rows = []
    for m in future_ms:
        # Build feature row for this future month
        # We need the full df including all history for lag computation
        # Create a stub row for each city with NaN target, then build features
        stubs = pd.DataFrame([
            {
                CITY_COL: city,
                "month_start": m,
                TARGET: np.nan,
                "tempmed": _last_val(history, city, "tempmed"),
                "humidmed": _last_val(history, city, "humidmed"),
                "transmissao": _last_val(history, city, "transmissao"),
                "receptivo": _last_val(history, city, "receptivo"),
                "nivel_inc": _last_val(history, city, "nivel_inc"),
                "pop": _last_val(history, city, "pop"),
                "p_inc100k": _last_val(history, city, "p_inc100k"),
                "nino34_anom": _last_val(history, None, "nino34_anom"),
                "roni": _last_val(history, None, "roni"),
            }
            for city in CITIES
        ])

        extended = pd.concat([history, stubs], ignore_index=True).sort_values(
            [CITY_COL, "month_start"]
        )

        X_all, _, meta_all, _ = build_features(
            extended, feature_set, climate_fit_stats=climate_stats
        )
        mask = meta_all["month_start"] == m

        if mask.sum() == 0:
            continue

        X_m = X_all[mask]
        cities_m = meta_all[mask][CITY_COL].values

        preds = np.expm1(predict_xgb(model, X_m))
        if residual_quantiles is not None:
            proxy = X_m[residual_quantiles["proxy_col"]].values
            lower, upper = apply_residual_quantile_table(preds, proxy, residual_quantiles)
        else:
            proxy = np.full_like(preds, np.nan)
            lower = upper = np.full_like(preds, np.nan)

        for i, city in enumerate(cities_m):
            rows.append({
                "city": city, "forecast_month": m,
                "predicted_cases": float(preds[i]),
                "lower_95": float(lower[i]),
                "upper_95": float(upper[i]),
                "proxy_value": float(proxy[i]),
                "model_name": artifact["model_name"],
            })

            # Append prediction to history for next-step lag computation
            stub_row = stubs[stubs[CITY_COL] == city].copy()
            stub_row[TARGET] = preds[i]
            history = pd.concat([history, stub_row], ignore_index=True)

    return pd.DataFrame(rows)


# ── xRFM ──────────────────────────────────────────────────────────────────────

def _forecast_xrfm(
    artifact: dict, latest_df: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """
    Autoregressive multi-step forecast for xRFM — identical loop shape to
    _forecast_xgb; xRFM's val-set requirement only applies at fit time
    (already satisfied when the model in `artifact` was trained), so this
    is a pure inference loop just like XGBoost's.
    """
    feature_set  = artifact["feature_set"]
    climate_stats = artifact.get("climate_stats")
    model        = artifact["model"]
    residual_quantiles = artifact.get("residual_quantiles")

    history = latest_df.copy()
    last_m  = history["month_start"].max()
    future_ms = _next_months(last_m, horizon)

    rows = []
    for m in future_ms:
        stubs = pd.DataFrame([
            {
                CITY_COL: city,
                "month_start": m,
                TARGET: np.nan,
                "tempmed": _last_val(history, city, "tempmed"),
                "humidmed": _last_val(history, city, "humidmed"),
                "transmissao": _last_val(history, city, "transmissao"),
                "receptivo": _last_val(history, city, "receptivo"),
                "nivel_inc": _last_val(history, city, "nivel_inc"),
                "pop": _last_val(history, city, "pop"),
                "p_inc100k": _last_val(history, city, "p_inc100k"),
                "nino34_anom": _last_val(history, None, "nino34_anom"),
                "roni": _last_val(history, None, "roni"),
            }
            for city in CITIES
        ])

        extended = pd.concat([history, stubs], ignore_index=True).sort_values(
            [CITY_COL, "month_start"]
        )

        X_all, _, meta_all, _ = build_features(
            extended, feature_set, climate_fit_stats=climate_stats
        )
        mask = meta_all["month_start"] == m

        if mask.sum() == 0:
            continue

        X_m = X_all[mask]
        cities_m = meta_all[mask][CITY_COL].values

        preds = np.expm1(predict_xrfm(model, X_m))
        if residual_quantiles is not None:
            proxy = X_m[residual_quantiles["proxy_col"]].values
            lower, upper = apply_residual_quantile_table(preds, proxy, residual_quantiles)
        else:
            proxy = np.full_like(preds, np.nan)
            lower = upper = np.full_like(preds, np.nan)

        for i, city in enumerate(cities_m):
            rows.append({
                "city": city, "forecast_month": m,
                "predicted_cases": float(preds[i]),
                "lower_95": float(lower[i]),
                "upper_95": float(upper[i]),
                "proxy_value": float(proxy[i]),
                "model_name": artifact["model_name"],
            })

            stub_row = stubs[stubs[CITY_COL] == city].copy()
            stub_row[TARGET] = preds[i]
            history = pd.concat([history, stub_row], ignore_index=True)

    return pd.DataFrame(rows)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _next_months(last_m: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    months = []
    m = pd.Period(last_m, freq="M") + 1
    for _ in range(n):
        months.append(m.to_timestamp())
        m += 1
    return months


def _last_val(df: pd.DataFrame, city: str | None, col: str):
    if col not in df.columns:
        return np.nan
    if city is not None:
        sub = df[df[CITY_COL] == city]
    else:
        sub = df
    if sub.empty or col not in sub.columns:
        return np.nan
    val = sub.sort_values("month_start")[col].dropna()
    return val.iloc[-1] if not val.empty else np.nan
