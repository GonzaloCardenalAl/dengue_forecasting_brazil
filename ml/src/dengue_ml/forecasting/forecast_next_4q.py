import numpy as np
import pandas as pd
import joblib

from dengue_ml.config import CITY_COL, TARGET, FORECAST_HORIZON, CITIES
from dengue_ml.features.feature_pipeline import build_features
from dengue_ml.models.xgboost_models import predict_xgb
from dengue_ml.models.sarima import forecast_sarima
from dengue_ml.validation.conditional_residuals import apply_residual_quantile_table


def generate_next_4q_forecast(
    artifact: dict,
    latest_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Generate a `horizon`-quarter ahead point forecast + 95% CI.

    Parameters
    ----------
    artifact   : dict returned by train_final_model (or loaded via joblib)
    latest_df  : full historical quarterly table (all available data)
    horizon    : number of quarters to forecast (default 4)

    Returns
    -------
    DataFrame with columns:
        city, forecast_quarter, predicted_cases, lower_95, upper_95, model_name
    """
    model_name = artifact["model_name"]

    if model_name == "baseline":
        return _forecast_baseline(artifact, latest_df, horizon)

    if model_name == "sarima":
        return _forecast_sarima(artifact, latest_df, horizon)

    return _forecast_xgb(artifact, latest_df, horizon)


# ── Baseline ──────────────────────────────────────────────────────────────────

def _forecast_baseline(
    artifact: dict, latest_df: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    from dengue_ml.models.baseline import seasonal_naive_forecast

    last_q = latest_df["quarter_start"].max()
    future_qs = _next_quarters(last_q, horizon)
    rows = []
    for city in CITIES:
        for q in future_qs:
            # Build a fake test row
            fake = pd.DataFrame([{CITY_COL: city, "quarter_start": q}])
            result = seasonal_naive_forecast(latest_df, fake)
            pred = result["predicted"].iloc[0] if not result.empty else np.nan
            rows.append({
                "city": city, "forecast_quarter": q,
                "predicted_cases": pred,
                "lower_95": np.nan, "upper_95": np.nan,
                "model_name": "baseline",
            })
    return pd.DataFrame(rows)


# ── SARIMA ────────────────────────────────────────────────────────────────────

def _forecast_sarima(
    artifact: dict, latest_df: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    city_models = artifact["models"]
    last_q = latest_df["quarter_start"].max()
    future_qs = _next_quarters(last_q, horizon)
    rows = []
    for city, fit_result in city_models.items():
        preds_log, lower_log, upper_log = forecast_sarima(fit_result, horizon=horizon)
        preds = np.expm1(preds_log)
        lower = np.expm1(lower_log)
        upper = np.expm1(upper_log)
        for i, q in enumerate(future_qs):
            rows.append({
                "city": city, "forecast_quarter": q,
                "predicted_cases": float(preds[i]),
                "lower_95": float(lower[i]),
                "upper_95": float(upper[i]),
                "model_name": "sarima",
            })
    return pd.DataFrame(rows)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def _forecast_xgb(
    artifact: dict, latest_df: pd.DataFrame, horizon: int
) -> pd.DataFrame:
    """
    Autoregressive multi-step forecast: predict Q+1, append to history,
    predict Q+2, etc. Uses quantile models for 95% CI.
    """
    feature_set  = artifact["feature_set"]
    climate_stats = artifact.get("climate_stats")
    model        = artifact["model"]
    residual_quantiles = artifact.get("residual_quantiles")

    history = latest_df.copy()
    last_q  = history["quarter_start"].max()
    future_qs = _next_quarters(last_q, horizon)

    rows = []
    for q in future_qs:
        # Build feature row for this future quarter
        # We need the full df including all history for lag computation
        # Create a stub row for each city with NaN target, then build features
        stubs = pd.DataFrame([
            {
                CITY_COL: city,
                "quarter_start": q,
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
            [CITY_COL, "quarter_start"]
        )

        X_all, _, meta_all, _ = build_features(
            extended, feature_set, climate_fit_stats=climate_stats
        )
        mask = meta_all["quarter_start"] == q

        if mask.sum() == 0:
            continue

        X_q = X_all[mask]
        cities_q = meta_all[mask][CITY_COL].values

        preds = np.expm1(predict_xgb(model, X_q))
        if residual_quantiles is not None:
            proxy = X_q[residual_quantiles["proxy_col"]].values
            lower, upper = apply_residual_quantile_table(preds, proxy, residual_quantiles)
        else:
            lower = upper = np.full_like(preds, np.nan)

        for i, city in enumerate(cities_q):
            rows.append({
                "city": city, "forecast_quarter": q,
                "predicted_cases": float(preds[i]),
                "lower_95": float(lower[i]),
                "upper_95": float(upper[i]),
                "model_name": artifact["model_name"],
            })

            # Append prediction to history for next-step lag computation
            stub_row = stubs[stubs[CITY_COL] == city].copy()
            stub_row[TARGET] = preds[i]
            history = pd.concat([history, stub_row], ignore_index=True)

    return pd.DataFrame(rows)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _next_quarters(last_q: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    quarters = []
    q = pd.Period(last_q, freq="Q") + 1
    for _ in range(n):
        quarters.append(q.to_timestamp())
        q += 1
    return quarters


def _last_val(df: pd.DataFrame, city: str | None, col: str):
    if col not in df.columns:
        return np.nan
    if city is not None:
        sub = df[df[CITY_COL] == city]
    else:
        sub = df
    if sub.empty or col not in sub.columns:
        return np.nan
    val = sub.sort_values("quarter_start")[col].dropna()
    return val.iloc[-1] if not val.empty else np.nan
