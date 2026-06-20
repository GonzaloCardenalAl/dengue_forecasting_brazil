import numpy as np
import pandas as pd
import joblib

from dengue_ml.config import CITY_COL, TARGET, FORECAST_HORIZON, CITIES
from dengue_ml.features.feature_pipeline import build_features, build_classification_features
from dengue_ml.features.rt_estimation import estimate_rt_p_rt1
from dengue_ml.models.xgboost_models import predict_xgb
from dengue_ml.models.xrfm_models import predict_xrfm
from dengue_ml.models.sarima import forecast_sarima
from dengue_ml.training.final_train import predict_proba_classifier
from dengue_ml.validation.conditional_residuals import apply_residual_quantile_table

# Generation-time cap (weeks) for the Rt/p_rt1 estimator -- matches the
# reference implementation's GTmax. See features/rt_estimation.py.
_RT_GT_MAX = 5


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
    future_ws = _next_weeks(last_w, horizon)
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
    future_ws = _next_weeks(last_w, horizon)
    rows = []
    for city, fit_result in city_models.items():
        preds_log, lower_log, upper_log = forecast_sarima(fit_result, horizon=horizon)
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

def _classifier_proxy_by_city(
    classifier_artifact: dict | None, extended: pd.DataFrame, w: pd.Timestamp
) -> dict:
    """
    {city_name: predicted_proba} for week `w`, built from the classifier's
    own feature set (which may differ from the regression model's) on the
    same extended (history + already-forecasted weeks) df. {} if no
    classifier artifact is available -- callers treat missing cities as NaN.
    """
    if classifier_artifact is None:
        return {}
    X_clf, _, meta_clf, _ = build_classification_features(extended, classifier_artifact["feature_set"])
    mask = meta_clf["week_start"] == w
    if mask.sum() == 0:
        return {}
    proba = predict_proba_classifier(classifier_artifact, X_clf[mask])
    cities = meta_clf[mask][CITY_COL].values
    return dict(zip(cities, proba))


def _forecast_xgb(
    artifact: dict, latest_df: pd.DataFrame, horizon: int,
    classifier_artifact: dict | None = None,
) -> pd.DataFrame:
    """Autoregressive multi-step forecast for XGBoost. See _forecast_with_rt_estimation."""
    return _forecast_with_rt_estimation(artifact, latest_df, horizon, predict_xgb, classifier_artifact)


# ── xRFM ──────────────────────────────────────────────────────────────────────

def _forecast_xrfm(
    artifact: dict, latest_df: pd.DataFrame, horizon: int,
    classifier_artifact: dict | None = None,
) -> pd.DataFrame:
    """
    Autoregressive multi-step forecast for xRFM -- identical to _forecast_xgb;
    xRFM's val-set requirement only applies at fit time (already satisfied
    when the model in `artifact` was trained), so this is a pure inference
    loop just like XGBoost's. See _forecast_with_rt_estimation.
    """
    return _forecast_with_rt_estimation(artifact, latest_df, horizon, predict_xrfm, classifier_artifact)


# ── Shared two-pass autoregressive loop (XGBoost + xRFM) ─────────────────────

def _forecast_with_rt_estimation(
    artifact: dict, latest_df: pd.DataFrame, horizon: int, predict_fn, classifier_artifact: dict | None,
) -> pd.DataFrame:
    """
    Two passes over the same autoregressive loop:

    1. Draft pass: predict W+1, append to history, predict W+2, etc., with
       Rt/p_rt1/sustained_rt carried forward flat (same as before this
       feature existed) -- this pass only exists to produce a plausible
       case trajectory for the forecast horizon, not the final answer.
    2. Estimate Rt/p_rt1 (features/rt_estimation.py's temperature-dependent
       Wallinga-Teunis estimator) over the full extended series (real
       history + draft forecast), once -- see _estimate_rt_lookup.
    3. Final pass: identical loop, now seeding Rt/p_rt1/sustained_rt stubs
       from that estimate instead of flat carry-forward, with the real CI
       calibration (residual_quantiles/classifier_artifact) applied.

    Recomputing the full loop twice is deliberately cheap relative to
    re-running the (much more expensive) generation-time/renewal-equation
    estimation every single forecast week -- see module docstring in
    features/rt_estimation.py for why a one-shot calculation over the whole
    horizon is preferred over an incremental per-week update.
    """
    feature_set   = artifact["feature_set"]
    climate_stats = artifact.get("climate_stats")
    model         = artifact["model"]
    model_name    = artifact["model_name"]
    residual_quantiles = artifact.get("residual_quantiles")

    last_w = latest_df["week_start"].max()
    future_ws = _next_weeks(last_w, horizon)

    _, draft_history = _autoregressive_loop(
        model, predict_fn, model_name, latest_df, future_ws, feature_set, climate_stats,
        residual_quantiles=None, classifier_artifact=None, rt_lookup=None,
    )

    rt_lookup = _estimate_rt_lookup(draft_history, future_ws)

    rows, _ = _autoregressive_loop(
        model, predict_fn, model_name, latest_df, future_ws, feature_set, climate_stats,
        residual_quantiles=residual_quantiles, classifier_artifact=classifier_artifact, rt_lookup=rt_lookup,
    )
    return rows


def _autoregressive_loop(
    model, predict_fn, model_name: str,
    latest_df: pd.DataFrame, future_ws: list, feature_set: str, climate_stats,
    residual_quantiles, classifier_artifact, rt_lookup: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Shared per-week autoregressive loop body for both XGBoost and xRFM.
    Returns (rows_df, final_history_df) -- final_history_df (latest_df +
    every predicted week) is needed by the caller's draft pass to seed
    _estimate_rt_lookup; the final pass only uses rows_df.
    """
    history = latest_df.copy()
    rows = []
    for w in future_ws:
        # Build feature row for this future week
        # We need the full df including all history for lag computation
        # Create a stub row for each city with NaN target, then build features
        stubs = pd.DataFrame([
            {
                CITY_COL: city,
                "week_start": w,
                TARGET: np.nan,
                "tempmed": _climatological_val(history, city, "tempmed", w),
                "humidmed": _climatological_val(history, city, "humidmed", w),
                "pop": _last_val(history, city, "pop"),
                "p_inc100k": _last_val(history, city, "p_inc100k"),
                **_rt_stub_values(rt_lookup, history, city, w),
                "nino34_anom": _last_val(history, None, "nino34_anom"),
                "roni": _last_val(history, None, "roni"),
            }
            for city in CITIES
        ])

        extended = pd.concat([history, stubs], ignore_index=True).sort_values(
            [CITY_COL, "week_start"]
        )

        X_all, _, meta_all, _ = build_features(
            extended, feature_set, climate_fit_stats=climate_stats
        )
        mask = meta_all["week_start"] == w

        if mask.sum() == 0:
            continue

        X_w = X_all[mask]
        cities_w = meta_all[mask][CITY_COL].values

        preds = np.expm1(predict_fn(model, X_w))
        if residual_quantiles is not None:
            proxy_by_city = _classifier_proxy_by_city(classifier_artifact, extended, w)
            proxy = np.array([proxy_by_city.get(c, np.nan) for c in cities_w])
            lower, upper = apply_residual_quantile_table(preds, proxy, residual_quantiles)
        else:
            proxy = np.full_like(preds, np.nan)
            lower = upper = np.full_like(preds, np.nan)

        for i, city in enumerate(cities_w):
            rows.append({
                "city": city, "forecast_week": w,
                "predicted_cases": float(preds[i]),
                "lower_95": float(lower[i]),
                "upper_95": float(upper[i]),
                "proxy_value": float(proxy[i]),
                "model_name": model_name,
            })

            # Append prediction to history for next-step lag computation
            stub_row = stubs[stubs[CITY_COL] == city].copy()
            stub_row[TARGET] = preds[i]
            history = pd.concat([history, stub_row], ignore_index=True)

    return pd.DataFrame(rows), history


def _rt_stub_values(rt_lookup: dict | None, history: pd.DataFrame, city: str, w: pd.Timestamp) -> dict:
    """Rt/p_rt1/sustained_rt for one forecast stub row: from the estimated
    lookup if available (final pass), else flat carry-forward (draft pass,
    or any week the lookup doesn't cover)."""
    if rt_lookup is not None and (city, w) in rt_lookup:
        Rt_est, p_rt1_est, sustained_rt_est = rt_lookup[(city, w)]
        return {"Rt": Rt_est, "p_rt1": p_rt1_est, "sustained_rt": sustained_rt_est}
    return {
        "Rt": _last_val(history, city, "Rt"),
        "p_rt1": _last_val(history, city, "p_rt1"),
        "sustained_rt": _last_val(history, city, "sustained_rt"),
    }


def _estimate_rt_lookup(draft_history: pd.DataFrame, future_ws: list) -> dict:
    """
    Run the temperature-dependent Wallinga-Teunis estimator once per city
    over the full extended series (real history + draft forecast), and
    return {(city, week_start): (Rt, p_rt1, sustained_rt)} for the forecast
    weeks only -- historical weeks keep InfoDengue's real reported values
    untouched in `history` itself; this lookup only ever feeds stub rows for
    weeks beyond the last real observation.
    """
    n_extra = _RT_GT_MAX  # extra weeks of climatological temp beyond the horizon
    extra_ws = _next_weeks(future_ws[-1], n_extra)
    future_set = set(future_ws)

    lookup = {}
    for city in CITIES:
        city_df = draft_history[draft_history[CITY_COL] == city].sort_values("week_start").reset_index(drop=True)
        # The draft pass's predicted casos_est can dip slightly negative for
        # very-low-incidence cities/weeks (np.expm1 of a log1p-prediction
        # near 0 is only bounded below by -1, not 0) -- the renewal-equation
        # machinery requires non-negative incidence, so clip here. Doesn't
        # touch real historical casos_est (always >= 0).
        cases = np.clip(city_df[TARGET].values, 0.0, None)
        temp = city_df["tempmed"].values
        extra_temp = np.array([_climatological_val(city_df, city, "tempmed", w) for w in extra_ws])
        temp_extended = np.concatenate([temp, extra_temp])

        R, p_rt1 = estimate_rt_p_rt1(cases, temp_extended, gt_max=_RT_GT_MAX, nsim=500, seed=0)

        # Same rule as preprocessing.py's sustained_rt: p_rt1 > 0.95 for >= 3
        # consecutive weeks, recomputed over the full (history + estimated
        # forecast) p_rt1 series so the forecast horizon's flag is internally
        # consistent with how it's defined historically.
        high_rt = pd.Series(p_rt1 > 0.95)
        sustained_rt = (high_rt.rolling(3).sum() >= 3).values

        weeks = city_df["week_start"].values
        for j, w in enumerate(weeks):
            w = pd.Timestamp(w)
            if w in future_set:
                lookup[(city, w)] = (float(R[j]), float(p_rt1[j]), bool(sustained_rt[j]))

    return lookup


# ── Helpers ───────────────────────────────────────────────────────────────────

def _next_weeks(last_w: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    return [last_w + pd.DateOffset(weeks=i) for i in range(1, n + 1)]


def _last_val(df: pd.DataFrame, city: str | None, col: str):
    if col not in df.columns:
        return np.nan
    if city is not None:
        sub = df[df[CITY_COL] == city]
    else:
        sub = df
    if sub.empty or col not in sub.columns:
        return np.nan
    val = sub.sort_values("week_start")[col].dropna()
    return val.iloc[-1] if not val.empty else np.nan


def _climatological_val(df: pd.DataFrame, city: str, col: str, target_week: pd.Timestamp):
    """
    City + ISO-week-of-year historical mean of `col` -- same logic as
    preprocessing._impute_climate_columns, but looked up for one future week
    instead of filling NaNs in place. Climate is seasonal, so the
    week-of-year mean is a far better forecast-horizon stand-in than flat
    carry-forward of the last observed value.
    """
    if col not in df.columns:
        return np.nan
    sub = df[df[CITY_COL] == city]
    if sub.empty:
        return np.nan
    target_woy = target_week.isocalendar().week
    woy = sub["week_start"].dt.isocalendar().week
    vals = sub.loc[woy == target_woy, col].dropna()
    if not vals.empty:
        return vals.mean()
    # Fallback (should only hit for ISO week 53, which most years lack):
    # last known value rather than NaN.
    return _last_val(df, city, col)
