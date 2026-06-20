import numpy as np
import pandas as pd

from dengue_ml.config import CITY_COL, TARGET, CITIES
from dengue_ml.features.feature_pipeline import build_features, build_classification_features
from dengue_ml.features.forecast_proxies import last_val as _last_val, climatological_val as _climatological_val
from dengue_ml.features.rt_estimation import estimate_rt_p_rt1
from dengue_ml.training.final_train import predict_proba_classifier
from dengue_ml.validation.conditional_residuals import apply_residual_quantile_table

# Generation-time cap (weeks) for the Rt/p_rt1 estimator -- matches the
# reference implementation's GTmax. See features/rt_estimation.py.
_RT_GT_MAX = 5


def next_weeks(last_w: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    return [last_w + pd.DateOffset(weeks=i) for i in range(1, n + 1)]


def forecast_with_rt_estimation(
    artifact: dict, latest_df: pd.DataFrame, horizon: int, predict_fn, classifier_artifact: dict | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Two passes over the same autoregressive loop, starting right after
    `latest_df`'s last week and running `horizon` weeks forward:

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

    Used both for production forecasting (`forecasting/forecast_next_52w.py`,
    `latest_df` = real history up to today) and for the autoregressive CV
    backtest (`validation/autoregressive_cv.py`, `latest_df` = an outer
    fold's training window, `horizon` = that fold's test-window length) --
    the rollout mechanics are identical either way, only what "now" means
    differs.

    Returns (rows_df, final_history_df) -- final_history_df (latest_df +
    every predicted week) is occasionally useful to callers that want the
    full extended trajectory (e.g. for diagnostics); most callers only need
    rows_df.
    """
    feature_set   = artifact["feature_set"]
    climate_stats = artifact.get("climate_stats")
    model         = artifact["model"]
    model_name    = artifact["model_name"]
    residual_quantiles = artifact.get("residual_quantiles")

    last_w = latest_df["week_start"].max()
    future_ws = next_weeks(last_w, horizon)

    _, draft_history = _autoregressive_loop(
        model, predict_fn, model_name, latest_df, future_ws, feature_set, climate_stats,
        residual_quantiles=None, classifier_artifact=None, rt_lookup=None,
    )

    rt_lookup = _estimate_rt_lookup(draft_history, future_ws)

    rows, final_history = _autoregressive_loop(
        model, predict_fn, model_name, latest_df, future_ws, feature_set, climate_stats,
        residual_quantiles=residual_quantiles, classifier_artifact=classifier_artifact, rt_lookup=rt_lookup,
    )
    return rows, final_history


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

        # proxy_value is computed whenever a classifier is available,
        # independent of whether residual_quantiles is provided -- callers
        # that only want the raw rollout + regime label to calibrate a table
        # FROM (e.g. validation/autoregressive_cv.py) pass
        # residual_quantiles=None but still need proxy_value populated.
        if classifier_artifact is not None:
            proxy_by_city = _classifier_proxy_by_city(classifier_artifact, extended, w)
            proxy = np.array([proxy_by_city.get(c, np.nan) for c in cities_w])
        else:
            proxy = np.full_like(preds, np.nan)

        if residual_quantiles is not None:
            lower, upper = apply_residual_quantile_table(preds, proxy, residual_quantiles)
        else:
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
    extra_ws = next_weeks(future_ws[-1], n_extra)
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


