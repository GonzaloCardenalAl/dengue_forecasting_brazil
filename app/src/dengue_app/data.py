"""Data access layer for the FastAPI backend.

Reads ml pipeline outputs straight from the latest run dir under
ml/results/, reusing dengue_ml's own path/aggregation helpers rather than
re-implementing them here.
"""

from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import HTTPException

from dengue_ml.config import CITY_COL
from dengue_ml.forecasting.quarterly_aggregation import (
    aggregate_weekly_classifier_to_quarterly,
    aggregate_weekly_history_to_quarterly,
    aggregate_weekly_oof_predictions_to_quarterly,
)
from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.run_dir import get_latest_run_dir

from dengue_app.risk import compute_risk_tier


def get_run_dir() -> Path:
    try:
        return get_latest_run_dir()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e


def _read_csv(filename: str, parse_dates: list[str] | None = None) -> pd.DataFrame | None:
    path = get_run_dir() / filename
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)


@lru_cache(maxsize=1)
def _history_quarterly_cached(run_dir_key: str) -> pd.DataFrame:
    weekly = prepare_model_table()
    return aggregate_weekly_history_to_quarterly(weekly)


def load_history_quarterly() -> pd.DataFrame:
    """Actual quarterly case counts (city_name, quarter_start, casos_est, ...).
    Cached per run dir -- recomputed only when a new pipeline run completes."""
    return _history_quarterly_cached(str(get_run_dir()))


def load_quarterly_forecast() -> pd.DataFrame:
    df = _read_csv("final_quarterly_forecast.csv", parse_dates=["forecast_quarter"])
    if df is None:
        raise HTTPException(
            status_code=503,
            detail="final_quarterly_forecast.csv not found in the latest run -- "
                   "has generate_forecasts.py completed for this run?",
        )
    return df


def load_classifier_model_name() -> str | None:
    path = get_run_dir() / "selected_classifier.txt"
    return path.read_text().strip() if path.exists() else None


def load_backtest_quarterly(model_name: str | None = None) -> pd.DataFrame:
    """Quarterly OOF backtest: predicted vs actual cases, plus the OOF
    classifier's epidemic status for the same quarters."""
    fold_predictions = _read_csv("fold_predictions.csv", parse_dates=["week_start"])
    if fold_predictions is None:
        raise HTTPException(status_code=503, detail="fold_predictions.csv not found in the latest run.")

    if model_name is None:
        model_name = load_quarterly_forecast()["model_name"].iloc[0]

    predicted_q = aggregate_weekly_oof_predictions_to_quarterly(fold_predictions, model_name)

    model_rows = fold_predictions[fold_predictions["model"] == model_name].copy()
    model_rows["quarter_start"] = pd.PeriodIndex(model_rows["week_start"], freq="Q").to_timestamp()
    actual_q = (
        model_rows.groupby([CITY_COL, "quarter_start"], sort=True)["casos_est"]
        .sum()
        .reset_index()
    )

    merged = predicted_q.merge(actual_q, on=[CITY_COL, "quarter_start"], how="left")

    clf_model_name = load_classifier_model_name()
    fold_predictions_clf = _read_csv("fold_predictions_clf.csv", parse_dates=["week_start"])
    if clf_model_name is not None and fold_predictions_clf is not None and not fold_predictions_clf.empty:
        clf_q = aggregate_weekly_classifier_to_quarterly(fold_predictions_clf, clf_model_name)
        merged = merged.merge(clf_q, on=[CITY_COL, "quarter_start"], how="left")
    else:
        merged["predicted_proba"] = pd.NA
        merged["is_epidemic"] = pd.NA

    return merged


def risk_tier_for(city: str, predicted_cases: float, epidemic_proba: float | None) -> str:
    history = load_history_quarterly()
    city_history = history.loc[history[CITY_COL] == city, "casos_est"]
    return compute_risk_tier(predicted_cases, city_history, epidemic_proba)


def filter_by_date_range(
    df: pd.DataFrame, date_col: str, start: str | None, end: str | None
) -> pd.DataFrame:
    if start is not None:
        df = df[df[date_col] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df[date_col] <= pd.Timestamp(end)]
    return df


def records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records: Timestamps -> ISO date strings, NaN/NaT -> None."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.astype(object).where(out.notna(), None).to_dict(orient="records")
