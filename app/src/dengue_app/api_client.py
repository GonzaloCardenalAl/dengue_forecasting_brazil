"""HTTP client for the dengue_app FastAPI backend.

Pure data-fetch functions, no Streamlit calls -- kept separate from the
chart/page modules so the data layer stays independently testable.
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("DENGUE_API_URL", "http://localhost:8000")


@st.cache_data(ttl=300)
def api_get(path: str, params: dict | None = None) -> list | dict:
    r = requests.get(f"{API_URL}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def get_cities() -> pd.DataFrame:
    return pd.DataFrame(api_get("/cities"))


def get_forecast() -> pd.DataFrame:
    df = pd.DataFrame(api_get("/forecast/quarterly"))
    df["forecast_quarter"] = pd.to_datetime(df["forecast_quarter"])
    return df


def get_backtest(city: str | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    params = {k: v for k, v in {"city": city, "start": start, "end": end}.items() if v is not None}
    df = pd.DataFrame(api_get("/backtest/quarterly", params=params))
    if not df.empty:
        df["quarter_start"] = pd.to_datetime(df["quarter_start"])
    return df


def get_history(city: str | None = None, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    params = {k: v for k, v in {"city": city, "start": start, "end": end}.items() if v is not None}
    df = pd.DataFrame(api_get("/history/quarterly", params=params))
    if not df.empty:
        df["quarter_start"] = pd.to_datetime(df["quarter_start"])
    return df


def get_seasonal_profile(city: str) -> pd.DataFrame:
    return pd.DataFrame(api_get("/history/seasonal-profile", params={"city": city}))


def get_history_monthly(city: str) -> pd.DataFrame:
    df = pd.DataFrame(api_get("/history/monthly", params={"city": city}))
    if not df.empty:
        df["month_start"] = pd.to_datetime(df["month_start"])
    return df


def get_forecast_monthly(city: str) -> pd.DataFrame:
    df = pd.DataFrame(api_get("/forecast/monthly", params={"city": city}))
    if not df.empty:
        df["forecast_month"] = pd.to_datetime(df["forecast_month"])
    return df


def get_recommendations() -> dict:
    return api_get("/risk/recommendations")
