"""DORA -- Dengue Outbreak Response Assistant.

Streamlit dashboard, talks to the dengue_app FastAPI backend over HTTP only
(no direct dengue_ml/filesystem access from this module -- keeps the API
independently usable by other consumers).
"""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("DENGUE_API_URL", "http://localhost:8000")

QUARTER_LABELS = {1: "Q1", 4: "Q2", 7: "Q3", 10: "Q4"}


def quarter_label(date_str: str) -> str:
    d = pd.Timestamp(date_str)
    return f"{d.year} {QUARTER_LABELS[d.month]}"


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


def get_recommendations() -> dict:
    return api_get("/risk/recommendations")


st.set_page_config(page_title="DORA", page_icon="🦟", layout="wide")
st.title("🦟 DORA — Dengue Outbreak Response Assistant")
st.caption("AI-powered dengue forecasting and risk monitoring for Southeast Brazil")

cities_df = get_cities()
city_names = cities_df["city_name"].tolist()

mode = st.sidebar.radio("Mode", ["Current forecast", "Historical"])

# Full unfiltered backtest, used both to bound the historical date picker and
# to build the 3-year trend comparison in historical mode.
backtest_all = get_backtest()
available_quarters = sorted(backtest_all["quarter_start"].dropna().unique()) if not backtest_all.empty else []

if mode == "Historical":
    if not available_quarters:
        st.warning("No backtest data available yet.")
        st.stop()
    quarter_options = [pd.Timestamp(q).strftime("%Y-%m-%d") for q in available_quarters]
    selected_quarter = st.sidebar.selectbox(
        "Quarter", quarter_options, index=len(quarter_options) - 1,
        format_func=quarter_label,
    )
else:
    selected_quarter = None

# ── Map status data: one row per city for the quarter currently in focus ───
if mode == "Current forecast":
    forecast_df = get_forecast()
    soonest_idx = forecast_df.groupby("city_name")["forecast_quarter"].idxmin()
    status_df = forecast_df.loc[soonest_idx].copy()
    status_df["epidemic_proba"] = status_df["proxy_value"]
    status_df["is_epidemic"] = status_df["epidemic_proba"] >= 0.5
    focus_quarter_label = quarter_label(status_df["forecast_quarter"].iloc[0].strftime("%Y-%m-%d"))
else:
    status_df = backtest_all[backtest_all["quarter_start"] == pd.Timestamp(selected_quarter)].copy()
    status_df["epidemic_proba"] = status_df["predicted_proba"]
    focus_quarter_label = quarter_label(selected_quarter)

status_df = status_df.merge(cities_df, on="city_name", how="left")

st.subheader(f"Status map — {focus_quarter_label}")

fig_map = go.Figure(
    go.Scattergeo(
        lon=status_df["lon"],
        lat=status_df["lat"],
        text=status_df["city_name"],
        mode="markers+text",
        textposition="top center",
        marker=dict(
            size=22,
            color=["#e74c3c" if e else "#2ecc71" for e in status_df["is_epidemic"]],
            line=dict(width=1, color="black"),
        ),
        hovertext=[
            f"{row.city_name}<br>Predicted cases: {row.predicted_cases:,.0f}<br>"
            f"Epidemic probability: {row.epidemic_proba:.0%}"
            for row in status_df.itertuples()
        ],
        hoverinfo="text",
    )
)
fig_map.update_geos(
    scope="south america", fitbounds="locations", visible=True,
    showcountries=True, showsubunits=True,
)
fig_map.update_layout(height=450, margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_map, width="stretch")

# ── City selector + detail panel ────────────────────────────────────────────
selected_city = st.selectbox("City", city_names)

city_row = status_df[status_df["city_name"] == selected_city]
if city_row.empty:
    st.warning(f"No data for {selected_city} in this quarter.")
    st.stop()
city_row = city_row.iloc[0]

risk_tier = city_row["risk_tier"]
recs = get_recommendations()[risk_tier]

col1, col2, col3 = st.columns(3)
col1.metric("Predicted cases", f"{city_row['predicted_cases']:,.0f}")
if mode == "Current forecast":
    col2.metric("95% CI", f"{city_row['lower_95']:,.0f} – {city_row['upper_95']:,.0f}")
else:
    col2.metric("Actual cases", f"{city_row['casos_est']:,.0f}")
col3.metric("Epidemic probability", f"{city_row['epidemic_proba']:.0%}")

st.markdown(
    f"### {recs['emoji']} {recs['label']}\n*{recs['description']}*",
)
rec_cols = st.columns(len(recs["recommendations"]))
for col, (category, items) in zip(rec_cols, recs["recommendations"].items()):
    with col:
        st.markdown(f"**{category}**")
        for item in items:
            st.markdown(f"- {item}")

# ── Trend chart: last 3 years + last year highlighted + forecast/backtest ──
st.subheader(f"{selected_city} — quarterly trend")

if mode == "Current forecast":
    forecast_year = forecast_df[forecast_df["city_name"] == selected_city]["forecast_quarter"].dt.year.min()
    city_forecast = forecast_df[forecast_df["city_name"] == selected_city].sort_values("forecast_quarter")
    x_categories = [QUARTER_LABELS[d.month] for d in city_forecast["forecast_quarter"]]
else:
    focus_year = pd.Timestamp(selected_quarter).year
    forecast_year = focus_year
    city_backtest_year = backtest_all[
        (backtest_all["city_name"] == selected_city) & (backtest_all["quarter_start"].dt.year == focus_year)
    ].sort_values("quarter_start")
    x_categories = [QUARTER_LABELS[d.month] for d in city_backtest_year["quarter_start"]]

city_history = get_history(city=selected_city)

fig_trend = go.Figure()
for years_back in (3, 2, 1):
    year = forecast_year - years_back
    year_hist = city_history[city_history["quarter_start"].dt.year == year].sort_values("quarter_start")
    if year_hist.empty:
        continue
    fig_trend.add_trace(go.Scatter(
        x=[QUARTER_LABELS[d.month] for d in year_hist["quarter_start"]],
        y=year_hist["casos_est"],
        mode="lines+markers",
        name=f"{year}" + (" (last year)" if years_back == 1 else ""),
        line=dict(width=3 if years_back == 1 else 1.5, dash="solid" if years_back == 1 else "dot"),
        opacity=1.0 if years_back == 1 else 0.5,
    ))

if mode == "Current forecast":
    fig_trend.add_trace(go.Scatter(
        x=x_categories, y=city_forecast["predicted_cases"],
        mode="lines+markers", name=f"{forecast_year} forecast",
        line=dict(width=3, color="#3498db"),
    ))
    fig_trend.add_trace(go.Scatter(
        x=x_categories + x_categories[::-1],
        y=list(city_forecast["upper_95"]) + list(city_forecast["lower_95"])[::-1],
        fill="toself", fillcolor="rgba(52,152,219,0.15)", line=dict(width=0),
        name="95% CI", showlegend=True, hoverinfo="skip",
    ))
else:
    fig_trend.add_trace(go.Scatter(
        x=x_categories, y=city_backtest_year["predicted_cases"],
        mode="lines+markers", name=f"{forecast_year} predicted",
        line=dict(width=3, color="#3498db"),
    ))
    fig_trend.add_trace(go.Scatter(
        x=x_categories, y=city_backtest_year["casos_est"],
        mode="lines+markers", name=f"{forecast_year} actual",
        line=dict(width=3, color="#2c3e50", dash="dash"),
    ))

fig_trend.update_layout(
    xaxis_title="Quarter", yaxis_title="Estimated cases",
    height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_trend, width="stretch")
