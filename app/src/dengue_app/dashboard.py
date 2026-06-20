"""DORA -- Dengue Outbreak Response Assistant.

Streamlit dashboard, talks to the dengue_app FastAPI backend over HTTP only
(no direct dengue_ml/filesystem access from this module -- keeps the API
independently usable by other consumers).
"""

import hmac
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("DENGUE_API_URL", "http://localhost:8000")
DASHBOARD_PASSWORD = os.environ.get("DENGUE_DASHBOARD_PASSWORD")
LOGO_PATH = Path(__file__).resolve().parents[2] / "DORA_logo.png"

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


st.set_page_config(page_title="DORA", page_icon=str(LOGO_PATH), layout="wide")

# Password gate -- skipped entirely if DENGUE_DASHBOARD_PASSWORD isn't set
# (so local dev without the env var still works; production deployments must
# set it). One shared secret, no per-user accounts.
if DASHBOARD_PASSWORD and not st.session_state.get("authenticated"):
    st.title("DORA")
    pw = st.text_input("Password", type="password")
    if st.button("Log in"):
        if hmac.compare_digest(pw, DASHBOARD_PASSWORD):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

with st.sidebar:
    st.markdown("### Admin")
    if st.button("Refresh data & forecast"):
        with st.spinner("Fetching latest InfoDengue data and re-running inference..."):
            try:
                r = requests.post(
                    f"{API_URL}/admin/refresh",
                    headers={"X-Admin-Token": DASHBOARD_PASSWORD or ""},
                    timeout=120,
                )
                r.raise_for_status()
                summary = r.json()
                api_get.clear()  # bust the 300s cache so the refreshed data shows immediately
                st.success(
                    f"Refreshed. Rows added: {summary.get('rows_added')}. "
                    f"Forecast now covers: {', '.join(summary.get('forecast_quarters', []))}"
                )
                st.rerun()
            except requests.HTTPError as e:
                detail = e.response.text if e.response is not None else str(e)
                st.error(f"Refresh failed: {detail}")
            except requests.RequestException as e:
                st.error(f"Refresh failed: {e}")

# Font (Inter) + larger mode selector + larger city selector + larger
# subtitle/title -- several selector variants stacked since Streamlit's
# internal data-testids can shift between versions.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .dora-title h1 { font-size: 3.2rem !important; margin: 0 !important; line-height: 1; }
    .dora-subtitle { font-size: 1.25rem !important; color: #555; margin-top: 4px; }

    div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p { font-size: 1.3rem !important; }
    div[data-testid="stRadio"] label p { font-size: 1.2rem !important; }

    div[data-testid="stSelectbox"] label p { font-size: 1.25rem !important; }
    div[data-testid="stSelectbox"] div[data-baseweb="select"] * { font-size: 1.15rem !important; }
    div[data-baseweb="popover"] li { font-size: 1.15rem !important; }
    ul[role="listbox"] * { font-size: 1.15rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

logo_col, title_col = st.columns([1, 5], vertical_alignment="center")
with logo_col:
    st.image(str(LOGO_PATH), width=320)
with title_col:
    st.markdown('<div class="dora-title">', unsafe_allow_html=True)
    st.title("Dengue Outbreak Response Assistant")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        '<p class="dora-subtitle">AI-powered dengue forecasting and risk monitoring for Southeast Brazil</p>',
        unsafe_allow_html=True,
    )

cities_df = get_cities()
city_names = cities_df["city_name"].tolist()

mode = st.radio("Mode", ["Current forecast", "Future forecast", "Historical"], horizontal=True)
is_forecast_mode = mode in ("Current forecast", "Future forecast")

forecast_df = get_forecast() if is_forecast_mode else None
# Full unfiltered backtest, used both to bound the historical quarter picker
# and to build the 3-year trend comparison.
backtest_all = get_backtest()
available_quarters = sorted(backtest_all["quarter_start"].dropna().unique()) if not backtest_all.empty else []
backtest_quarter_options = [pd.Timestamp(q).strftime("%Y-%m-%d") for q in available_quarters]

if mode == "Historical" and not backtest_quarter_options:
    st.warning("No backtest data available yet.")
    st.stop()

if mode == "Future forecast":
    forecast_quarter_options = [
        d.strftime("%Y-%m-%d") for d in sorted(forecast_df["forecast_quarter"].unique())
    ]
    if not forecast_quarter_options:
        st.warning("No forecast data available yet.")
        st.stop()

selected_city = st.selectbox("City", city_names)

# ── Map (left) + status stats (right), same row ─────────────────────────────
map_col, stats_col = st.columns([2, 1])

with map_col:
    map_title_col, picker_col = st.columns([2, 2])

    selected_quarter = None
    selected_forecast_quarter = None
    focus_quarter_label = None

    if mode == "Historical":
        with picker_col:
            selected_quarter = st.selectbox(
                "Quarter", backtest_quarter_options, index=len(backtest_quarter_options) - 1,
                format_func=quarter_label, label_visibility="collapsed",
            )
        focus_quarter_label = quarter_label(selected_quarter)
    elif mode == "Future forecast":
        with picker_col:
            selected_forecast_quarter = st.selectbox(
                "Forecast quarter", forecast_quarter_options, index=0,
                format_func=quarter_label, label_visibility="collapsed",
            )
        focus_quarter_label = quarter_label(selected_forecast_quarter)

    if mode == "Current forecast":
        soonest_idx = forecast_df.groupby("city_name")["forecast_quarter"].idxmin()
        status_df = forecast_df.loc[soonest_idx].copy()
        status_df["epidemic_proba"] = status_df["proxy_value"]
        status_df["is_epidemic"] = status_df["epidemic_proba"] >= 0.5
        focus_quarter_label = quarter_label(status_df["forecast_quarter"].iloc[0].strftime("%Y-%m-%d"))
    elif mode == "Future forecast":
        status_df = forecast_df[forecast_df["forecast_quarter"] == pd.Timestamp(selected_forecast_quarter)].copy()
        status_df["epidemic_proba"] = status_df["proxy_value"]
        status_df["is_epidemic"] = status_df["epidemic_proba"] >= 0.5
    else:
        status_df = backtest_all[backtest_all["quarter_start"] == pd.Timestamp(selected_quarter)].copy()
        status_df["epidemic_proba"] = status_df["predicted_proba"]

    status_df = status_df.merge(cities_df, on="city_name", how="left")

    with map_title_col:
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

city_row = status_df[status_df["city_name"] == selected_city]
if city_row.empty:
    st.warning(f"No data for {selected_city} in this quarter.")
    st.stop()
city_row = city_row.iloc[0]
risk_tier = city_row["risk_tier"]

with stats_col:
    st.markdown(f"#### {selected_city}")
    st.metric("Predicted cases", f"{city_row['predicted_cases']:,.0f}")
    if is_forecast_mode:
        st.metric("95% CI", f"{city_row['lower_95']:,.0f} – {city_row['upper_95']:,.0f}")
    else:
        st.metric("Actual cases", f"{city_row['casos_est']:,.0f}")
    st.metric("Epidemic probability", f"{city_row['epidemic_proba']:.0%}")

# ── Trend chart: last 3 years + last year highlighted + forecast/backtest ──
st.subheader(f"{selected_city} — quarterly trend")

if is_forecast_mode:
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

if is_forecast_mode:
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

# ── Risk tier badge + decision-support recommendations (below the charts) ──
recs = get_recommendations()[risk_tier]
st.markdown(f"### {recs['emoji']} {recs['label']}\n*{recs['description']}*")
rec_cols = st.columns(len(recs["recommendations"]))
for col, (category, items) in zip(rec_cols, recs["recommendations"].items()):
    with col:
        st.markdown(f"**{category}**")
        for item in items:
            st.markdown(f"- {item}")
