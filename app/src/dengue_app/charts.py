"""Plotly figure builders for the DORA dashboard.

Each function returns a `go.Figure` and makes no Streamlit calls, so charts
stay independently testable/reusable from the page modules that render them.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

QUARTER_LABELS = {1: "Q1", 4: "Q2", 7: "Q3", 10: "Q4"}

# Q4->Q1->Q2->Q3 display order for the seasonal profile chart -- see
# data.load_seasonal_profile for why seasons are grouped this way.
SEASON_QUARTER_POS_LABELS = {1: "Q4", 2: "Q1", 3: "Q2", 4: "Q3"}

BRAZIL_CENTER = {"lat": -14.2, "lon": -51.9}
BRAZIL_ZOOM = 3.2

ACTUAL_COLOR = "#2c3e50"
FORECAST_COLOR = "#e67e22"


def quarter_label(date_str: str) -> str:
    d = pd.Timestamp(date_str)
    return f"{d.year} {QUARTER_LABELS[d.month]}"


def month_label(date_str: str) -> str:
    return pd.Timestamp(date_str).strftime("%b %Y")


def build_status_map(status_df: pd.DataFrame, selected_city: str | None = None) -> go.Figure:
    """City status markers over OpenStreetMap tiles, centered on all of
    Brazil by default (rather than fitbounds-to-points) so the map gives
    geographic context and the user zooms into the cities themselves."""
    is_selected = status_df["city_name"] == selected_city
    fig = go.Figure(
        go.Scattermapbox(
            lon=status_df["lon"],
            lat=status_df["lat"],
            text=status_df["city_name"],
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=[28 if sel else 18 for sel in is_selected],
                color=["#e74c3c" if e else "#2ecc71" for e in status_df["is_epidemic"]],
            ),
            hovertext=[
                f"{row.city_name}<br>Predicted cases: {row.predicted_cases:,.0f}<br>"
                f"Epidemic probability: {row.epidemic_proba:.0%}"
                for row in status_df.itertuples()
            ],
            hoverinfo="text",
        )
    )
    fig.update_layout(
        mapbox=dict(style="open-street-map", center=BRAZIL_CENTER, zoom=BRAZIL_ZOOM),
        height=450,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


def build_trend_chart(
    city_history: pd.DataFrame,
    focus_df: pd.DataFrame,
    x_categories: list[str],
    *,
    is_forecast_mode: bool,
    forecast_year: int,
) -> go.Figure:
    """Multi-year quarterly trend: prior years thin/light, last year bold,
    plus the forecast (with 95% CI band) or backtest-vs-actual for the
    focus year."""
    prior_year_colors = {3: "#c3cdd8", 2: "#8a9bb0", 1: "#2c3e50"}

    fig = go.Figure()
    for years_back in (3, 2, 1):
        year = forecast_year - years_back
        year_hist = city_history[city_history["quarter_start"].dt.year == year].sort_values("quarter_start")
        if year_hist.empty:
            continue
        fig.add_trace(go.Scatter(
            x=[QUARTER_LABELS[d.month] for d in year_hist["quarter_start"]],
            y=year_hist["casos_est"],
            mode="lines+markers",
            marker=dict(size=7 if years_back == 1 else 5, symbol="circle"),
            name=f"{year}" + (" (last year)" if years_back == 1 else ""),
            line=dict(width=3 if years_back == 1 else 1.5, dash="solid" if years_back == 1 else "dot",
                      color=prior_year_colors[years_back]),
            opacity=1.0 if years_back == 1 else 0.7,
        ))

    if is_forecast_mode:
        fig.add_trace(go.Scatter(
            x=x_categories, y=focus_df["predicted_cases"],
            mode="lines+markers", marker=dict(size=8), name=f"{forecast_year} forecast",
            line=dict(width=3, color=FORECAST_COLOR),
        ))
        fig.add_trace(go.Scatter(
            x=x_categories + x_categories[::-1],
            y=list(focus_df["upper_95"]) + list(focus_df["lower_95"])[::-1],
            fill="toself", fillcolor="rgba(230,126,34,0.15)", line=dict(width=0),
            name="95% CI", showlegend=True, hoverinfo="skip",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x_categories, y=focus_df["predicted_cases"],
            mode="lines+markers", marker=dict(size=8), name=f"{forecast_year} predicted",
            line=dict(width=3, color=FORECAST_COLOR),
        ))
        fig.add_trace(go.Scatter(
            x=x_categories, y=focus_df["casos_est"],
            mode="lines+markers", marker=dict(size=8, symbol="diamond"), name=f"{forecast_year} actual",
            line=dict(width=3, color=ACTUAL_COLOR, dash="dash"),
        ))

    fig.update_layout(
        xaxis_title="Quarter", yaxis_title="Estimated cases",
        height=420, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#ddd", rangemode="tozero")
    return fig


def build_incidence_bar_chart(
    actual_labels: list[str], actual_incidence: list[float],
    forecast_labels: list[str], forecast_incidence: list[float],
) -> go.Figure:
    """Previous 12 actual months vs next 12 forecasted months, on the
    incidence-per-100k scale so it's comparable across cities of very
    different population sizes -- monthly rather than quarterly bars for
    finer resolution."""
    fig = go.Figure()
    fig.add_trace(go.Bar(x=actual_labels, y=actual_incidence, name="Actual", marker_color=ACTUAL_COLOR))
    fig.add_trace(go.Bar(x=forecast_labels, y=forecast_incidence, name="Forecast", marker_color=FORECAST_COLOR))
    fig.update_layout(
        xaxis_title="Month", yaxis_title="Incidence per 100k",
        height=380, bargap=0.25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        plot_bgcolor="white",
    )
    fig.update_xaxes(categoryorder="array", categoryarray=actual_labels + forecast_labels,
                      showgrid=False, tickangle=-45)
    fig.update_yaxes(showgrid=True, gridcolor="#eee", zeroline=True, zerolinecolor="#ddd")
    return fig


def build_seasonal_profile_chart(seasonal_df: pd.DataFrame) -> go.Figure:
    """Annual quarterly profile for one city: each historical season (Q4 of
    year Y through Q3 of year Y+1) drawn as its own line, colored if it
    contained an epidemic week, light grey otherwise, plus a dashed average
    line -- single-city twin of ml/notebooks figure 09."""
    palette = px.colors.qualitative.Dark24
    epidemic_seasons = sorted(seasonal_df.loc[seasonal_df["is_epidemic_season"], "season_year"].unique())
    color_by_season = {sy: palette[i % len(palette)] for i, sy in enumerate(epidemic_seasons)}

    fig = go.Figure()
    for season_year, grp in seasonal_df.groupby("season_year"):
        grp = grp.sort_values("season_quarter_pos")
        is_epidemic = bool(grp["is_epidemic_season"].iloc[0])
        fig.add_trace(go.Scatter(
            x=grp["season_quarter_pos"], y=grp["p_inc100k"],
            mode="lines+markers",
            line=dict(color=color_by_season[season_year] if is_epidemic else "lightgrey",
                      width=2.2 if is_epidemic else 1),
            marker=dict(size=5 if is_epidemic else 3),
            opacity=1.0 if is_epidemic else 0.5,
            name=f"{season_year}/{season_year + 1}",
            showlegend=is_epidemic,
        ))

    avg = seasonal_df.groupby("season_quarter_pos")["p_inc100k"].mean().sort_index()
    fig.add_trace(go.Scatter(
        x=avg.index, y=avg.values, mode="lines",
        line=dict(color="black", dash="dash", width=2.5), name="Average (all seasons)",
    ))

    fig.update_xaxes(tickmode="array", tickvals=[1, 2, 3, 4],
                      ticktext=[SEASON_QUARTER_POS_LABELS[p] for p in (1, 2, 3, 4)], title="Quarter")
    fig.update_yaxes(title="Incidence per 100k (quarterly)", showgrid=True, gridcolor="#eee")
    fig.update_layout(height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02), plot_bgcolor="white")
    return fig
