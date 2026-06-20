"""Home page: status map, trend/incidence charts, seasonal profile, and
risk-tier recommendations for the selected city."""

import pandas as pd
import streamlit as st

from dengue_app import api_client, charts


def render_home() -> None:
    st.markdown('<div class="dora-title">', unsafe_allow_html=True)
    st.title("Dengue Outbreak Response Assistant")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        '<p class="dora-subtitle">AI-powered dengue forecasting and risk monitoring for Southeast Brazil</p>',
        unsafe_allow_html=True,
    )

    cities_df = api_client.get_cities()
    city_names = cities_df["city_name"].tolist()
    population_by_city = cities_df.set_index("city_name")["population"].to_dict()

    mode = st.radio("Mode", ["Current forecast", "Future forecast", "Historical"], horizontal=True)
    is_forecast_mode = mode in ("Current forecast", "Future forecast")

    # Always loaded (not just in forecast modes) -- the incidence bar chart
    # below needs the next-4-quarter forecast regardless of `mode`.
    forecast_df = api_client.get_forecast()
    # Full unfiltered backtest, used both to bound the historical quarter picker
    # and to build the 3-year trend comparison.
    backtest_all = api_client.get_backtest()
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
    population = population_by_city.get(selected_city)

    # ── Map (left) + status stats (right), same row ─────────────────────────
    map_col, stats_col = st.columns([2, 1])

    with map_col:
        map_title_col, picker_col = st.columns([2, 2])

        selected_quarter = None
        selected_forecast_quarter = None
        focus_quarter_label = None
        focus_quarter_ts = None

        if mode == "Historical":
            with picker_col:
                selected_quarter = st.selectbox(
                    "Quarter", backtest_quarter_options, index=len(backtest_quarter_options) - 1,
                    format_func=charts.quarter_label, label_visibility="collapsed",
                )
            focus_quarter_label = charts.quarter_label(selected_quarter)
            focus_quarter_ts = pd.Timestamp(selected_quarter)
        elif mode == "Future forecast":
            with picker_col:
                selected_forecast_quarter = st.selectbox(
                    "Forecast quarter", forecast_quarter_options, index=0,
                    format_func=charts.quarter_label, label_visibility="collapsed",
                )
            focus_quarter_label = charts.quarter_label(selected_forecast_quarter)
            focus_quarter_ts = pd.Timestamp(selected_forecast_quarter)

        if mode == "Current forecast":
            soonest_idx = forecast_df.groupby("city_name")["forecast_quarter"].idxmin()
            status_df = forecast_df.loc[soonest_idx].copy()
            status_df["epidemic_proba"] = status_df["proxy_value"]
            status_df["is_epidemic"] = status_df["epidemic_proba"] >= 0.5
            focus_quarter_label = charts.quarter_label(status_df["forecast_quarter"].iloc[0].strftime("%Y-%m-%d"))
            focus_quarter_ts = status_df["forecast_quarter"].iloc[0]
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

        st.plotly_chart(charts.build_status_map(status_df, selected_city), width="stretch")

    city_row = status_df[status_df["city_name"] == selected_city]
    if city_row.empty:
        st.warning(f"No data for {selected_city} in this quarter.")
        st.stop()
    city_row = city_row.iloc[0]
    risk_tier = city_row["risk_tier"]
    focus_cases = city_row["predicted_cases"] if is_forecast_mode else city_row["casos_est"]
    focus_incidence = focus_cases / population * 100_000 if population else None

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
        x_categories = [charts.QUARTER_LABELS[d.month] for d in city_forecast["forecast_quarter"]]
        focus_df = city_forecast
    else:
        focus_year = pd.Timestamp(selected_quarter).year
        forecast_year = focus_year
        city_backtest_year = backtest_all[
            (backtest_all["city_name"] == selected_city) & (backtest_all["quarter_start"].dt.year == focus_year)
        ].sort_values("quarter_start")
        x_categories = [charts.QUARTER_LABELS[d.month] for d in city_backtest_year["quarter_start"]]
        focus_df = city_backtest_year

    city_history = api_client.get_history(city=selected_city)

    st.plotly_chart(
        charts.build_trend_chart(city_history, focus_df, x_categories,
                                  is_forecast_mode=is_forecast_mode, forecast_year=forecast_year),
        width="stretch",
    )

    # ── Incidence bar chart: previous 4 actual quarters vs next 4 forecasted ──
    st.subheader(f"{selected_city} — incidence per 100k, last 4 quarters vs next 4 forecasted")
    city_forecast_all = forecast_df[forecast_df["city_name"] == selected_city].sort_values("forecast_quarter")
    if population and not city_forecast_all.empty and not city_history.empty:
        earliest_forecast_quarter = city_forecast_all["forecast_quarter"].min()
        prev_actual = city_history[city_history["quarter_start"] < earliest_forecast_quarter].sort_values(
            "quarter_start"
        ).tail(4)
        next_forecast = city_forecast_all.head(4)

        actual_labels = [charts.quarter_label(d) for d in prev_actual["quarter_start"]]
        forecast_labels = [charts.quarter_label(d) for d in next_forecast["forecast_quarter"]]
        forecast_incidence = (next_forecast["predicted_cases"] / population * 100_000).tolist()

        st.plotly_chart(
            charts.build_incidence_bar_chart(
                actual_labels, prev_actual["p_inc100k"].tolist(), forecast_labels, forecast_incidence,
            ),
            width="stretch",
        )
    else:
        st.info("Not enough data to build the incidence bar chart for this city yet.")

    # ── Percentage stats: vs historical same-quarter average, vs last year ──
    if population and focus_incidence is not None and not city_history.empty and focus_quarter_ts is not None:
        same_qoy = city_history[
            (city_history["quarter_start"].dt.quarter == focus_quarter_ts.quarter)
            & (city_history["quarter_start"] != focus_quarter_ts)
        ]
        avg_same_qoy_incidence = same_qoy["p_inc100k"].mean() if not same_qoy.empty else None

        last_year_row = city_history[city_history["quarter_start"] == focus_quarter_ts - pd.DateOffset(years=1)]
        last_year_incidence = last_year_row["p_inc100k"].iloc[0] if not last_year_row.empty else None

        pct_col1, pct_col2 = st.columns(2)
        with pct_col1:
            pct_vs_avg = (
                (focus_incidence - avg_same_qoy_incidence) / avg_same_qoy_incidence * 100
                if avg_same_qoy_incidence else None
            )
            st.metric(
                "Incidence vs. historical average for this quarter",
                f"{focus_incidence:,.1f} /100k",
                delta=f"{pct_vs_avg:+.0f}%" if pct_vs_avg is not None else None,
            )
        with pct_col2:
            pct_vs_last_year = (
                (focus_incidence - last_year_incidence) / last_year_incidence * 100
                if last_year_incidence else None
            )
            st.metric(
                "Incidence vs. same quarter last year",
                f"{focus_incidence:,.1f} /100k",
                delta=f"{pct_vs_last_year:+.0f}%" if pct_vs_last_year is not None else None,
            )

    # ── Seasonal profile (Q4->Q1->Q2->Q3), historical only ──────────────────
    st.subheader(f"{selected_city} — annual quarterly profile (historical seasons)")
    seasonal_df = api_client.get_seasonal_profile(selected_city)
    if not seasonal_df.empty:
        st.plotly_chart(charts.build_seasonal_profile_chart(seasonal_df), width="stretch")
    else:
        st.info("Not enough historical data to build the seasonal profile chart yet.")

    # ── Risk tier badge + decision-support recommendations (below the charts) ──
    recs = api_client.get_recommendations()[risk_tier]
    st.markdown(f"### {recs['emoji']} {recs['label']}\n*{recs['description']}*")
    rec_cols = st.columns(len(recs["recommendations"]))
    for col, (category, items) in zip(rec_cols, recs["recommendations"].items()):
        with col:
            st.markdown(f"**{category}**")
            for item in items:
                st.markdown(f"- {item}")
