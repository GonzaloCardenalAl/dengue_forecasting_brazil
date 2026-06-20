import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster/script use
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
from pathlib import Path

from dengue_ml.config import CITIES, CITY_COL, TARGET
from dengue_ml.validation.conditional_residuals import (
    apply_horizon_bucketed_quantile_table, aggregate_oof_to_monthly, assign_loFo_conditional_ci,
    quantile_bounds, is_known_data_gap, PROXY_COL, REGIME_THRESHOLD,
)

CITY_COLORS = dict(zip(CITIES, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]))


def _resolve_fig_dir(outputs_dir: Path | None) -> Path:
    if outputs_dir is None:
        from dengue_ml.run_dir import get_latest_run_dir
        outputs_dir = get_latest_run_dir()
    return outputs_dir / "figures"


def _savefig(name: str, fig: plt.Figure, outputs_dir: Path | None = None) -> None:
    fig_dir = _resolve_fig_dir(outputs_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _savefig_into(dir_path: Path | str, name: str, fig: plt.Figure) -> None:
    """Like `_savefig`, but writes directly into `dir_path` instead of
    `dir_path / "figures"` -- for callers that already pass a fully-formed
    destination directory (e.g. run_dir / "figures" / "forecast" / grain)
    rather than the run_dir convention `_resolve_fig_dir` assumes."""
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(dir_path / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _log_floor(series: pd.Series, floor: float = 1.0) -> pd.Series:
    """
    Dengue case counts can be genuinely 0 (e.g. Vitória Q1 2021), which log
    scale can't represent — matplotlib drops/clips those points, making the
    line appear to plunge off the bottom of the chart. Floor at a small
    positive value for plotting only; does not touch the underlying data.
    """
    return series.clip(lower=floor)


def plot_historical_cases(df: pd.DataFrame, outputs_dir: Path | None = None) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    for ax, city in zip(axes, CITIES):
        sub = df[df[CITY_COL] == city].sort_values("week_start")
        ax.bar(sub["week_start"], sub[TARGET], width=5, color=CITY_COLORS[city], alpha=0.8)
        ax.set_title(city)
        ax.set_ylabel("Estimated cases")
    fig.suptitle("Historical dengue cases by city (weekly)", fontsize=13)
    fig.tight_layout()
    _savefig("historical_cases.png", fig, outputs_dir)


def plot_seasonality(df: pd.DataFrame, outputs_dir: Path | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for city in CITIES:
        sub = df[df[CITY_COL] == city].copy()
        iso_week = sub["week_start"].dt.isocalendar()["week"]
        seasonal = sub.groupby(iso_week)[TARGET].median()
        ax.plot(seasonal.index, seasonal.values, marker="o", label=city,
                color=CITY_COLORS[city], ms=3)
    ax.set_xticks(range(1, 54, 4))
    ax.set_xlabel("ISO week of year")
    ax.set_ylabel("Median estimated cases")
    ax.set_title("Seasonal pattern (median cases by ISO week)")
    ax.legend()
    _savefig("seasonality.png", fig, outputs_dir)


def plot_actual_vs_predicted(fold_predictions: pd.DataFrame, outputs_dir: Path | None = None) -> None:
    models = fold_predictions["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = fold_predictions[fold_predictions["model"] == model]
        ax.scatter(sub[TARGET], sub["predicted"], alpha=0.4, s=20)
        lim = max(sub[TARGET].max(), sub["predicted"].max()) * 1.05
        ax.plot([0, lim], [0, lim], "r--", lw=1)
        ax.set_title(model)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    fig.suptitle("Actual vs Predicted (all folds & cities)", fontsize=12)
    fig.tight_layout()
    _savefig("actual_vs_predicted.png", fig, outputs_dir)


def plot_model_comparison(
    fold_metrics: pd.DataFrame,
    outputs_dir: Path | None = None,
    log_scale: bool = False,
    baseline_model: str = "baseline",
) -> None:
    """
    Bars = median MAE across (fold, city); whiskers = IQR (q25-q75). Median is
    used rather than mean because the mean is dominated by a couple of outlier
    folds (the 2024 outbreak, ~7x any prior year) and gives a misleading ranking.
    Each non-baseline bar is annotated with its % improvement in median MAE
    over the seasonal-naive baseline.
    """
    summary = (
        fold_metrics.groupby("model")["mae"]
        .agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
        .sort_values("median")
    )
    err_lower = summary["median"] - summary["q25"]
    err_upper = summary["q75"] - summary["median"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(summary))
    ax.bar(x, summary["median"], yerr=[err_lower, err_upper], capsize=4, alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary.index, rotation=20, ha="right")
    ax.set_ylabel("MAE (cases)")

    if baseline_model in summary.index:
        baseline_mae = summary.loc[baseline_model, "median"]
        for i, model in enumerate(summary.index):
            if model == baseline_model:
                continue
            pct = (baseline_mae - summary.loc[model, "median"]) / baseline_mae * 100
            ax.annotate(
                f"{pct:+.0f}% vs baseline",
                xy=(i, summary.loc[model, "median"] + err_upper.iloc[i]),
                xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8,
                color="darkgreen" if pct > 0 else "darkred",
            )

    title_suffix = " (log scale)" if log_scale else ""
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(f"Model comparison — median MAE (IQR) across folds & cities{title_suffix}")
    fig.tight_layout()
    fname = "model_comparison_log.png" if log_scale else "model_comparison.png"
    _savefig(fname, fig, outputs_dir)


def plot_final_forecast(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    oof_quarterly_df: pd.DataFrame | None = None,
    n_historical_q: int = 12,
    outputs_dir: Path | None = None,
    filename: str = "final_forecast.png",
) -> None:
    """
    historical_df should be quarterly-aggregated (e.g. via
    quarterly_aggregation.aggregate_weekly_history_to_quarterly on top of
    prepare_model_table(apply_reliability_cutoff=False)) so the most recent
    (still-converging) quarter — excluded from training — is included here
    and shows its genuinely wide casos_est_min/casos_est_max band. Older,
    converged quarters have a near-zero-width band, which is expected.

    oof_quarterly_df (optional, e.g. via quarterly_aggregation.
    aggregate_weekly_oof_predictions_to_quarterly for the final model) adds
    the model's own historical out-of-fold predictions as a red dashed line
    leading into the solid-red forecast line, so the "model prediction"
    line reads as one continuous (past + future) series, distinct from the
    blue actual-cases line.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for ax, city in zip(axes, CITIES):
        hist = (
            historical_df[historical_df[CITY_COL] == city]
            .sort_values("quarter_start")
            .tail(n_historical_q)
        )
        fcast = forecast_df[forecast_df["city"] == city].sort_values("forecast_quarter")

        if {"casos_est_min", "casos_est_max"}.issubset(hist.columns):
            ci_lower = hist[["casos_est_min", TARGET]].min(axis=1).clip(lower=0)
            ci_upper = hist[["casos_est_max", TARGET]].max(axis=1)
            ax.fill_between(
                hist["quarter_start"], _log_floor(ci_lower), _log_floor(ci_upper),
                alpha=0.2, color="blue", label="Historical 95% CI (estimate)",
            )
            # The band can be only a few pixels wide on the full multi-year,
            # log-scale view even when real, so spell out the last point's
            # bounds in text rather than relying on it being visible.
            last_width = ci_upper.iloc[-1] - ci_lower.iloc[-1]
            if last_width > 0:
                ax.annotate(
                    f"95% CI: {ci_lower.iloc[-1]:,.0f}–{ci_upper.iloc[-1]:,.0f}",
                    xy=(hist["quarter_start"].iloc[-1], ci_upper.iloc[-1]),
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=7, color="blue", ha="right",
                )
        ax.plot(hist["quarter_start"], _log_floor(hist[TARGET]), "b-o", ms=4, label="Historical (actual)")

        if oof_quarterly_df is not None:
            oof = (
                oof_quarterly_df[oof_quarterly_df[CITY_COL] == city]
                .sort_values("quarter_start")
            )
            oof = oof[oof["quarter_start"] >= hist["quarter_start"].min()]
            if not oof.empty:
                ax.plot(
                    oof["quarter_start"], _log_floor(oof["predicted_cases"]),
                    "r--o", ms=3, alpha=0.7, label="Model prediction (past, OOF)",
                )

        has_ci = fcast["lower_95"].notna().any()
        if has_ci:
            ax.fill_between(
                fcast["forecast_quarter"],
                _log_floor(fcast["lower_95"]), _log_floor(fcast["upper_95"]),
                alpha=0.2, color="red", label="Forecast 95% CI",
            )
        ax.plot(fcast["forecast_quarter"], _log_floor(fcast["predicted_cases"]), "r-o", ms=4, label="Forecast")

        # Log scale: these series span ~2 orders of magnitude (calm quarters vs
        # the 2024 outbreak), which flattens recent/forecast CI bands to a few
        # pixels on a linear axis even when their width is real and meaningful.
        ax.set_yscale("log")
        ax.set_title(city)
        ax.set_ylabel("Estimated cases (log scale)")
        ax.legend(fontsize=8)
    fig.suptitle("4-quarter dengue forecast by city", fontsize=13)
    fig.tight_layout()
    _savefig(filename, fig, outputs_dir)


def plot_forecast_vs_previous_year(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    n_lead_in_q: int = 2,
    outputs_dir: Path | None = None,
    filename: str = "forecast_vs_previous_year.png",
) -> None:
    """
    Zoomed-in companion to plot_final_forecast: just the forecast year plus
    `n_lead_in_q` quarters of actual history leading into it, with the same
    quarters from exactly one year earlier overlaid on the same x-axis
    positions (shifted forward a year) -- so the new forecast can be read
    directly against what actually happened in the equivalent quarters last
    year, rather than against a multi-year, log-scale view where a one-year
    shift is hard to judge by eye.

    historical_df: full quarterly history (not pre-truncated -- the
    previous-year lookup needs quarters older than `n_lead_in_q`).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for ax, city in zip(axes, CITIES):
        hist_all = (
            historical_df[historical_df[CITY_COL] == city]
            .sort_values("quarter_start")
        )
        hist_recent = hist_all.tail(n_lead_in_q)
        fcast = forecast_df[forecast_df["city"] == city].sort_values("forecast_quarter")
        n_hist = len(hist_recent)

        quarters = list(hist_recent["quarter_start"]) + list(fcast["forecast_quarter"])
        x = list(range(len(quarters)))
        labels = [f"{q.year} Q{q.quarter}" for q in quarters]

        hist_lookup = hist_all.set_index("quarter_start")[TARGET]
        prev_quarters = [q - pd.DateOffset(years=1) for q in quarters]
        prev_cycle = [hist_lookup.get(q, np.nan) for q in prev_quarters]

        has_ci = {"lower_95", "upper_95"}.issubset(fcast.columns) and fcast["lower_95"].notna().any()
        if has_ci:
            ax.fill_between(
                x[n_hist:], _log_floor(fcast["lower_95"]), _log_floor(fcast["upper_95"]),
                alpha=0.2, color="red", label="Forecast 95% CI",
            )

        ax.plot(x[:n_hist], _log_floor(hist_recent[TARGET]), "b-o", ms=5, label="Actual (recent)")
        ax.plot(x[n_hist:], _log_floor(fcast["predicted_cases"]), "r-o", ms=5, label="Forecast")
        ax.plot(
            x, _log_floor(pd.Series(prev_cycle)), color="gray", linestyle="--",
            marker="o", ms=4, label="Same quarters, previous year (actual)",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_title(city)
        ax.set_ylabel("Estimated cases (log scale)")
        ax.legend(fontsize=8)
    fig.suptitle("Forecast year vs. same quarters last year", fontsize=13)
    fig.tight_layout()
    _savefig(filename, fig, outputs_dir)


def plot_forecast_year_over_year(
    period_labels: list[str],
    prev_actual: pd.DataFrame,
    prev_oof: pd.DataFrame,
    forecast: pd.DataFrame,
    prev_year: int,
    forecast_year: int,
    outputs_dir: Path | str,
    filename: str = "forecast_year_over_year.png",
    reveal_n: int | None = None,
    ylim_by_city: dict | None = None,
) -> None:
    """
    Year-over-year companion to plot_final_forecast: a single period-of-year
    x-axis (`period_labels`, e.g. ["Q1".."Q4"] / ["Jan".."Dec"] / ["W1".."W52"])
    overlaying three series instead of the old concatenated-timeline +
    gray-dashed-previous-year layout:

    - `prev_actual` (city_name, x_pos, value): real reported cases for
      `prev_year`, blue dotted, no CI.
    - `prev_oof` (city_name, x_pos, value, lower_95, upper_95): the model's
      own leave-one-fold-out conditional prediction for `prev_year` (see
      assign_loFo_conditional_ci / aggregate_oof_to_{monthly,quarterly}),
      red dotted with its own shaded 97.5% CI -- lets the new forecast's
      uncertainty be read directly against the model's known historical
      accuracy.
    - `forecast` (city_name, x_pos, value, lower_95, upper_95): the
      `forecast_year` forecast, green solid with its own shaded
      (horizon-aware where available) 97.5% CI.

    `reveal_n`, if given, drops `forecast` rows with x_pos > reveal_n -- used
    by plot_forecast_year_over_year_frames to build a GIF that reveals the
    forecast one point at a time. `prev_actual` and `prev_oof` are always
    shown in full since they're both already-known data (real cases and the
    model's own historical estimate for them) -- only the new forecast is
    something being revealed. `ylim_by_city`, if given, fixes each subplot's
    y-axis to a city's pre-computed (min, max) so frames don't rescale as
    more points appear.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for ax, city in zip(axes, CITIES):
        pa = prev_actual[prev_actual[CITY_COL] == city].sort_values("x_pos")
        po = prev_oof[prev_oof[CITY_COL] == city].sort_values("x_pos")
        fc = forecast[forecast[CITY_COL] == city].sort_values("x_pos")
        if reveal_n is not None:
            fc = fc[fc["x_pos"] <= reveal_n]

        ax.plot(
            pa["x_pos"], _log_floor(pa["value"]), color="blue", linestyle=":",
            marker="o", ms=4, label=f"{prev_year} Historical Data",
        )

        has_oof_ci = {"lower_95", "upper_95"}.issubset(po.columns) and po["lower_95"].notna().any()
        if has_oof_ci:
            ax.fill_between(
                po["x_pos"], _log_floor(po["lower_95"]), _log_floor(po["upper_95"]),
                alpha=0.2, color="red",
            )
        ax.plot(
            po["x_pos"], _log_floor(po["value"]), color="red", linestyle=":",
            marker="o", ms=4, label=f"{prev_year} Estimated Cases",
        )

        has_fc_ci = {"lower_95", "upper_95"}.issubset(fc.columns) and fc["lower_95"].notna().any()
        if has_fc_ci:
            ax.fill_between(
                fc["x_pos"], _log_floor(fc["lower_95"]), _log_floor(fc["upper_95"]),
                alpha=0.2, color="green",
            )
        ax.plot(
            fc["x_pos"], _log_floor(fc["value"]), color="green", linestyle="-",
            marker="o", ms=4, label=f"{forecast_year} Forecast",
        )

        ax.set_xticks(range(1, len(period_labels) + 1))
        ax.set_xticklabels(period_labels)
        ax.set_xlim(0.5, len(period_labels) + 0.5)
        if ylim_by_city is not None and city in ylim_by_city:
            ax.set_ylim(*ylim_by_city[city])
        ax.set_yscale("log")
        ax.set_title(city)
        ax.set_ylabel("Estimated cases (log scale)")
        ax.legend(fontsize=8)
    fig.suptitle(f"{forecast_year} forecast vs. {prev_year}", fontsize=13)
    fig.tight_layout()
    _savefig_into(outputs_dir, filename, fig)


def _city_ylim(
    prev_actual: pd.DataFrame, prev_oof: pd.DataFrame, forecast: pd.DataFrame, city: str,
) -> tuple[float, float]:
    cols = ["lower_95", "upper_95", "value"]
    vals = pd.concat([
        prev_actual.loc[prev_actual[CITY_COL] == city, "value"],
        *[prev_oof.loc[prev_oof[CITY_COL] == city, c] for c in cols if c in prev_oof.columns],
        *[forecast.loc[forecast[CITY_COL] == city, c] for c in cols if c in forecast.columns],
    ], ignore_index=True).dropna()
    vals = _log_floor(vals)
    return float(vals.min()) * 0.8, float(vals.max()) * 1.25


def plot_forecast_year_over_year_frames(
    period_labels: list[str],
    prev_actual: pd.DataFrame,
    prev_oof: pd.DataFrame,
    forecast: pd.DataFrame,
    prev_year: int,
    forecast_year: int,
    outputs_dir: Path | str,
    frame_duration_ms: int = 600,
) -> None:
    """
    One frame per forecast period (x_pos = 1..len(period_labels)), each
    revealing one more point of `forecast` than the last (see
    plot_forecast_year_over_year's `reveal_n`); `prev_actual`/`prev_oof` are
    static across every frame. Frames are then assembled into forecast.gif
    via Pillow. Y-axis is fixed per city across every frame (see
    `_city_ylim`) so the GIF doesn't rescale as points are added.
    """
    from PIL import Image

    fig_dir = Path(outputs_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ylim_by_city = {city: _city_ylim(prev_actual, prev_oof, forecast, city) for city in CITIES}

    frame_paths = []
    for i in range(1, len(period_labels) + 1):
        fname = f"frame_{i:02d}.png"
        plot_forecast_year_over_year(
            period_labels, prev_actual, prev_oof, forecast, prev_year, forecast_year,
            outputs_dir=outputs_dir, filename=fname, reveal_n=i, ylim_by_city=ylim_by_city,
        )
        frame_paths.append(fig_dir / fname)

    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        fig_dir / "forecast.gif", save_all=True, append_images=frames[1:],
        duration=frame_duration_ms, loop=0,
    )


def plot_validation_rollout(
    city: str,
    actual: pd.DataFrame,
    lead_in_oof: pd.DataFrame,
    predicted: pd.DataFrame,
    date_col: str,
    lead_in_year: int,
    validation_year: int,
    outputs_dir: Path | str,
    filename: str = "validation_rollout.png",
    reveal_n: int | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Single-city, continuous-timeline validation figure: `validation_year` is
    a year whose true outcome is already known (e.g. the most recent outer
    CV fold's test year), so the same autoregressive rollout used in
    production can be checked directly against reality as its horizon (and
    calibrated CI) widens. Unlike plot_forecast_year_over_year's
    period-of-year overlay, the x-axis here is a real two-year timeline
    (`lead_in_year` followed by `validation_year`) so it reads as one
    chronological series, not two years stacked on the same 12 ticks.

    - `actual` (`date_col`, value): real cases spanning both `lead_in_year`
      and `validation_year`, black solid, no CI -- always shown in full
      since the point is to compare the rollout against a known answer.
    - `lead_in_oof` (`date_col`, value): the model's ordinary one-step
      (non-autoregressive, real-lag) out-of-fold prediction for
      `lead_in_year` only -- shows how the model performs when it isn't
      compounding its own errors, as contrast against the rollout.
    - `predicted` (`date_col`, value, lower_95, upper_95, x_pos): the
      autoregressive rollout's own prediction for `validation_year` only
      (see validation/autoregressive_cv.run_autoregressive_cv), red dashed
      with its horizon-bucketed CI -- the only series `reveal_n` truncates
      (via `x_pos`, the rollout's own 1..N step count).
    """
    fig, ax = plt.subplots(figsize=(11, 5))

    pr = predicted if reveal_n is None else predicted[predicted["x_pos"] <= reveal_n]

    ax.plot(
        actual[date_col], _log_floor(actual["value"]), color="black", linestyle="-",
        marker="o", ms=3, label=f"Actual ({lead_in_year}–{validation_year})",
    )
    ax.plot(
        lead_in_oof[date_col], _log_floor(lead_in_oof["value"]), color="blue", linestyle="--",
        marker="o", ms=3, label=f"{lead_in_year} OOF prediction (non-autoregressive)",
    )

    has_ci = {"lower_95", "upper_95"}.issubset(pr.columns) and pr["lower_95"].notna().any()
    if has_ci:
        ax.fill_between(
            pr[date_col], _log_floor(pr["lower_95"]), _log_floor(pr["upper_95"]),
            alpha=0.2, color="red", label="95% CI (autoregressive)",
        )
    ax.plot(
        pr[date_col], _log_floor(pr["value"]), color="red", linestyle="--",
        marker="o", ms=4, label=f"{validation_year} Predicted (autoregressive)",
    )

    ax.axvline(pd.Timestamp(f"{validation_year}-01-01"), color="gray", linestyle=":", linewidth=1)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_yscale("log")
    ax.set_title(f"{city} — {validation_year} autoregressive rollout vs. true values", fontsize=12)
    ax.set_ylabel("Estimated cases (log scale)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _savefig_into(outputs_dir, filename, fig)


def plot_validation_rollout_frames(
    city: str,
    actual: pd.DataFrame,
    lead_in_oof: pd.DataFrame,
    predicted: pd.DataFrame,
    date_col: str,
    lead_in_year: int,
    validation_year: int,
    outputs_dir: Path | str,
    frame_duration_ms: int = 600,
) -> None:
    """
    GIF twin of plot_validation_rollout: one frame per rollout step
    (`predicted`'s `x_pos`), each revealing one more point than the last;
    `actual`/`lead_in_oof` are static across every frame. Y-axis is fixed
    across every frame (same min/max as plot_forecast_year_over_year_frames'
    `_city_ylim`, just for this single city/series set) so the GIF doesn't
    rescale as points are added.
    """
    from PIL import Image

    fig_dir = Path(outputs_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cols = ["value", "lower_95", "upper_95"]
    vals = pd.concat(
        [actual["value"], lead_in_oof["value"], *[predicted[c] for c in cols if c in predicted.columns]],
        ignore_index=True,
    ).dropna()
    vals = _log_floor(vals)
    ylim = (float(vals.min()) * 0.8, float(vals.max()) * 1.25)

    n_steps = int(predicted["x_pos"].max())
    frame_paths = []
    for i in range(1, n_steps + 1):
        fname = f"frame_{i:02d}.png"
        plot_validation_rollout(
            city, actual, lead_in_oof, predicted, date_col, lead_in_year, validation_year,
            outputs_dir=outputs_dir, filename=fname, reveal_n=i, ylim=ylim,
        )
        frame_paths.append(fig_dir / fname)

    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        fig_dir / "forecast.gif", save_all=True, append_images=frames[1:],
        duration=frame_duration_ms, loop=0,
    )


def plot_validation_rollout_grid(
    cities: list[str],
    actual: pd.DataFrame,
    lead_in_oof: pd.DataFrame,
    predicted: pd.DataFrame,
    date_col: str,
    lead_in_year: int,
    validation_year: int,
    outputs_dir: Path | str,
    filename: str = "validation_rollout_grid.png",
    reveal_n: int | None = None,
    ylim_by_city: dict | None = None,
) -> None:
    """
    All-cities twin of plot_validation_rollout: same continuous-timeline,
    three-series-per-panel design, one panel per city in a 2x2 grid instead
    of a single standalone figure. `actual`/`lead_in_oof`/`predicted` carry
    `CITY_COL` so each panel can filter down to its own city.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for ax, city in zip(axes, cities):
        a = actual[actual[CITY_COL] == city].sort_values(date_col)
        lo = lead_in_oof[lead_in_oof[CITY_COL] == city].sort_values(date_col)
        pr = predicted[predicted[CITY_COL] == city].sort_values(date_col)
        if reveal_n is not None:
            pr = pr[pr["x_pos"] <= reveal_n]

        ax.plot(
            a[date_col], _log_floor(a["value"]), color="black", linestyle="-",
            marker="o", ms=3, label=f"Actual ({lead_in_year}–{validation_year})",
        )
        ax.plot(
            lo[date_col], _log_floor(lo["value"]), color="blue", linestyle="--",
            marker="o", ms=3, label=f"{lead_in_year} OOF prediction (non-autoregressive)",
        )

        has_ci = {"lower_95", "upper_95"}.issubset(pr.columns) and pr["lower_95"].notna().any()
        if has_ci:
            ax.fill_between(
                pr[date_col], _log_floor(pr["lower_95"]), _log_floor(pr["upper_95"]),
                alpha=0.2, color="red", label="95% CI (autoregressive)",
            )
        ax.plot(
            pr[date_col], _log_floor(pr["value"]), color="red", linestyle="--",
            marker="o", ms=4, label=f"{validation_year} Predicted (autoregressive)",
        )

        ax.axvline(pd.Timestamp(f"{validation_year}-01-01"), color="gray", linestyle=":", linewidth=1)
        if ylim_by_city is not None and city in ylim_by_city:
            ax.set_ylim(*ylim_by_city[city])
        ax.set_yscale("log")
        ax.set_title(city)
        ax.set_ylabel("Estimated cases (log scale)")
        ax.legend(fontsize=7)
    fig.suptitle(
        f"{validation_year} autoregressive rollout vs. true values (lead-in {lead_in_year})",
        fontsize=13,
    )
    fig.tight_layout()
    _savefig_into(outputs_dir, filename, fig)


def plot_validation_rollout_grid_frames(
    cities: list[str],
    actual: pd.DataFrame,
    lead_in_oof: pd.DataFrame,
    predicted: pd.DataFrame,
    date_col: str,
    lead_in_year: int,
    validation_year: int,
    outputs_dir: Path | str,
    frame_duration_ms: int = 600,
) -> None:
    """
    GIF twin of plot_validation_rollout_grid: same per-city fixed y-axis
    (`_city_ylim`) and Pillow assembly as plot_forecast_year_over_year_frames.
    """
    from PIL import Image

    fig_dir = Path(outputs_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    ylim_by_city = {city: _city_ylim(actual, lead_in_oof, predicted, city) for city in cities}

    n_steps = int(predicted["x_pos"].max())
    frame_paths = []
    for i in range(1, n_steps + 1):
        fname = f"frame_{i:02d}.png"
        plot_validation_rollout_grid(
            cities, actual, lead_in_oof, predicted, date_col, lead_in_year, validation_year,
            outputs_dir=outputs_dir, filename=fname, reveal_n=i, ylim_by_city=ylim_by_city,
        )
        frame_paths.append(fig_dir / fname)

    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        fig_dir / "forecast.gif", save_all=True, append_images=frames[1:],
        duration=frame_duration_ms, loop=0,
    )


def plot_horizon_widening_example(
    fold_predictions_ar: pd.DataFrame,
    model_name: str,
    horizon_quantiles: dict,
    fold_label: int,
    year_label: str = "2023",
    outputs_dir: Path | None = None,
) -> None:
    """
    Backtest figure for one concrete historical outer fold: aggregates that
    fold's autoregressive-rollout weekly predictions
    (validation/autoregressive_cv.run_autoregressive_cv) to quarterly sums,
    applies the quarter-position-bucketed calibration
    (conditional_residuals.compute_horizon_bucketed_quarterly_residual_quantile_table)
    to those same predictions, and plots the result against known actuals --
    demonstrating the CI band actually widening from Q1 to Q4 of the
    forecast horizon. This is the same mechanism behind the parallel
    horizon-aware production deliverable in generate_forecasts.py, just
    shown against a year whose outcome is already known.
    """
    sub = fold_predictions_ar[
        (fold_predictions_ar["fold"] == fold_label) & (fold_predictions_ar["model"] == model_name)
    ].copy()

    grouped = sub.groupby([CITY_COL, "quarter_position"]).agg(
        actual_sum=(TARGET, "sum"),
        predicted_sum=("predicted", "sum"),
        growth_proxy=("growth_proxy", "first"),
    ).reset_index()

    lower, upper = apply_horizon_bucketed_quantile_table(
        grouped["predicted_sum"].values, grouped["growth_proxy"].values,
        grouped["quarter_position"].values, horizon_quantiles,
    )
    grouped["lower_95"] = lower
    grouped["upper_95"] = upper

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    for ax, city in zip(axes, CITIES):
        city_df = grouped[grouped[CITY_COL] == city].sort_values("quarter_position")
        x = city_df["quarter_position"].values
        labels = [f"{year_label} Q{q}" for q in x]

        ax.fill_between(
            x, _log_floor(city_df["lower_95"]), _log_floor(city_df["upper_95"]),
            alpha=0.2, color="red", label="Horizon-aware 95% CI",
        )
        ax.plot(x, _log_floor(city_df["actual_sum"]), "b-o", ms=5, label="Actual")
        ax.plot(x, _log_floor(city_df["predicted_sum"]), "r-o", ms=5, label="Autoregressive rollout")

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_yscale("log")
        ax.set_title(city)
        ax.set_ylabel("Estimated cases (log scale)")
        ax.legend(fontsize=8)
    fig.suptitle(
        f"Horizon-widening backtest — {model_name}, fold {fold_label} ({year_label})", fontsize=13,
    )
    fig.tight_layout()
    _savefig(f"horizon_widening_example_{year_label}.png", fig, outputs_dir)


def plot_oof_predictions(
    fold_predictions: pd.DataFrame,
    model_name: str,
    outputs_dir: Path | None = None,
    log_scale: bool = True,
) -> None:
    """
    Out-of-fold predictions vs actual, concatenated across all outer test folds,
    per city. Since outer folds are non-overlapping in time, this forms one
    continuous chronological series covering the full CV evaluation window.
    """
    sub = fold_predictions[fold_predictions["model"] == model_name].copy()
    sub["week_start"] = pd.to_datetime(sub["week_start"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes = axes.flatten()
    for ax, city in zip(axes, CITIES):
        city_df = sub[sub[CITY_COL] == city].sort_values("week_start")

        has_ci = (
            {"lower_95", "upper_95"}.issubset(city_df.columns)
            and city_df["lower_95"].notna().any()
        )
        actual    = city_df[TARGET]
        predicted = city_df["predicted"]
        lower     = city_df["lower_95"] if has_ci else None
        if log_scale:
            actual    = _log_floor(actual)
            predicted = _log_floor(predicted)
            if has_ci:
                lower = _log_floor(lower)

        if has_ci:
            ax.fill_between(
                city_df["week_start"], lower, city_df["upper_95"],
                alpha=0.2, color="red", label="OOF prediction 95% CI",
            )
        ax.plot(city_df["week_start"], actual, "b-o", ms=4, label="Actual")
        ax.plot(city_df["week_start"], predicted, "r--o", ms=4, label="OOF predicted")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(city)
        ax.set_ylabel("Estimated cases" + (" (log scale)" if log_scale else ""))
        ax.legend(fontsize=8)
    fig.suptitle(f"Out-of-fold predictions vs actual — {model_name}", fontsize=13)
    fig.tight_layout()
    suffix = "_log" if log_scale else ""
    _savefig(f"oof_predictions_{model_name}{suffix}.png", fig, outputs_dir)


def plot_oof_predictions_monthly(
    fold_predictions: pd.DataFrame,
    model_name: str,
    outputs_dir: Path | None = None,
    log_scale: bool = True,
) -> None:
    """
    Monthly-aggregated counterpart of plot_oof_predictions. Weekly OOF
    residuals are dominated by a handful of near-zero-count weeks (a swing
    from 2 to 20 cases is a huge log1p move but a tiny absolute one), which
    drags the regime-conditional quantile band wide for every week sharing
    that regime -- summing to monthly before computing the LOFO band removes
    most of that near-zero noise (see aggregate_oof_to_monthly), the same way
    compute_quarterly_residual_quantile_table already does for the forecast
    deliverable.
    """
    monthly = aggregate_oof_to_monthly(fold_predictions, model_name)
    monthly = assign_loFo_conditional_ci(monthly, model_name)
    lower_q, upper_q = quantile_bounds()
    ci_label = f"OOF prediction {(upper_q - lower_q) * 100:g}% CI"

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes = axes.flatten()
    for ax, city in zip(axes, CITIES):
        city_df = monthly[monthly[CITY_COL] == city].sort_values("month_start")

        has_ci = (
            {"lower_95", "upper_95"}.issubset(city_df.columns)
            and city_df["lower_95"].notna().any()
        )
        actual    = city_df[TARGET]
        predicted = city_df["predicted"]
        lower     = city_df["lower_95"] if has_ci else None
        if log_scale:
            actual    = _log_floor(actual)
            predicted = _log_floor(predicted)
            if has_ci:
                lower = _log_floor(lower)

        if has_ci:
            ax.fill_between(
                city_df["month_start"], lower, city_df["upper_95"],
                alpha=0.2, color="red", label=ci_label,
            )
        ax.plot(city_df["month_start"], actual, "b-o", ms=4, label="Actual")
        ax.plot(city_df["month_start"], predicted, "r--o", ms=4, label="OOF predicted")
        if log_scale:
            ax.set_yscale("log")
        ax.set_title(city)
        ax.set_ylabel("Estimated cases" + (" (log scale)" if log_scale else ""))
        ax.legend(fontsize=8)
    fig.suptitle(f"Out-of-fold predictions vs actual (monthly) — {model_name}", fontsize=13)
    fig.tight_layout()
    suffix = "_log" if log_scale else ""
    _savefig(f"oof_predictions_{model_name}_monthly{suffix}.png", fig, outputs_dir)


def plot_proxy_comparison(comparison_table: pd.DataFrame, outputs_dir: Path | None = None) -> None:
    """
    Bar chart of precision/recall/f1/coverage across CI-regime-proxy candidates
    (nivel_inc rule, sustained_rt rule, trained classifiers) -- replaces the
    role of the original ad-hoc notebook figure (01f_proxy_precision_recall_
    coverage.png), but built from real pipeline numbers on the fair
    weekly-grain population (see results_tables.proxy_comparison_table)
    instead of the notebook's original weekly-onset-detection code.
    """
    metrics = ["precision", "recall", "f1", "coverage"]
    x = np.arange(len(comparison_table))
    width = 0.2
    colors = {"precision": "#d95f02", "recall": "#7570b3", "f1": "#e7298a", "coverage": "#1b9e77"}
    lower_q, upper_q = quantile_bounds()
    ci_pct = (upper_q - lower_q) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, comparison_table[metric], width, label=metric, color=colors[metric])
    ax.axhline(ci_pct / 100, color="red", linestyle="--", linewidth=1, label=f"{ci_pct:g}% target (coverage only)")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{c}\n(n_high={n:.0f})" for c, n in zip(comparison_table["candidate"], comparison_table["n_high_regime"])],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("CI-regime proxy comparison: rules vs. trained classifier (fair, weekly grain)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig("proxy_comparison.png", fig, outputs_dir)


def plot_coverage_by_gap(coverage_table: pd.DataFrame, outputs_dir: Path | None = None) -> None:
    """
    Bar chart of coverage_by_gap_table's per-city + overall coverage, with
    vs. without KNOWN_DATA_GAPS rows included in the evaluation -- shows how
    much a known data-collection outage (e.g. Vitória's 2021 reporting gap)
    still drags down the headline number for the city it happened in, while
    every other (unaffected) city's bars are identical either way.
    """
    lower_q, upper_q = quantile_bounds()
    ci_pct = (upper_q - lower_q) * 100

    x = np.arange(len(coverage_table))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, coverage_table["coverage_including_gap"], width,
           label="Including known data gap", color="#d62728", alpha=0.8)
    ax.bar(x + width / 2, coverage_table["coverage_excluding_gap"], width,
           label="Excluding known data gap", color="#1f77b4", alpha=0.8)
    ax.axhline(ci_pct / 100, color="gray", linestyle="--", linewidth=1, label=f"Nominal target ({ci_pct:g}%)")
    ax.set_xticks(x)
    ax.set_xticklabels(coverage_table[CITY_COL])
    ax.set_ylabel(f"OOF {ci_pct:g}% CI coverage")
    ax.set_ylim(0.6, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(f"Monthly OOF coverage at {ci_pct:g}% CI: including vs. excluding known data gaps")
    for i, row in coverage_table.iterrows():
        ax.annotate(f"{row['coverage_including_gap']:.1%}", (x[i] - width / 2, row["coverage_including_gap"] + 0.01),
                    ha="center", fontsize=8)
        ax.annotate(f"{row['coverage_excluding_gap']:.1%}", (x[i] + width / 2, row["coverage_excluding_gap"] + 0.01),
                    ha="center", fontsize=8)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _savefig("coverage_by_gap.png", fig, outputs_dir)


def plot_residual_distribution(
    fold_predictions: pd.DataFrame,
    model_name: str,
    outputs_dir: Path | None = None,
) -> None:
    """
    Histogram of log1p residuals (log1p(actual) - log1p(predicted)), split by
    CI regime (growth_proxy >= REGIME_THRESHOLD), with the quantile_bounds()
    cutoffs actually used to build that regime's CI band overlaid as dashed
    vertical lines -- shows directly *why* each regime's band is the width it
    is (e.g. the high-regime distribution is visibly tighter than low-regime),
    rather than just reporting a width number. Pooled across cities, weekly
    grain (the grain compute_residual_quantile_table calibrates the
    production forecast band on). Rows inside a KNOWN_DATA_GAPS window are
    excluded -- they're a reporting outage, not real forecasting error.
    """
    sub = fold_predictions[fold_predictions["model"] == model_name].copy()
    sub = sub[~is_known_data_gap(sub)]
    log_resid = np.log1p(sub[TARGET]) - np.log1p(sub["predicted"])
    is_high = sub[PROXY_COL] >= REGIME_THRESHOLD
    lower_q, upper_q = quantile_bounds()
    ci_pct = (upper_q - lower_q) * 100

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(log_resid.min(), log_resid.max(), 60)
    for mask, color, label in [
        (~is_high, "#1f77b4", f"Low regime (n={(~is_high).sum()})"),
        (is_high,  "#d62728", f"High regime (n={is_high.sum()})"),
    ]:
        ax.hist(log_resid[mask], bins=bins, alpha=0.5, color=color, label=label, density=True)
        ax.axvline(log_resid[mask].quantile(lower_q), color=color, linestyle="--", linewidth=1.5)
        ax.axvline(log_resid[mask].quantile(upper_q), color=color, linestyle="--", linewidth=1.5)

    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("log1p residual: log1p(actual) − log1p(predicted)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Residual distribution by CI regime — {model_name}\n"
        f"(dashed lines = {ci_pct:g}% CI cutoffs per regime; known data-gap rows excluded)"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    _savefig(f"residual_distribution_{model_name}.png", fig, outputs_dir)


def plot_feature_importance(
    model,
    feature_names: list[str],
    top_n: int = 20,
    outputs_dir: Path | None = None,
    model_label: str = "XGBoost",
) -> None:
    scores = model.feature_importances_
    top_n = min(top_n, len(scores))
    idx = np.argsort(scores)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(top_n), scores[idx][::-1], align="center")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx[::-1]])
    ax.set_xlabel("Feature importance (gain)" if model_label == "XGBoost" else "Feature importance (AGOP diagonal, normalized)")
    ax.set_title(f"Top {top_n} {model_label} feature importances")
    fig.tight_layout()
    _savefig("feature_importance.png", fig, outputs_dir)
