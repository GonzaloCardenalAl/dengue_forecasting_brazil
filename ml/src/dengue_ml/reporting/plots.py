import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster/script use
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
from pathlib import Path

from dengue_ml.config import CITIES, CITY_COL, TARGET

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
    _savefig("final_forecast.png", fig, outputs_dir)


def plot_forecast_vs_previous_year(
    forecast_df: pd.DataFrame,
    historical_df: pd.DataFrame,
    n_lead_in_q: int = 2,
    outputs_dir: Path | None = None,
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
    _savefig("forecast_vs_previous_year.png", fig, outputs_dir)


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

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, comparison_table[metric], width, label=metric, color=colors[metric])
    ax.axhline(0.95, color="red", linestyle="--", linewidth=1, label="95% target (coverage only)")
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
