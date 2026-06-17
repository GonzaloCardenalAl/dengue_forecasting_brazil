import pandas as pd
from pathlib import Path


def _resolve_outputs_dir(outputs_dir: Path | None) -> Path:
    if outputs_dir is None:
        from dengue_ml.run_dir import get_latest_run_dir
        return get_latest_run_dir()
    return outputs_dir


def model_comparison_table(
    fold_metrics: pd.DataFrame,
    outputs_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Mean/std are dominated by a couple of outlier folds (e.g. the 2024 outbreak,
    ~7x any prior year) — median/IQR give a fairer per-fold comparison and are
    sorted on by default.
    """
    outputs_dir = _resolve_outputs_dir(outputs_dir)
    summary = (
        fold_metrics.groupby("model")[["mae", "rmse", "mape"]]
        .agg(["mean", "std", "median",
              lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
        .round(1)
    )
    summary.columns = [
        "_".join(c).replace("<lambda_0>", "q25").replace("<lambda_1>", "q75")
        for c in summary.columns
    ]
    summary = summary.sort_values("mae_median")
    summary.to_csv(outputs_dir / "model_comparison.csv")
    return summary


def baseline_improvement_table(
    fold_metrics: pd.DataFrame,
    outputs_dir: Path | None = None,
    baseline_model: str = "baseline",
) -> pd.DataFrame:
    """
    % improvement in median fold MAE of each model over the seasonal-naive
    baseline. Median is used for the same reason as model_comparison_table:
    the mean is dominated by outlier folds (2024 outbreak).
    """
    outputs_dir = _resolve_outputs_dir(outputs_dir)
    median_mae = fold_metrics.groupby("model")["mae"].median()
    baseline_mae = median_mae.loc[baseline_model]
    table = (
        pd.DataFrame({
            "model": median_mae.index,
            "median_mae": median_mae.values,
            "pct_improvement_vs_baseline": (baseline_mae - median_mae.values) / baseline_mae * 100,
        })
        .sort_values("median_mae")
        .round(1)
        .reset_index(drop=True)
    )
    table.to_csv(outputs_dir / "baseline_improvement.csv", index=False)
    return table


def metrics_by_fold(
    fold_metrics: pd.DataFrame,
    outputs_dir: Path | None = None,
) -> pd.DataFrame:
    outputs_dir = _resolve_outputs_dir(outputs_dir)
    table = fold_metrics.groupby(["fold", "model"])[["mae", "rmse", "mape"]].mean().round(1)
    table.to_csv(outputs_dir / "metrics_by_fold.csv")
    return table


def metrics_by_city(
    fold_predictions: pd.DataFrame,
    outputs_dir: Path | None = None,
) -> pd.DataFrame:
    outputs_dir = _resolve_outputs_dir(outputs_dir)
    from dengue_ml.validation.metrics import calculate_all_metrics
    rows = []
    for (model, city), grp in fold_predictions.groupby(["model", "city_name"]):
        m = calculate_all_metrics(grp["casos_est"].values, grp["predicted"].values)
        rows.append({"model": model, "city": city, **m})
    table = pd.DataFrame(rows).round(1)
    table.to_csv(outputs_dir / "metrics_by_city.csv", index=False)
    return table


def final_forecast_table(
    forecast_df: pd.DataFrame,
    outputs_dir: Path | None = None,
) -> pd.DataFrame:
    outputs_dir = _resolve_outputs_dir(outputs_dir)
    out = forecast_df.copy()
    out["forecast_quarter"] = out["forecast_quarter"].astype(str)
    out.to_csv(outputs_dir / "final_4q_forecast.csv", index=False)
    return out


def feature_importance_table(
    model,
    feature_names: list[str],
    outputs_dir: Path | None = None,
) -> pd.DataFrame:
    outputs_dir = _resolve_outputs_dir(outputs_dir)
    scores = model.feature_importances_
    table = (
        pd.DataFrame({"feature": feature_names, "importance": scores})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    table.to_csv(outputs_dir / "feature_importance.csv", index=False)
    return table
