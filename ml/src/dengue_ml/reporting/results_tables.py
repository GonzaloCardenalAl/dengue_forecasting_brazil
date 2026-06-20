import numpy as np
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
    period_col: str = "forecast_week",
    filename: str = "final_weekly_forecast.csv",
) -> pd.DataFrame:
    outputs_dir = _resolve_outputs_dir(outputs_dir)
    out = forecast_df.copy()
    out[period_col] = out[period_col].astype(str)
    out.to_csv(outputs_dir / filename, index=False)
    return out


def proxy_comparison_table(
    fold_predictions_clf: pd.DataFrame,
    fold_predictions_reg: pd.DataFrame,
    outputs_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Fair side-by-side comparison of the nivel_inc rule, sustained_rt rule, and
    trained classifier(s) -- all scored on the SAME (city, week, fold)
    rows/label from fold_predictions_clf (see nested_cv_classifier.py), plus
    each candidate's downstream LOFO coverage when substituted for the
    production CI-regime proxy (averaged across every regression model that
    has a growth_proxy, i.e. the XGBoost/xRFM families).
    """
    from dengue_ml.validation.classification_metrics import calculate_all_classification_metrics
    from dengue_ml.validation.conditional_residuals import compute_regime_coverage, PROXY_COL

    outputs_dir = _resolve_outputs_dir(outputs_dir)

    reg_models = [
        m for m in fold_predictions_reg["model"].unique()
        if fold_predictions_reg.loc[fold_predictions_reg["model"] == m, PROXY_COL].notna().any()
    ]

    def _coverage_for(key_to_high: dict | None) -> tuple[float, float, float]:
        covs, n_highs, n_lows = [], [], []
        for m in reg_models:
            mask = None
            if key_to_high is not None:
                sub = fold_predictions_reg[fold_predictions_reg["model"] == m]
                keys = list(zip(sub["city_name"], sub["week_start"]))
                mask = pd.Series([bool(key_to_high.get(k, False)) for k in keys], index=sub.index)
            result = compute_regime_coverage(fold_predictions_reg, m, high_regime_mask=mask)
            covs.append(result["coverage"]); n_highs.append(result["n_high"]); n_lows.append(result["n_low"])
        return float(np.mean(covs)), float(np.mean(n_highs)), float(np.mean(n_lows))

    rows = []

    # Both are legacy rule-based candidates, scored for comparison only --
    # neither is the production proxy anymore (that's now the selected
    # classifier model's own predicted_proba, joined into fold_predictions_reg
    # as growth_proxy by conditional_residuals.attach_classifier_proxy), so
    # both need their own key_to_high mask rather than relying on the
    # (classifier-based) production default.
    ref = fold_predictions_clf.drop_duplicates(["city_name", "week_start"])
    for rule_col, label in [
        ("nivel_inc_rule", "nivel_inc"),
        ("sustained_rt_rule", "sustained_rt"),
    ]:
        m = calculate_all_classification_metrics(ref["is_epidemic"].values, ref[rule_col].values)
        key_to_high = dict(zip(zip(ref["city_name"], ref["week_start"]), ref[rule_col].astype(bool)))
        coverage, n_high, n_low = _coverage_for(key_to_high)
        rows.append({"candidate": label, **m, "coverage": coverage, "n_high_regime": n_high, "n_low_regime": n_low})

    for model_name in fold_predictions_clf["model"].unique():
        sub = fold_predictions_clf[fold_predictions_clf["model"] == model_name]
        pred = (sub["predicted_proba"].values >= 0.5).astype(int)
        m = calculate_all_classification_metrics(sub["is_epidemic"].values, pred, sub["predicted_proba"].values)
        key_to_high = dict(zip(zip(sub["city_name"], sub["week_start"]), sub["predicted_proba"] >= 0.5))
        coverage, n_high, n_low = _coverage_for(key_to_high)
        rows.append({"candidate": model_name, **m, "coverage": coverage, "n_high_regime": n_high, "n_low_regime": n_low})

    table = pd.DataFrame(rows).round(3)
    table.to_csv(outputs_dir / "proxy_comparison.csv", index=False)
    return table


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
