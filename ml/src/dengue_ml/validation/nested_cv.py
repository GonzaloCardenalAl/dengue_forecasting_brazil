import numpy as np
import pandas as pd

from dengue_ml.config import (
    TARGET, CITY_COL, FORECAST_HORIZON,
    MODEL_NAMES, FEATURE_SET_FOR_MODEL, CITIES,
)
from dengue_ml.features.feature_pipeline import build_features_for_split
from dengue_ml.models.baseline import seasonal_naive_forecast
from dengue_ml.models.sarima import tune_sarima, fit_sarima, forecast_sarima
from dengue_ml.models.xgboost_models import train_xgb, predict_xgb
from dengue_ml.models.xrfm_models import train_xrfm, predict_xrfm
from dengue_ml.training.hyperparameter_search import random_search_xgb, random_search_xrfm
from dengue_ml.validation.time_splits import make_outer_splits, make_inner_splits
from dengue_ml.validation.metrics import calculate_all_metrics


def run_nested_cv(
    df: pd.DataFrame,
    model_names: list[str] = MODEL_NAMES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full nested rolling cross-validation.

    Returns
    -------
    fold_metrics     : one row per (fold, model, city)
    fold_predictions : one row per (fold, model, city, week)
    best_hyperparams : one row per (fold, model)
    """
    outer_splits = make_outer_splits(df)
    all_metrics  = []
    all_preds    = []
    all_hparams  = []

    for fold_idx, (outer_train, outer_test) in enumerate(outer_splits):
        print(f"\n=== Outer fold {fold_idx + 1}/{len(outer_splits)} "
              f"(test: {outer_test['week_start'].min().date()} – "
              f"{outer_test['week_start'].max().date()}) ===")

        inner_splits = make_inner_splits(outer_train)

        for model_name in model_names:
            print(f"  [{model_name}] ... ", end="", flush=True)

            try:
                preds_df, hparams = _run_one_model(
                    model_name, outer_train, outer_test, inner_splits
                )
            except Exception as e:
                print(f"FAILED: {e}")
                continue

            # Merge predictions with ground truth
            preds_df = preds_df.merge(
                outer_test[[CITY_COL, "week_start", TARGET]],
                on=[CITY_COL, "week_start"],
                how="left",
            )
            preds_df["fold"]  = fold_idx + 1
            preds_df["model"] = model_name
            all_preds.append(preds_df)

            # Metrics — overall and per city
            for city in preds_df[CITY_COL].unique():
                city_df = preds_df[preds_df[CITY_COL] == city]
                m = calculate_all_metrics(
                    city_df[TARGET].values, city_df["predicted"].values
                )
                all_metrics.append({
                    "fold": fold_idx + 1, "model": model_name, "city": city, **m,
                })

            all_hparams.append({
                "fold": fold_idx + 1, "model": model_name, **hparams,
            })
            print("done")

    fold_metrics     = pd.DataFrame(all_metrics)
    fold_predictions = pd.concat(all_preds, ignore_index=True) if all_preds else pd.DataFrame()
    best_hyperparams = pd.DataFrame(all_hparams)

    # growth_proxy (the epidemic classifier's predicted probability) isn't
    # available here -- it comes from a separate classifier nested CV run.
    # train_pipeline.py joins it in afterward (conditional_residuals.
    # attach_classifier_proxy) and re-runs assign_loFo_conditional_ci once
    # growth_proxy is actually populated.
    return fold_metrics, fold_predictions, best_hyperparams


def _run_one_model(
    model_name: str,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    inner_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict]:
    """Return (predictions_df with 'predicted' column, hparams_dict)."""

    if model_name == "baseline":
        result = seasonal_naive_forecast(outer_train, outer_test)
        return result[[CITY_COL, "week_start", "predicted"]], {}

    if model_name == "sarima":
        city_preds = []
        city_hparams = {}
        for city in CITIES:
            city_train = outer_train[outer_train[CITY_COL] == city].copy()
            city_test  = outer_test[outer_test[CITY_COL] == city].copy()
            if city_train.empty or city_test.empty:
                continue

            order, sorder = tune_sarima(city_train, inner_splits, city=city)
            train_s = (
                city_train.set_index("week_start")[TARGET].sort_index()
            )
            result = fit_sarima(np.log1p(train_s), order, sorder)
            preds_log, lower_log, upper_log = forecast_sarima(result, horizon=len(city_test))
            preds = np.expm1(preds_log)

            city_df = city_test[[CITY_COL, "week_start"]].copy()
            city_df["predicted"] = preds
            city_df["lower_95"]  = np.expm1(lower_log)
            city_df["upper_95"]  = np.expm1(upper_log)
            city_preds.append(city_df)
            city_hparams[f"{city}_order"]  = str(order)
            city_hparams[f"{city}_sorder"] = str(sorder)

        return pd.concat(city_preds, ignore_index=True), city_hparams

    if model_name.startswith("xgb"):
        feature_set = FEATURE_SET_FOR_MODEL[model_name]
        best_params = random_search_xgb(outer_train, feature_set, inner_splits)

        X_tr, y_tr, X_te, _, meta_te, _ = build_features_for_split(
            outer_train, outer_test, feature_set
        )

        model = train_xgb(X_tr, y_tr, params=best_params)
        preds = np.expm1(predict_xgb(model, X_te))

        preds_df = meta_te.copy()
        preds_df["predicted"]  = preds
        # growth_proxy/lower_95/upper_95 are filled in later, after the
        # classifier proxy has been joined in — see
        # conditional_residuals.attach_classifier_proxy and train_pipeline.py.
        return preds_df, best_params

    if model_name.startswith("xrfm"):
        feature_set = FEATURE_SET_FOR_MODEL[model_name]

        # xRFM's fit() requires a held-out X_val/y_val (used internally for
        # early stopping), unlike XGBoost which trains on 100% of outer_train.
        # inner_splits' last entry is exactly "all-but-the-most-recent-horizon-
        # weeks vs. the most-recent-horizon-weeks" — reuse it instead of
        # carving a separate split. This means xRFM's kernel regression only
        # sees train_tail as support points (slightly less history than
        # XGBoost gets), an inherent consequence of the required val split.
        if not inner_splits:
            raise ValueError(
                f"Not enough history in outer_train to carve an xRFM val "
                f"split for {model_name}."
            )
        train_tail, val_tail = inner_splits[-1]

        best_params = random_search_xrfm(outer_train, feature_set, inner_splits)

        X_tr, y_tr, X_val, y_val, _, _ = build_features_for_split(
            train_tail, val_tail, feature_set
        )
        # Test features still use the FULL outer_train as history (not
        # train_tail) + outer_test, matching XGBoost's test construction.
        _, _, X_te, _, meta_te, _ = build_features_for_split(
            outer_train, outer_test, feature_set
        )

        model = train_xrfm(X_tr, y_tr, X_val, y_val, params=best_params)
        preds = np.expm1(predict_xrfm(model, X_te))

        preds_df = meta_te.copy()
        preds_df["predicted"] = preds
        return preds_df, best_params

    raise ValueError(f"Unknown model_name '{model_name}'.")
