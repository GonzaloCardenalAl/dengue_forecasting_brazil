import pandas as pd
from joblib import Parallel, delayed

from dengue_ml.config import (
    CITY_COL, TARGET, MODEL_NAMES, FEATURE_SET_FOR_MODEL, CLASSIFICATION_FEATURE_SET,
)
from dengue_ml.features.feature_pipeline import build_features, build_features_for_split, build_classification_features
from dengue_ml.models.xgboost_models import train_xgb, predict_xgb
from dengue_ml.models.xrfm_models import train_xrfm, predict_xrfm
from dengue_ml.models.classifier_models import train_logreg, predict_proba_logreg, train_xgb_clf, predict_proba_xgb_clf
from dengue_ml.validation.time_splits import make_outer_splits, make_inner_splits
from dengue_ml.forecasting.autoregressive import forecast_with_rt_estimation

# Compounding-error problem only applies to the autoregressive xgb/xrfm model
# families (lag features fed from the model's own prior predictions).
# `baseline` has no such feedback and `sarima`'s Kalman-filter CI already
# widens with horizon -- see plan/PR notes. Restrict this CV to those.
AR_CV_MODEL_NAMES = [m for m in MODEL_NAMES if m.startswith("xgb") or m.startswith("xrfm")]


def run_autoregressive_cv(
    df: pd.DataFrame,
    best_hyperparams: pd.DataFrame,
    classifier_model_name: str,
    classifier_hyperparams: pd.DataFrame,
    model_names: list[str] = AR_CV_MODEL_NAMES,
    n_jobs: int = 8,
) -> pd.DataFrame:
    """
    Second validation procedure, structurally separate from run_nested_cv():
    rolls the *actual* autoregressive multi-step forecast loop
    (forecasting/autoregressive.forecast_with_rt_estimation, the same code
    production forecasting uses) forward across each of the same 10 outer
    fold test windows, instead of evaluating 1-step-ahead predictions built
    from real historical lags. This produces residuals that reflect real
    compounding error, used to calibrate a horizon-dependent (quarter-
    position) CI -- see validation/conditional_residuals.py's
    compute_horizon_bucketed_quarterly_residual_quantile_table.

    Reuses already-tuned hyperparameters from `best_hyperparams` /
    `classifier_hyperparams` (run_nested_cv.py's / run_nested_cv_classifier's
    output) rather than re-running random search -- the 10 outer folds are
    independent given fixed hyperparams, so this runs in parallel across
    folds via joblib.

    Returns
    -------
    fold_predictions_ar : one row per (fold, model, city, week) with
        columns [city_name, week_start, fold, model, predicted, casos_est,
        growth_proxy, quarter_position, month_position, week_position].
        quarter_position (1-4) and month_position (1-12) are simply the
        calendar quarter/month of week_start, since every outer fold's test
        window is exactly one calendar year starting Jan 1
        (OUTER_CUTOFFS are all Dec-31 -- see config.py). week_position (1..
        horizon) is the rollout's own step count instead, since ISO week
        numbering has 52/53-week year edge cases that calendar month/quarter
        don't.
    """
    outer_splits = make_outer_splits(df)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_fold)(
            fold_idx, outer_train, outer_test, model_names,
            best_hyperparams, classifier_model_name, classifier_hyperparams,
        )
        for fold_idx, (outer_train, outer_test) in enumerate(outer_splits)
    )
    results = [r for r in results if r is not None and not r.empty]
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _run_one_fold(
    fold_idx: int,
    outer_train: pd.DataFrame,
    outer_test: pd.DataFrame,
    model_names: list[str],
    best_hyperparams: pd.DataFrame,
    classifier_model_name: str,
    classifier_hyperparams: pd.DataFrame,
) -> pd.DataFrame:
    fold = fold_idx + 1
    print(f"[autoregressive CV] fold {fold}: test "
          f"{outer_test['week_start'].min().date()} - {outer_test['week_start'].max().date()}",
          flush=True)

    clf_params = _params_from_row(
        classifier_hyperparams[
            (classifier_hyperparams["fold"] == fold) & (classifier_hyperparams["model"] == classifier_model_name)
        ]
    )
    try:
        clf_model, clf_feature_set = _train_classifier(classifier_model_name, outer_train, clf_params)
    except Exception as e:
        print(f"[autoregressive CV] fold {fold}: classifier training FAILED: {e}", flush=True)
        return pd.DataFrame()
    classifier_artifact = {"model_name": classifier_model_name, "model": clf_model, "feature_set": clf_feature_set}

    horizon = outer_test["week_start"].nunique()

    rows = []
    for model_name in model_names:
        params = _params_from_row(
            best_hyperparams[(best_hyperparams["fold"] == fold) & (best_hyperparams["model"] == model_name)]
        )
        try:
            model, predict_fn, feature_set, climate_stats = _train_regression_model(model_name, outer_train, params)

            artifact = {
                "model_name": model_name, "model": model, "feature_set": feature_set,
                "climate_stats": climate_stats, "residual_quantiles": None,
            }
            rollout_rows, _ = forecast_with_rt_estimation(
                artifact, outer_train, horizon, predict_fn, classifier_artifact
            )
        except Exception as e:
            print(f"[autoregressive CV] fold {fold} [{model_name}]: FAILED: {e}", flush=True)
            continue

        rollout_rows = rollout_rows.rename(columns={
            "city": CITY_COL, "forecast_week": "week_start",
            "predicted_cases": "predicted", "proxy_value": "growth_proxy",
        })
        rollout_rows = rollout_rows.merge(
            outer_test[[CITY_COL, "week_start", TARGET]], on=[CITY_COL, "week_start"], how="left",
        )
        rollout_rows["fold"] = fold
        rollout_rows["model"] = model_name
        rollout_rows["quarter_position"] = rollout_rows["week_start"].dt.quarter
        # Same Jan-1-aligned-test-window property that makes quarter_position
        # exact also makes calendar month exact for month_position. week_position
        # is the rollout's own step count (1..horizon) rather than calendar week,
        # since ISO week numbering has 52/53-week year edge cases that calendar
        # month/quarter don't.
        rollout_rows["month_position"] = rollout_rows["week_start"].dt.month
        rollout_rows["week_position"] = (
            rollout_rows.groupby(CITY_COL)["week_start"].rank(method="first").astype(int)
        )
        rows.append(rollout_rows[
            [CITY_COL, "week_start", "fold", "model", "predicted", TARGET, "growth_proxy",
             "quarter_position", "month_position", "week_position"]
        ])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _train_regression_model(model_name: str, outer_train: pd.DataFrame, params: dict):
    """Mirrors nested_cv.py's _run_one_model training calls, minus the
    hyperparameter search (params are already tuned)."""
    feature_set = FEATURE_SET_FOR_MODEL[model_name]
    X_tr, y_tr, _, climate_stats = build_features(outer_train, feature_set)

    if model_name.startswith("xgb"):
        model = train_xgb(X_tr, y_tr, params=params)
        return model, predict_xgb, feature_set, climate_stats

    if model_name.startswith("xrfm"):
        inner_splits = make_inner_splits(outer_train)
        if not inner_splits:
            raise ValueError(
                f"Not enough history in outer_train to carve an xRFM val split for {model_name}."
            )
        train_tail, val_tail = inner_splits[-1]
        X_tr_tail, y_tr_tail, X_val, y_val, _, _ = build_features_for_split(train_tail, val_tail, feature_set)
        model = train_xrfm(X_tr_tail, y_tr_tail, X_val, y_val, params=params)
        return model, predict_xrfm, feature_set, climate_stats

    raise ValueError(f"Unknown model_name '{model_name}'.")


def _train_classifier(classifier_model_name: str, outer_train: pd.DataFrame, params: dict):
    """Mirrors nested_cv_classifier.py's _run_one_classifier training calls,
    minus the hyperparameter search."""
    X_tr, y_tr, _, _ = build_classification_features(outer_train, CLASSIFICATION_FEATURE_SET)

    if classifier_model_name == "logreg":
        model = train_logreg(X_tr, y_tr, params=params)
    elif classifier_model_name == "xgb_clf":
        model = train_xgb_clf(X_tr, y_tr, params=params)
    else:
        raise ValueError(f"Unknown classifier model_name '{classifier_model_name}'.")

    return model, CLASSIFICATION_FEATURE_SET


def _params_from_row(rows: pd.DataFrame) -> dict:
    """
    best_hyperparams/classifier_hyperparams are saved as one flattened CSV
    across all model types, so non-applicable columns are NaN for any given
    (fold, model) row. Drops fold/model/NaN columns, and rounds whole-number
    floats back to int (CSV round-tripping loses the original int dtype that
    random_search_* produced via rng.choice(...).item()) -- both XGBoost and
    sklearn accept an int where a float param is expected, so this is a safe
    direction to coerce.
    """
    if rows.empty:
        return {}
    row = rows.iloc[0]
    params = {}
    for k, v in row.items():
        if k in ("fold", "model") or pd.isna(v):
            continue
        if isinstance(v, float) and v == int(v):
            v = int(v)
        params[k] = v
    return params
