from pathlib import Path
import pandas as pd

from dengue_ml.training_config import load_training_config

_tcfg = load_training_config()

# ── Paths ────────────────────────────────────────────────────────────────────
_ML_DIR = Path(__file__).resolve().parents[2]  # ml/
CONFIGS_DIR = _ML_DIR / "configs"

RAW_DIR       = _ML_DIR / "data" / "raw"
RESULTS_DIR   = _ML_DIR / "results"

# Pointer written by run_nested_cv.py at the start of each run so that
# subsequent scripts (train_final_model, generate_forecasts) find the same dir.
LATEST_RUN_FILE = RESULTS_DIR / "latest_run.txt"

DENGUE_FILE = RAW_DIR / "infodengue_capitals_subsetBR.csv"
SST_FILE    = RAW_DIR / "sst_indices.csv"
RONI_FILE   = RAW_DIR / "RONI_nino_3_4.csv"

# ── Column names (match raw CSV exactly) ─────────────────────────────────────
TARGET   = "casos_est"
DATE_COL = "data_iniSE"
CITY_COL = "city_name"

CITIES = ["Vitória", "Belo Horizonte", "Rio de Janeiro", "São Paulo"]

# ── Reporting lag: weeks whose end date is within ~13 weeks of "now" are
# considered potentially unreliable (InfoDengue's nowcast model hasn't
# converged yet, so casos_est would still be revised upward later). Computed
# dynamically off wall-clock time -- not the raw CSV's own max date -- so
# refreshing the CSV (see data_refresh.py) never requires touching this.
_RELIABILITY_LAG_WEEKS = 13


def compute_max_reliable_week(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """Latest week-start considered safe for training/eval/inference.

    Snapped to the end of the last calendar QUARTER that is both fully
    elapsed and old enough to have converged (>= ~13 weeks old) -- not just
    "now minus 13 weeks" snapped to the nearest Sunday. Every deliverable
    this cutoff feeds into is a quarterly forecast, so the forecast horizon
    must start exactly on a quarter boundary; a cutoff that rolls forward
    continuously week-by-week would land mid-quarter most of the time,
    producing a "first forecast quarter" that's really a partial-quarter sum
    of just a few predicted weeks tacked onto a quarter that's mostly
    already-known actual data (see aggregate_weekly_forecast_to_quarterly).
    """
    if now is None:
        now = pd.Timestamp.now()
    safe_now = now - pd.Timedelta(weeks=_RELIABILITY_LAG_WEEKS)
    current_quarter_start = pd.Timestamp(pd.Period(safe_now, freq="Q").start_time)
    last_full_quarter_end = current_quarter_start - pd.Timedelta(days=1)
    return last_full_quarter_end - pd.Timedelta(days=(last_full_quarter_end.dayofweek + 1) % 7)

# ── Forecasting / CV settings (sourced from configs/model_training.yaml) ──────
FORECAST_HORIZON = _tcfg["cv"]["forecast_horizon"]  # weeks ahead
N_INNER_FOLDS    = _tcfg["cv"]["n_inner_folds"]
RANDOM_SEED      = _tcfg["random_seed"]
XGB_N_TRIALS     = _tcfg["hyperparameter_search"]["xgb_n_trials"]
XRFM_N_TRIALS    = _tcfg["hyperparameter_search"]["xrfm_n_trials"]

# ── Epidemic classifier (separate, parallel evaluation track) ────────────────
CLASSIFICATION_N_TRIALS      = _tcfg["classification"]["n_trials"]
CLASSIFICATION_FEATURE_SET   = _tcfg["classification"]["feature_set"]
CLASSIFIER_MODEL_NAMES       = ["logreg", "xgb_clf"]

# ── CV splits ────────────────────────────────────────────────────────────────
# Outer fold cutoffs: train <= cutoff, test = the following calendar year
# (cutoff, cutoff + 1 year], resolved via calendar-date arithmetic in
# make_outer_splits/make_inner_splits, NOT a fixed row count -- calendar
# years have 52 or 53 weeks depending on the year (e.g. 2012/2017/2023 each
# have 53 distinct week-start dates in this dataset), so a fixed-count slice
# would silently misalign folds in those years. Cutoffs are Dec-31-of-year
# (not Dec-1, since `week_start <= cutoff` needs to unambiguously include
# every week of year Y regardless of which weekday Dec 31 falls on).
# 10 folds, stepping annually, same 10 years as the quarterly/monthly folds.
OUTER_CUTOFFS = [
    pd.Timestamp("2015-12-31"),  # fold 1:  train <= 2015, test 2016
    pd.Timestamp("2016-12-31"),  # fold 2:  train <= 2016, test 2017
    pd.Timestamp("2017-12-31"),  # fold 3:  train <= 2017, test 2018
    pd.Timestamp("2018-12-31"),  # fold 4:  train <= 2018, test 2019
    pd.Timestamp("2019-12-31"),  # fold 5:  train <= 2019, test 2020
    pd.Timestamp("2020-12-31"),  # fold 6:  train <= 2020, test 2021
    pd.Timestamp("2021-12-31"),  # fold 7:  train <= 2021, test 2022
    pd.Timestamp("2022-12-31"),  # fold 8:  train <= 2022, test 2023
    pd.Timestamp("2023-12-31"),  # fold 9:  train <= 2023, test 2024
    pd.Timestamp("2024-12-31"),  # fold 10: train <= 2024, test 2025
]

# ── Models & feature sets ─────────────────────────────────────────────────────
FEATURE_SETS = ["cases_only", "cases_climate", "cases_climate_sst"]

MODEL_NAMES = [
    "baseline",
    "sarima",
    "xgb_cases_only",
    "xgb_cases_climate",
    "xgb_cases_climate_sst",
    "xrfm_cases_only",
    "xrfm_cases_climate",
    "xrfm_cases_climate_sst",
]

FEATURE_SET_FOR_MODEL = {
    "xgb_cases_only":        "cases_only",
    "xgb_cases_climate":     "cases_climate",
    "xgb_cases_climate_sst": "cases_climate_sst",
    "xrfm_cases_only":        "cases_only",
    "xrfm_cases_climate":     "cases_climate",
    "xrfm_cases_climate_sst": "cases_climate_sst",
}

# ── ENSO thresholds ───────────────────────────────────────────────────────────
EL_NINO_THRESHOLD =  0.5
LA_NINA_THRESHOLD = -0.5
