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

# ── Reporting lag: months whose end date is within 13 weeks of the data pull
# are considered potentially unreliable. Set to the last month that had
# sufficient time to converge. Data runs through Dec 2025; pulled mid-2026 so
# Dec 2025 is stable. Update this constant when refreshing the dataset.
MAX_RELIABLE_MONTH = pd.Timestamp("2025-12-01")  # start of Dec 2025

# ── Forecasting / CV settings (sourced from configs/model_training.yaml) ──────
FORECAST_HORIZON = _tcfg["cv"]["forecast_horizon"]  # months ahead
N_INNER_FOLDS    = _tcfg["cv"]["n_inner_folds"]
RANDOM_SEED      = _tcfg["random_seed"]
XGB_N_TRIALS     = _tcfg["hyperparameter_search"]["xgb_n_trials"]
XRFM_N_TRIALS    = _tcfg["hyperparameter_search"]["xrfm_n_trials"]

# ── Epidemic classifier (separate, parallel evaluation track) ────────────────
CLASSIFICATION_N_TRIALS      = _tcfg["classification"]["n_trials"]
CLASSIFICATION_FEATURE_SET   = _tcfg["classification"]["feature_set"]
CLASSIFIER_MODEL_NAMES       = ["logreg", "xgb_clf"]

# ── CV splits ────────────────────────────────────────────────────────────────
# Outer fold cutoffs: train ≤ cutoff, test = next FORECAST_HORIZON months.
# 10 folds, stepping annually. Cutoffs are Dec-1-of-year (not Oct-1/Q4-start)
# so that `month_start <= cutoff` includes all 12 months of year Y in train,
# and test = the next 12 months = Jan-Dec of year Y+1 -- identical calendar
# test windows to the original quarterly folds, just at monthly resolution.
# Fold 1 trains on 2010-01–2015-12 (72 months per city; ~6 years of history).
OUTER_CUTOFFS = [
    pd.Timestamp("2015-12-01"),  # fold 1:  train ≤ 2015-12, test 2016-01–12
    pd.Timestamp("2016-12-01"),  # fold 2:  train ≤ 2016-12, test 2017-01–12
    pd.Timestamp("2017-12-01"),  # fold 3:  train ≤ 2017-12, test 2018-01–12
    pd.Timestamp("2018-12-01"),  # fold 4:  train ≤ 2018-12, test 2019-01–12
    pd.Timestamp("2019-12-01"),  # fold 5:  train ≤ 2019-12, test 2020-01–12
    pd.Timestamp("2020-12-01"),  # fold 6:  train ≤ 2020-12, test 2021-01–12
    pd.Timestamp("2021-12-01"),  # fold 7:  train ≤ 2021-12, test 2022-01–12
    pd.Timestamp("2022-12-01"),  # fold 8:  train ≤ 2022-12, test 2023-01–12
    pd.Timestamp("2023-12-01"),  # fold 9:  train ≤ 2023-12, test 2024-01–12
    pd.Timestamp("2024-12-01"),  # fold 10: train ≤ 2024-12, test 2025-01–12
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
