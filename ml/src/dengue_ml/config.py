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

# ── Reporting lag: quarters whose end date is within 13 weeks of the data pull
# are considered potentially unreliable. Set to the last quarter that had
# sufficient time to converge. Data runs through Q4 2025; pulled mid-2026 so
# Q4 2025 is stable. Update this constant when refreshing the dataset.
MAX_RELIABLE_QUARTER = pd.Timestamp("2025-10-01")  # start of 2025-Q4

# ── Forecasting / CV settings (sourced from configs/model_training.yaml) ──────
FORECAST_HORIZON = _tcfg["cv"]["forecast_horizon"]  # quarters ahead
N_INNER_FOLDS    = _tcfg["cv"]["n_inner_folds"]
RANDOM_SEED      = _tcfg["random_seed"]
XGB_N_TRIALS     = _tcfg["hyperparameter_search"]["xgb_n_trials"]
XRFM_N_TRIALS    = _tcfg["hyperparameter_search"]["xrfm_n_trials"]

# ── CV splits ────────────────────────────────────────────────────────────────
# Outer fold cutoffs: train ≤ cutoff, test = next FORECAST_HORIZON quarters.
# 10 folds, stepping annually from 2015Q4 → 2024Q4.
# Fold 1 trains on 2010Q1–2015Q4 (24 quarters per city; ~6 years of history).
OUTER_CUTOFFS = [
    pd.Timestamp("2015-10-01"),  # fold 1:  train ≤ 2015Q4, test 2016Q1–Q4
    pd.Timestamp("2016-10-01"),  # fold 2:  train ≤ 2016Q4, test 2017Q1–Q4
    pd.Timestamp("2017-10-01"),  # fold 3:  train ≤ 2017Q4, test 2018Q1–Q4
    pd.Timestamp("2018-10-01"),  # fold 4:  train ≤ 2018Q4, test 2019Q1–Q4
    pd.Timestamp("2019-10-01"),  # fold 5:  train ≤ 2019Q4, test 2020Q1–Q4
    pd.Timestamp("2020-10-01"),  # fold 6:  train ≤ 2020Q4, test 2021Q1–Q4
    pd.Timestamp("2021-10-01"),  # fold 7:  train ≤ 2021Q4, test 2022Q1–Q4
    pd.Timestamp("2022-10-01"),  # fold 8:  train ≤ 2022Q4, test 2023Q1–Q4
    pd.Timestamp("2023-10-01"),  # fold 9:  train ≤ 2023Q4, test 2024Q1–Q4
    pd.Timestamp("2024-10-01"),  # fold 10: train ≤ 2024Q4, test 2025Q1–Q4
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
