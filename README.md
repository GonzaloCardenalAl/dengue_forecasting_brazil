# Dengue Forecasting Brazil

Quarterly dengue forecasting MVP for Brazilian Southeast capitals.

## Project Overview

This project forecasts dengue case counts (`casos_est`, InfoDengue's nowcast-corrected estimate) for four Brazilian Southeast state capitals — **Vitória**, **Belo Horizonte**, **Rio de Janeiro**, and **São Paulo** — at weekly granularity, aggregated into a quarterly deliverable with 95% confidence intervals. The goal is to give public-health stakeholders an early-warning signal for upcoming dengue epidemic seasons, conditioned on case history and climate/ENSO (El Niño–Southern Oscillation) signals.

The repository is a [uv](https://docs.astral.sh/uv/) workspace with three packages:

- **`ml/`** (`dengue-ml`) — the training, validation, and forecasting pipeline. This is what the rest of this document describes.
- **`app/`** (`dengue-app`) — a FastAPI backend + Streamlit dashboard ("DORA", the Dengue Outbreak Response Assistant) that serves the pipeline's forecasts. See [RUN_APP.md](RUN_APP.md) for how to run it.
- **`packages/dengue_core`** (`dengue-core`) — a shared-utilities package (see [Repository Structure](#repository-structure)).

High-level workflow: raw weekly case + climate data → feature engineering → nested cross-validation across multiple model families → final model trained on all data → 52-week-ahead forecast with calibrated 95% CI, aggregated to quarterly → served by `app/`.

## Repository Structure

```
.
├── packages/dengue_core/   # shared-utilities package (see TODO below)
├── ml/                     # training/forecasting pipeline (package: dengue-ml)
│   ├── data/raw/           # source CSVs (case data, SST/ENSO indices)
│   ├── src/dengue_ml/      # library code (preprocessing, features, models, validation, training, forecasting, reporting)
│   ├── configs/            # model_training.yaml — hyperparameter/CV config
│   ├── scripts/            # pipeline entry points + run_pipeline.sh (SLURM)
│   ├── results/            # per-run outputs (run_<timestamp>/, production_run/, latest_run.txt)
│   ├── notebooks/          # EDA notebooks + committed figures
│   └── tests/              # pytest suite
├── app/                    # FastAPI backend + Streamlit dashboard (package: dengue-app)
│   ├── src/dengue_app/     # main.py (API), dashboard.py (Streamlit), views/, etc.
│   ├── Dockerfile
│   └── docker-entrypoint.sh
├── presentation/           # slide-deck source (gitignored, not part of the pipeline)
├── pyproject.toml          # uv workspace root
└── uv.lock
```

**TODO:** `packages/dengue_core` is declared as a workspace dependency of both `ml` and `app` (`[tool.uv.sources]` in their `pyproject.toml`), but `src/dengue_core/__init__.py` is currently empty and nothing in the codebase imports `dengue_core`. It is reserved for future shared utilities but not yet populated — decide whether to build it out (e.g. shared path/column-name constants currently duplicated between `ml` and `app`) or remove it from the workspace.

## Data and Exploratory Data Analysis

Raw data lives in `ml/data/raw/`:

| File | Contents |
|---|---|
| `infodengue_capitals_subsetBR.csv` | Weekly dengue case data (`casos_est`, incidence, climate covariates, alert levels) per city, sourced from InfoDengue |
| `sst_indices.csv` | NOAA sea-surface-temperature / Niño region indices |
| `RONI_nino_3_4.csv` | ENSO (El Niño–Southern Oscillation) index, Niño 3.4 region |

Exploratory analysis lives in `ml/notebooks/`:

- `data_exploratory_analysis_infodengue.ipynb` — dengue case data exploration (seasonality, reporting lag, missingness)
- `data_exploratory_analysis_sst.ipynb` — SST/ENSO climate data exploration

Supporting figures from this EDA are committed at `ml/notebooks/figures/` (~40 PNGs) — these are deliverables, not pipeline-generated artifacts, so unlike `ml/results/`, they are intentionally tracked in git.

**Dataset assumption:** the most recent ~13 weeks of `casos_est` are treated as not-yet-converged (InfoDengue revises nowcasts upward as more reports arrive), so the pipeline's reliability cutoff (`compute_max_reliable_week` in `config.py`) excludes them from training/evaluation/inference.

## Machine Learning Pipeline

### Preprocessing

`data_loading.py` loads the three raw CSVs; `preprocessing.py` (`prepare_model_table`) merges weekly case data with SST indices into a single per-city, per-week model table.

### Feature engineering

Built by `features/feature_pipeline.py` (`build_features`, `build_classification_features`, `build_features_for_split`), drawing on:

- **Temporal**: `temporal_features.py` (calendar/seasonal encodings)
- **Autoregressive/lag**: `target_lag_features.py`, `weekly_lag_features.py`, `monthly_lag_features.py`
- **Climate**: `climate_features.py`, `sst_features.py` (El Niño/La Niña regime flags, thresholds ±0.5 on the Niño 3.4 anomaly)
- **Forecast-time proxies**: `forecast_proxies.py` (values usable at inference time when the true target isn't yet known)
- **Epidemiological**: `rt_estimation.py` — a Wallinga–Teunis effective reproduction number (Rt) estimator, ported from an original R implementation

Three named feature sets — `cases_only`, `cases_climate`, `cases_climate_sst` — control which of the above groups are included, and map directly to model name suffixes (see below).

### Models

Eight point-forecast models (`MODEL_NAMES` in `config.py`), grouped by family:

- `baseline` — seasonal naive (same quarter, prior year)
- `sarima` — SARIMAX (seasonal period 4, Fourier seasonal terms)
- `xgb_cases_only`, `xgb_cases_climate`, `xgb_cases_climate_sst` — XGBoost, one per feature set, with separate quantile models (2.5%/97.5%) for interval estimation
- `xrfm_cases_only`, `xrfm_cases_climate`, `xrfm_cases_climate_sst` — xRFM (CPU-only; requires a validation split for early stopping, unlike XGBoost)

A separate **epidemic classifier** (`logreg` and `xgb_clf`, `models/classifier_models.py`) predicts "epidemic this week" and is used to calibrate forecast intervals (see Metrics below) rather than to produce point forecasts itself.

### Validation methodology

Nested rolling cross-validation, defined in `validation/time_splits.py` and `validation/nested_cv.py`:

- **Outer folds**: 10 annual folds (`OUTER_CUTOFFS` in `config.py`), train ≤ year *Y*, test = year *Y+1*, stepping from 2015 → 2024.
- **Inner folds**: used for hyperparameter search within each outer training window (`training/hyperparameter_search.py`, random search per `ml/configs/model_training.yaml`).
- **Classifier CV**: `validation/nested_cv_classifier.py` runs the same nested structure for the epidemic classifier.
- **Autoregressive CV**: `validation/autoregressive_cv.py` rolls the actual multi-step forecast loop forward within each outer test window (rather than 1-step-ahead predictions from real historical lags), to capture horizon-compounding error for interval calibration.

### Metrics

`validation/metrics.py` computes MAE, RMSE, and MAPE (`calculate_all_metrics`) in the original (non-log) scale, per fold/model/city.

95% confidence intervals use a **regime-conditional residual quantile** approach (`validation/conditional_residuals.py`): the residual distribution is split by whether the epidemic classifier's predicted probability of "epidemic this week" is ≥ 0.5 (rather than one uniform quantile band), and bucketed by forecast horizon since multi-step autoregressive error compounds. Empirical leave-one-fold-out coverage against the nominal 97.5% upper-bound target (`ml/results/production_run/coverage_by_gap_with_ar.csv`) is ~94.6% overall for one-step predictions (~96.8% excluding a known Vitória data-collection gap).

### Final training and artifact generation

`training/final_train.py` selects and trains the best model (`select_best_model_with_ar_stability`) and best classifier (`select_best_classifier`) on all available data. `forecasting/forecast_next_52w.py` and `forecasting/autoregressive.py` produce the 52-week-ahead forecast; `forecasting/quarterly_aggregation.py` sums it into the quarterly deliverable with a separately-calibrated quarterly CI. `reporting/plots.py` (~20 functions) and `reporting/results_tables.py` (~8 functions) write the diagnostic PNGs and CSVs described below.

### Model storage locations

Each pipeline run writes to `ml/results/run_<YYYYMMDD_HHMMSS>/`, including:

- `final_model.pkl`, `final_classifier.pkl` — trained artifacts
- `final_quarterly_forecast.csv`, `final_quarterly_forecast_horizon_aware.csv`, `final_weekly_forecast.csv` — forecast deliverables
- `fold_metrics.csv`, `fold_predictions.csv`, `fold_predictions_ar.csv`, `fold_predictions_clf.csv` — out-of-fold validation results
- `best_hyperparameters.csv`, `best_hyperparameters_clf.csv`, `model_comparison.csv`, `model_selection_ranking.csv`, `coverage_by_gap.csv`, `proxy_comparison.csv`, `feature_importance.csv`, `feature_columns.json`
- `figures/` — diagnostic plots (model comparison, OOF predictions, forecast vs. prior year, coverage, feature importance, etc.)

`ml/results/latest_run.txt` points to the most recently created run directory. `ml/results/run_*/` is gitignored (regenerated, not committed), except for `ml/results/production_run/` — a manually promoted, git-committed copy that the deployed app reads by default (see [Pipeline Architecture](#pipeline-architecture) and [RUN_APP.md](RUN_APP.md)).

## Pipeline Architecture

```
Raw data (ml/data/raw/*.csv)
   │
   ▼
[1] run_nested_cv.py          nested CV across 8 models × 10 annual folds        (fatal)
   │
   ▼
[2] run_autoregressive_cv.py  AR-rollout CV, horizon-aware CI calibration        (non-fatal)
   │
   ▼
[3] run_classifier_cv.py      epidemic classifier CV + proxy comparison         (non-fatal)
   │
   ▼
[4] train_final_model.py      select + train best model & classifier on all data (fatal)
   │
   ▼
[5] generate_forecasts.py     52-week forecast w/ 95% CI → quarterly aggregation (fatal)
   │
   ▼
ml/results/run_<timestamp>/   (final_model.pkl, final_classifier.pkl, *.csv, figures/)
   │
   ▼
ml/results/production_run/    (manually promoted, committed copy)
   │
   ▼
app/ (FastAPI + Streamlit) reads it via DENGUE_RUN_DIR — see RUN_APP.md
```

Steps 2 and 3 are non-fatal (the orchestrating script continues even if they fail); steps 1, 4, and 5 are fatal. `ml/scripts/run_pipeline.sh` is the SLURM orchestrator that runs all five steps in order under a shared `DENGUE_RUN_ID`, requesting 8 CPUs, 8GB/CPU, and an 18-hour wall time.

## Configuration

**`ml/src/dengue_ml/config.py`** — code-level constants that drive pipeline behavior: `TARGET` (`casos_est`), `DATE_COL` (`data_iniSE`), `CITY_COL` (`city_name`), `CITIES`, `OUTER_CUTOFFS` (the 10 annual CV folds), `MODEL_NAMES`, `FEATURE_SETS`, `FORECAST_HORIZON` (52 weeks), `EL_NINO_THRESHOLD`/`LA_NINA_THRESHOLD` (±0.5). Change here to add a city, model, or feature set.

**`ml/configs/model_training.yaml`** — tunable without touching code: `random_seed`, CV settings, per-model `default_params`/`param_distributions` (XGBoost, xRFM, SARIMA, logistic regression, XGBoost classifier), hyperparameter-search trial counts, and the quantile levels (0.0125/0.9875) used for 95% CI.

**Environment variables**: `DENGUE_RUN_DIR` overrides which run directory `ml/src/dengue_ml/run_dir.py` treats as "current" (used by `app/` to point at a specific run; see [RUN_APP.md](RUN_APP.md) for app-side environment variables).

## Reproducibility

```bash
uv sync                                          # sync the full workspace
```

`torch` is pinned to a CPU-only wheel index (`pytorch-cpu`) in `pyproject.toml`/`ml/pyproject.toml`, since this project runs on a GPU-less cluster and the default PyPI wheel pulls in several GB of unused CUDA dependencies. `uv.lock` is committed for exact dependency reproducibility.

`random_seed` in `ml/configs/model_training.yaml` governs stochastic steps (hyperparameter search, model initialization); the CV fold boundaries themselves are deterministic fixed calendar cutoffs.

To rerun the full pipeline (SLURM):

```bash
bash ml/scripts/run_pipeline.sh
```

To rerun (or debug) an individual stage locally:

```bash
uv run --package dengue-ml python ml/scripts/run_nested_cv.py
uv run --package dengue-ml python ml/scripts/run_autoregressive_cv.py
uv run --package dengue-ml python ml/scripts/run_classifier_cv.py
uv run --package dengue-ml python ml/scripts/train_final_model.py
uv run --package dengue-ml python ml/scripts/generate_forecasts.py
```

Each step after the first reads from the run directory recorded in `ml/results/latest_run.txt` (or `DENGUE_RUN_ID`, if set, to target a specific run), so they must run in order.

## Testing

`ml/tests/` contains a pytest suite:

- `test_config.py` — config constants
- `test_feature_pipeline.py` — feature engineering
- `test_metrics.py` — MAE/RMSE/MAPE
- `test_time_splits.py` — CV fold construction
- `test_conditional_residuals.py` — regime-conditional quantile calibration
- `test_classification_metrics.py` — classifier metrics
- `test_data_refresh.py` — data refresh logic
- `test_nested_cv_classifier.py` — classifier nested CV

```bash
uv run --package dengue-ml pytest ml/tests/
```

**Note:** `app/` currently has no automated test suite — see [RUN_APP.md](RUN_APP.md)'s Troubleshooting section for manual verification.
