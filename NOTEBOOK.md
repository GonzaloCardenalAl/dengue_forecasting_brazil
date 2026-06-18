# Research Notebook

_Auto-maintained by Claude Code. One entry per completed task._

---
## Convert ML pipeline from quarterly to monthly training grain — 2026-06-19 (approx, session end)

**Goal:** Train/validate the dengue forecasting models monthly instead of quarterly (to leverage the weekly InfoDengue data more finely), while still producing quarterly forecasts as the final deliverable (sum 3 months -> 1 quarter).

**Changes:**
- `ml/src/dengue_ml/config.py` — `MAX_RELIABLE_QUARTER` -> `MAX_RELIABLE_MONTH` (2025-12-01, verified against raw data's actual last week 2025-12-28); `OUTER_CUTOFFS` rebuilt as 10 Dec-1-of-year timestamps (not a literal reuse of the old Oct-1 quarter-start values — Dec-1 is required so `month_start <= cutoff` includes the full year in train, preserving identical Jan-Dec test windows to the original quarterly folds); `FORECAST_HORIZON` now 12 (months).
- `ml/src/dengue_ml/preprocessing.py` — `aggregate_dengue_to_quarterly` -> `aggregate_dengue_to_monthly` (groups by month instead of quarter); `prepare_sst_quarterly` -> `prepare_sst_monthly` (resample `"QS"`->`"MS"`, SST is already native monthly so this is now near-lossless); `prepare_weekly_table`'s reliability cutoff column renamed `quarter_start`->`month_start`.
- `ml/src/dengue_ml/features/*.py` — `temporal_features.py` (month/month_sin/month_cos replacing quarter equivalents, period=12 not 4); `target_lag_features.py`, `climate_features.py`, `sst_features.py` — all `_1q/2q/4q` lag/rolling columns renamed to their calendar-duration equivalents (`_3m/6m/12m`), preserving the true 12-month year-over-year seasonal lag; `weekly_lag_features.py`/`monthly_lag_features.py` — only the period-start param renamed (`quarter_start`->`month_start`), the windowing logic itself was already grain-agnostic; `feature_pipeline.py` — column-name lists updated to match.
- `ml/src/dengue_ml/models/sarima.py` — seasonal period now `m=12` (from yaml); minimum-training-history guard `<8` -> `<24` (2 years of months, was 2 years of quarters).
- `ml/src/dengue_ml/models/baseline.py` — seasonal-naive rewritten as "same month, prior year" (was "same quarter, prior year").
- `ml/src/dengue_ml/validation/time_splits.py` — `make_outer_splits`/`make_inner_splits` renamed to operate on `month_start`; `min_train_q=max(horizon*2,8)` -> `min_train_m=max(horizon*2,24)`.
- `ml/src/dengue_ml/validation/nested_cv.py`, `nested_cv_classifier.py` — `quarter_start`->`month_start` renames throughout (merge keys, SARIMA indexing, print statements); the classifier track shares the same fold machinery so it converts in lockstep.
- `ml/src/dengue_ml/validation/conditional_residuals.py` — added `compute_quarterly_residual_quantile_table`: aggregates monthly OOF predictions to (city, quarter, fold) sums and fits the same regime-conditional empirical-quantile calibration on the quarterly-aggregated residuals, for a statistically valid quarterly 95% CI (a naive sum of monthly CI bounds would be invalid).
- `ml/src/dengue_ml/forecasting/forecast_next_4q.py` -> renamed `forecast_next_12m.py` — autoregressive loop now runs 12 months ahead instead of 4 quarters; each output row now also carries `proxy_value` (the growth_proxy used for CI calibration) so the new quarterly aggregation step can pick it up without recomputing features.
- `ml/src/dengue_ml/forecasting/quarterly_aggregation.py` — new module: `aggregate_monthly_forecast_to_quarterly` (sums monthly point forecasts to quarters, applies the new quarterly residual-quantile table for CI) and `aggregate_monthly_history_to_quarterly` (rolls up historical casos_est/min/max for the final-forecast plot).
- `ml/scripts/generate_forecasts.py` — rewritten to produce both `final_monthly_forecast.csv` and `final_quarterly_forecast.csv` (the primary deliverable), wiring in the new quarterly calibration step.
- `ml/scripts/run_classifier_cv.py` — `quarter_start`->`month_start` in the `parse_dates` call.
- `ml/src/dengue_ml/reporting/plots.py` — `plot_historical_cases`/`plot_seasonality`/`plot_oof_predictions` now show monthly grain (x-ticks 1-12, narrower bars); `plot_final_forecast` is unchanged internally (it already expected `quarter_start`/`forecast_quarter` columns) but now must be fed the new quarterly-aggregated tables from `quarterly_aggregation.py` rather than the raw (now-monthly) model table.
- `ml/src/dengue_ml/reporting/results_tables.py` — `final_forecast_table` gained `period_col`/`filename` params so monthly and quarterly outputs don't collide; `proxy_comparison_table`'s join keys renamed to `month_start`.
- `ml/configs/model_training.yaml` — `forecast_horizon: 4`->`12`, `seasonal_period: 4`->`12` (SARIMA).
- `ml/tests/test_time_splits.py` — fixtures rewritten at monthly grain (freq="MS", horizon=12, rescaled periods/cutoffs); `ml/tests/test_nested_cv_classifier.py` — fixture column renamed `quarter_start`->`month_start`. All 21 tests pass.

**Why:** User wants finer-grained training/validation (leveraging weekly InfoDengue data better than quarterly aggregation does) while keeping quarterly as the final forecasting deliverable. Calendar-duration-preserving lag renames (rather than literal period-count) were chosen specifically to keep the year-over-year seasonal signal at exactly 12 months, which is core to dengue's epidemiology.

**Assumptions & trade-offs:**
- Full replacement, no dual quarterly/monthly code path (per user's explicit choice) — the pre-existing quarterly results in `ml/results/run_*/` and git history (commit `089f1c0`) remain the only available "before" baseline for comparison.
- Verified (via direct smoke tests, not committed as test code): `prepare_model_table()` produces a 768-row (192 months x 4 cities) table with correct column names; `make_outer_splits`/`make_inner_splits` produce exactly the expected 10 outer folds x 12-month test windows and 3 inner folds x 12-month val windows; `build_features` produces a 708x132 matrix with all renamed lag columns present; the `baseline` model runs cleanly end-to-end on a single fold.
- **Not yet run to completion**: the full nested CV (`run_nested_cv.py`), `train_final_model.py`, and `generate_forecasts.py` end-to-end, due to session time constraints. SARIMA at `m=12` (monthly seasonality) is substantially slower to fit per parameter combination than the old `m=4` SARIMA — a single fold's SARIMA grid search (48 combos x 3 inner folds x 4 cities) did not finish within several minutes on the shared login node. **Recommend running the full pipeline (`run_nested_cv.py` at minimum) as a SLURM batch job rather than interactively**, and consider trimming the SARIMA `param_grid` in `model_training.yaml` if the full run proves too slow even there. XGBoost/xRFM hyperparameter search trial counts (50/30) were left unchanged per the plan, but were also not timed end-to-end at the new monthly scale.
- `run_classifier_cv.py` was updated for the column rename but not run end-to-end either.

---
