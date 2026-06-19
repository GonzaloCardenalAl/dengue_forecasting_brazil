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
## Convert ML pipeline from monthly to weekly training/CV/forecast grain — 2026-06-20 00:01

**Goal:** Train, validate, and forecast all four model families (baseline, SARIMA, XGBoost, xRFM) at weekly grain instead of monthly, with quarterly remaining the final deliverable via post-hoc aggregation of ~13 weeks per quarter.

**Changes:**
- `ml/configs/model_training.yaml` — `cv.forecast_horizon: 12→52`; `sarima.seasonal_period: 12→52`; `sarima.param_grid` trimmed to 24 combos (`p:[0,1,2], d:[0,1], q:[0,1], P:[0,1], D:[1], Q:[0]`) to control runtime at `m=52`.
- `ml/src/dengue_ml/config.py` — `MAX_RELIABLE_MONTH`→`MAX_RELIABLE_WEEK`; `OUTER_CUTOFFS` rebuilt as 10 Dec-31-of-year timestamps (2015–2024).
- `ml/src/dengue_ml/preprocessing.py` — deleted `aggregate_dengue_to_monthly()` and `get_weekly_table()`/its `lru_cache`; rewrote `prepare_weekly_table()`/`prepare_model_table()` to build the model table directly at native weekly resolution; SST/RONI (no weekly source) bridged onto weeks via a derived containing-month key, forward-filled onto the existing monthly table.
- `ml/src/dengue_ml/validation/time_splits.py` — `make_outer_splits`/`make_inner_splits` rewritten from fixed-row-count slicing to calendar-date-arithmetic windows (`cutoff` to `cutoff + DateOffset(years=1)`), so 52- and 53-week calendar years (2012, 2017, 2023 in the real data) are both handled correctly with no off-by-one risk. `horizon` parameter removed from both.
- `ml/src/dengue_ml/features/temporal_features.py` — month-of-year cyclical encoding replaced with ISO week-of-year (`.dt.isocalendar()`, period=52).
- `ml/src/dengue_ml/features/target_lag_features.py`, `climate_features.py` — lag/rolling suffixes renamed to preserve calendar duration (`_3m/_6m/_12m`→`_13w/_26w/_52w`), shift/rolling window sizes updated accordingly.
- `ml/src/dengue_ml/features/sst_features.py` — same suffix renames (cosmetic; confirmed dead/unused code, not in any `FEATURE_COLS` list).
- `ml/src/dengue_ml/features/weekly_lag_features.py` — **core bug fix.** Replaced the `get_weekly_table()`-based cross-reference mechanism (`_city_weekly_arrays`/`_week_window_features`, `.iterrows()`-based) with direct vectorized `groupby(city).shift(k)` lags computed on whatever df is passed in. Same function signature and output column names, so no caller changes needed.
- `ml/src/dengue_ml/features/monthly_lag_features.py` — trivial: row-side cutoff param renamed from `month_start` to `week_start` (SST genuinely has no weekly source, so the cross-reference mechanism itself is unchanged).
- `ml/src/dengue_ml/features/feature_pipeline.py` — `_TEMPORAL_COLS`/`_TARGET_LAG_COLS`/`_CLIMATE_COLS` updated to new column names; meta column `month_start`→`week_start`.
- `ml/src/dengue_ml/models/sarima.py` — minimum training history check `< 24`→`< 104` (2 years of weeks); indexing renamed to `week_start`.
- `ml/src/dengue_ml/models/baseline.py` — seasonal-naive lookup rewritten from a `DateOffset(years=1)`+snap-to-month-start trick to an ISO-`(year, week)`-keyed lookup via `.dt.isocalendar()`, with a city-isoweek-median fallback.
- `ml/src/dengue_ml/validation/nested_cv.py`, `nested_cv_classifier.py` — column renames throughout (merge keys, print statements, SARIMA indexing).
- `ml/src/dengue_ml/validation/conditional_residuals.py` — `compute_quarterly_residual_quantile_table`'s hardcoded `month_start`→`week_start`.
- `ml/src/dengue_ml/forecasting/quarterly_aggregation.py` — `aggregate_monthly_forecast_to_quarterly`/`aggregate_monthly_history_to_quarterly` renamed to `aggregate_weekly_*`; column renames.
- `ml/src/dengue_ml/training/train_pipeline.py`, `final_train.py` — column renames only (the user's other unrelated pending edits in these two files were left untouched).
- `ml/src/dengue_ml/forecasting/forecast_next_12m.py` → renamed to `forecast_next_52w.py` (`generate_next_52w_forecast`); `_next_months`→`_next_weeks`; stub rows in the autoregressive loop now also carry forward `Rt`/`p_rt1`/`sustained_rt` (previously missing — needed now that lag features read directly off the model table's own rows instead of a separate cross-referenced source).
- `ml/src/dengue_ml/reporting/plots.py` — `plot_historical_cases` bar width 20→5 days; `plot_seasonality` switched to ISO-week x-axis; `plot_oof_predictions` column renames.
- `ml/src/dengue_ml/reporting/results_tables.py` — `final_forecast_table` defaults (`forecast_week`/`final_weekly_forecast.csv`); `proxy_comparison_table` column renames.
- `ml/scripts/generate_forecasts.py`, `run_classifier_cv.py` — import/column renames.
- `ml/tests/test_time_splits.py` — rewritten at weekly grain; added a new edge-case test that synthesizes a 53-week calendar year and verifies the date-arithmetic split captures all 53 weeks, not a fixed 52.
- `ml/tests/test_nested_cv_classifier.py` — fixture column rename only.

**Why:** The production forecast's autoregressive multi-step loop predicted month M+1, appended it to history, then predicted M+2 — but its weekly lag features (`cases_week_t-1`, etc.) were read from a separate, statically cached weekly table that never saw the model's own predicted future weeks. The result: every forecasted month beyond the first saw the exact same frozen "last known week" trajectory. Root-causing this led to switching the model's own row grain to weekly, so `weekly_lag_features.py`'s lags become plain `shift()` calls directly on the (growing, prediction-augmented) table — the staleness is structurally impossible once there's no separate cross-referenced source to go stale.

**Assumptions & trade-offs:**
- SARIMA now trains at `seasonal_period=52` (not kept at monthly as a coarser carve-out) per explicit user choice, despite being the only model that didn't actually have the staleness bug (SARIMA/baseline never consumed weekly lag features).
- The SARIMA param grid was trimmed from 96 to 24 combinations (fixing `D=1`, `Q=0`) specifically to control cost at `m=52`. Empirically, even this trimmed grid looks too expensive: a single fit (order `(1,1,1)x(1,1,0,52)` on ~835 weekly observations) did not converge within ~6–7 minutes, with memory climbing past 1GB before the test process was killed. The full nested CV needs 2,880 such fits. **This needs a decision before running the real nested CV**: trim the grid further (e.g. drop `p=2`, the documented fallback), or run as a SLURM batch job with substantial walltime/memory, or both.
- ISO week-of-year (`.dt.isocalendar()`, Monday-anchored) is used as an accepted approximation of the raw data's actual epi-week numbering (predominantly Sunday-anchored, InfoDengue's own convention) — not an exact match, by design.
- Verified the bug fix directly with a synthetic autoregressive-identity fake model: `predicted_cases` visibly evolved week-over-week (each step ≈1.02× the prior) instead of staying frozen, confirming `cases_week_t-1` now tracks the model's own predictions across the forecast loop.
- Full test suite (22 tests) passes. Full nested-CV run, `generate_forecasts.py`/`run_classifier_cv.py` end-to-end smoke tests, and a real SLURM batch run were not executed in this session — only the SARIMA timing concern above and the unit-level smoke tests (data load, splits, feature building, baseline, forecast-loop bug-fix verification) were run interactively.

---
