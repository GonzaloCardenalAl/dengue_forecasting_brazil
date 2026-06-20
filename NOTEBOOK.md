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
## Fix forecast-horizon feature generation: climate, alert features, Rt/p_rt1 — 2026-06-20 02:15

**Goal:** Stop seeding the 52-week-ahead autoregressive forecast loop with flat carry-forward values for features that should evolve (climate, Rt/p_rt1) or that can't legitimately be carried forward at all (InfoDengue's internal alert-classifier outputs), and replace the CI-regime proxy with something computable at forecast time.

**Changes:**
- `ml/src/dengue_ml/forecasting/forecast_next_52w.py` — added `_climatological_val()` (city + ISO-week-of-year historical mean) to seed future `tempmed`/`humidmed` instead of `_last_val()`; dropped `transmissao`/`receptivo`/`nivel_inc` stub seeding entirely; added a two-pass autoregressive design (`_forecast_with_rt_estimation`/`_autoregressive_loop`/`_estimate_rt_lookup`): a draft pass produces a plausible case trajectory, then `features/rt_estimation.py`'s Rt/p_rt1 estimator runs once over history+draft, and a final pass reuses that estimate (plus the trained classifier's predicted probability as the CI proxy) instead of carrying Rt/p_rt1/the CI proxy forward flat.
- `ml/src/dengue_ml/features/rt_estimation.py` — new module: from-scratch numpy/scipy port of the Codeco et al. (2017) temperature-dependent Wallinga-Teunis Rt estimator (Moschopoulos sum-of-gammas + renewal-equation multinomial resampling), with `p_rt1` derived as the fraction of simulated R(t) draws exceeding 1. No R/rpy2 dependency.
- `ml/src/dengue_ml/features/feature_pipeline.py` — shrank `_ALERT_VALUE_COLS` to just `Rt`/`p_rt1`/`sustained_rt` (drops `transmissao`/`receptivo`/`nivel_inc` as model inputs for both the cases models and the epidemic classifier); kept `nivel_inc_week_t-1` as a side-channel in `meta` for the legacy benchmark comparison and as the classifier's label source.
- `ml/src/dengue_ml/training/final_train.py`, `train_pipeline.py` — added `select_best_classifier`/`train_final_classifier`, wired the classifier nested CV into the main pipeline, and joined its OOF predicted probability into the existing CI calibration machinery.
- `ml/src/dengue_ml/validation/conditional_residuals.py`, `nested_cv.py`, `nested_cv_classifier.py` — `attach_classifier_proxy` joins the classifier's probability onto the regression fold rows; proxy threshold/docstrings updated from the old `nivel_inc`-based rule to the classifier-probability proxy.
- `ml/src/dengue_ml/preprocessing.py`, `reporting/results_tables.py`, `ml/scripts/generate_forecasts.py`, `train_final_model.py` — supporting renames/wiring for the above.
- `ml/tests/test_nested_cv_classifier.py` — updated fixtures for the `nivel_inc_week_t-1`-in-`meta` side channel.
- `ml/src/dengue_ml/features/Codeco-et-al-2017/` (+ top-level `EstRtGT_v4.R`, `sumgamma_v2.R`, `Rt_calc-example.rmd`) — reference materials (original paper, R scripts, example datasets) kept alongside the Python port for future validation/reference.

**Why:** The cases-forecasting loop was already correct (each predicted week's lags recompute live), but every other feature was flat-carried-forward for all 52 weeks — a defensible stub for population, but wrong for seasonal climate and conceptually broken for Rt/p_rt1 (an epidemiological parameter that should track the forecasted case curve) and for transmissao/receptivo/nivel_inc (InfoDengue's own undocumented internal classifier outputs, which can't be computed for the future at all). The CI-regime proxy previously read raw `nivel_inc`, which is exactly the kind of field that no longer exists for forecast weeks post-fix, so it was swapped for the trained epidemic classifier's predicted probability, which can run forward on forecasted weeks.

**Assumptions & trade-offs:**
- Rt/p_rt1 estimator validated against the Codeco et al. paper's own published example (FIdata.csv) and against the real R implementation (R0 package, installed via conda for this comparison) run on actual InfoDengue data — not just unit-tested in isolation.
- Two-pass design (draft forecast → estimate Rt/p_rt1 once over the full horizon → final forecast) is deliberately cheap relative to re-running the renewal-equation estimation every single forecast week.
- Two latent bugs fixed along the way: `sustained_rt`'s bool dtype upcasting to `object` after `.shift()` (rejected by XGBoost), and negative-valued draft case predictions breaking the renewal equation's non-negativity requirement (now clipped before Rt estimation).

---
## Add model-history continuity line and year-over-year comparison plot — 2026-06-20

**Goal:** Make `final_forecast.png`'s model-prediction line read as one continuous (past + future) series rather than just raw actuals followed by a forecast, and add a zoomed companion plot comparing the new forecast year directly against the same quarters one year earlier.

**Changes:**
- `ml/src/dengue_ml/forecasting/quarterly_aggregation.py` — new `aggregate_weekly_oof_predictions_to_quarterly(fold_predictions, model_name)`: sums one model's out-of-fold weekly CV predictions (`run_nested_cv`'s `fold_predictions`) to quarterly, for plotting.
- `ml/src/dengue_ml/reporting/plots.py` — `plot_final_forecast` gained an optional `oof_quarterly_df` param: plots the model's historical OOF predictions as a red dashed line (vs. the existing solid blue actual-cases line), leading into the existing solid-red forecast line, so the two red segments read as one continuous "model prediction" track. Also added new `plot_forecast_vs_previous_year(forecast_df, historical_df, n_lead_in_q=2)`: a per-city zoomed view of just the forecast year + `n_lead_in_q` quarters of lead-in actuals, with the same quarters from exactly one year earlier overlaid on the same x-axis positions (shifted forward a year) — saved as `forecast_vs_previous_year.png`.
- `ml/scripts/generate_forecasts.py` — builds `oof_quarterly_df` from the already-loaded `fold_predictions`/final model name and passes it to `plot_final_forecast`; calls the new `plot_forecast_vs_previous_year`.

**Why:** User wants to see the model's own historical track record (not just raw actuals) flow into the forecast on the main chart, and a separate, easier-to-read view focused on the new forecast year benchmarked against the equivalent quarters last year (the existing multi-year log-scale chart makes a one-year-ago comparison hard to judge by eye).

**Assumptions & trade-offs:**
- Confirmed via clarifying question with the user: the OOF line is *additive* (actual-cases history line stays as-is) rather than replacing it; the previous-year comparison plot shows previous year's *actual cases only* (not also last year's archived forecast, which would require locating a specific older run's saved forecast and was judged too fragile).
- Not validated against a real current-grain run: no `ml/results/run_*/fold_predictions.csv` on disk yet has the post-weekly-conversion `week_start` schema (all existing runs predate that conversion and use `month_start`/`quarter_start` from the old grain) — `aggregate_weekly_oof_predictions_to_quarterly` and both plot functions were smoke-tested against synthetic data matching the current schema instead (verified they render without error and look visually correct). Full test suite (22 tests) still passes.

---
## Fix oversized CI bands: exclude known data-collection gap from calibration, add monthly OOF validation plot, raise CI to 97.5% — 2026-06-20 15:22

**Goal:** Diagnose why the regime-conditional CI bands (`conditional_residuals.py`) looked absurdly wide during low-incidence/trough periods on the weekly OOF diagnostic plot, and fix it.

**Changes:**
- `ml/src/dengue_ml/validation/conditional_residuals.py`:
  - Added `KNOWN_DATA_GAPS` + `_is_known_data_gap()`: flags Vitória's 2021-01-03 to 2021-12-05 window (49 straight weeks of `casos_est == casos_est_min == casos_est_max == casos == 0`, confirmed against `ml/data/raw/infodengue_capitals_subsetBR.csv` — a real surveillance/reporting outage, not actual epidemiology). These rows are now dropped from the calibration pool in `assign_loFo_conditional_ci`, `compute_residual_quantile_table`, `compute_quarterly_residual_quantile_table`, and `compute_horizon_bucketed_quarterly_residual_quantile_table`, but still receive their own `lower_95`/`upper_95` and stay visible in plots/history.
  - Added `aggregate_oof_to_monthly(fold_predictions, model_name)`: sums weekly OOF rows to (city, month, fold), reusing `assign_loFo_conditional_ci` unmodified on the result.
  - Briefly added a `magnitude_bins` parameter (sub-binning each regime bucket by predicted-value magnitude) to fix the same symptom before the real root cause (the data gap) was found; removed again once the gap fix alone outperformed it on worst-case width and coverage. No `magnitude_bins`/`_magnitude_bin_edges` code remains.
  - Renamed `_quantile_bounds()` → `quantile_bounds()` (now a public cross-module API — `plots.py` calls it to label CI plots dynamically).
- `ml/src/dengue_ml/reporting/plots.py` — new `plot_oof_predictions_monthly(fold_predictions, model_name, outputs_dir, log_scale)`: monthly-grain counterpart of `plot_oof_predictions`, with a CI-level label computed dynamically from `quantile_bounds()` rather than hardcoded "95%".
- `ml/src/dengue_ml/training/train_pipeline.py` — calls `plot_oof_predictions_monthly` (log + linear) alongside the existing weekly OOF plots, so every future run generates both automatically.
- `ml/configs/model_training.yaml` — `xgboost.quantiles` changed from 95% (0.025/0.975) to 97.5% (0.0125/0.9875) per user's explicit choice. This is a global knob: it also changes the production quarterly forecast deliverable's band, not just the OOF plots.
- `ml/results/run_20260620_034022/figures/` — regenerated `oof_predictions_xrfm_cases_climate_monthly{,_log}.png` at 97.5% CI, plus two ad-hoc diagnostic bar charts (`coverage_with_vs_without_gap.png`, `coverage_old_vs_new_band.png`, `coverage_975_with_vs_without_gap.png`) comparing OOF coverage with/without the gap rows included in evaluation, and old (un-fixed) vs. new (gap-fixed) band width.

**Why:** The weekly OOF plot showed huge CI bands specifically during non-epidemic/trough periods, which looked backwards (peaks should be harder to predict, not troughs). Traced it to Vitória's 2021 data being fabricated zeros that get pooled into the shared low-regime calibration bucket used by *all four cities* — a handful of impossible residuals (predicted ~50-115, actual forced to 0 for a full year) were setting the lower-tail quantile for every city's quiet-season weeks. Excluding those rows from calibration (while keeping them visible as real history) dropped the worst-case monthly band width ratio from ~150x to ~3-4x with no loss to legitimate coverage elsewhere.

**Assumptions & trade-offs:**
- The gap window (2021-01-03 to 2021-12-05) is hardcoded as a list literal rather than auto-detected from the raw data — deliberate, since auto-detecting "implausible zero streaks" generically risks false positives on real quiet seasons; this is a known, manually-verified, one-off data anomaly.
- 97.5% CI was the user's explicit choice after comparing 80/90/95/97.5/99% empirically (coverage tracks the target monotonically, each landing 1-3pp below nominal — attributed to finite-sample LOFO calibration noise, year-to-year non-stationarity, a real systematic blind spot at epidemic-onset acceleration, and binomial sampling noise in the coverage estimate itself, not to the quantile choice).
- All 22 `ml/tests` pass throughout every step (gap exclusion, magnitude-bin add + revert, CI level change).

---
## Wire coverage-by-gap and residual-distribution diagnostics into every training run — 2026-06-20 16:21

**Goal:** Make the gap-vs-no-gap coverage comparison and a regime-conditional residual-distribution view permanent, automatic outputs of every training run, instead of one-off ad-hoc figures.

**Changes:**
- `ml/src/dengue_ml/reporting/results_tables.py` — new `coverage_by_gap_table(fold_predictions, model_name, outputs_dir)`: per-city + overall monthly OOF coverage, with vs. without `KNOWN_DATA_GAPS` rows included in the evaluation (they're already excluded from *calibration* everywhere; this table is about *evaluation*). Saves `coverage_by_gap.csv`.
- `ml/src/dengue_ml/reporting/plots.py`:
  - New `plot_coverage_by_gap(coverage_table, outputs_dir)` — bar chart of the above, saved as `coverage_by_gap.png`.
  - New `plot_residual_distribution(fold_predictions, model_name, outputs_dir)` — histogram of weekly log1p residuals split by CI regime (`growth_proxy >= REGIME_THRESHOLD`), with each regime's actual `quantile_bounds()` cutoffs overlaid as dashed vertical lines, gap rows excluded. Saved as `residual_distribution_{model_name}.png`.
  - `plot_proxy_comparison`'s nominal-coverage reference line was hardcoded to 95% — changed to read `quantile_bounds()` dynamically so it can't go stale again the way it already had after the CI level changed to 97.5%.
- `ml/src/dengue_ml/validation/conditional_residuals.py` — renamed `_is_known_data_gap` → `is_known_data_gap` (now genuinely cross-module public API, used by both `results_tables.py` and `plots.py`).
- `ml/src/dengue_ml/training/train_pipeline.py` — calls `coverage_by_gap_table`/`plot_coverage_by_gap`/`plot_residual_distribution` for `best_model` right after the existing OOF plots, and prints the coverage table to console.

**Why:** These started as one-off diagnostic figures generated by hand to answer specific questions during this session (does the gap fix actually help? what does the residual shape look like per regime?). Promoting them into the pipeline means every future run gets this visibility automatically rather than requiring another ad-hoc investigation.

**Assumptions & trade-offs:**
- `plot_residual_distribution` pools residuals across all 4 cities rather than faceting per-city (unlike the OOF time-series plots) — regime is the calibration axis that matters here, not city, and per-city faceting would mean re-deriving 4x smaller histograms for a question that's fundamentally about the shared regime-conditional calibration pool.
- Both new figures are generated only for `best_model`, matching the existing OOF-plot convention, not for every candidate model.
- All 25 `ml/tests` pass; new functions' output verified to match the manually-computed numbers from earlier in the session exactly.

---
## Add weekly/monthly year-over-year forecast deliverable + GIF frames — 2026-06-20 17:34

**Goal:** Extend the quarterly-only forecast deliverable to weekly and monthly grains, and replace the old "concatenated timeline + gray-dashed previous-year overlay" comparison plot with a single period-of-year axis (Q1-Q4 / Jan-Dec / W1-W52) overlaying last year's actuals, last year's model OOF estimate (with its own CI), and this year's forecast (with its own horizon-aware CI) — plus sequential per-point frames assembled into a GIF for each grain.

**Changes:**
- `ml/src/dengue_ml/validation/autoregressive_cv.py` — `_run_one_fold` now also tags each autoregressive-rollout row with `month_position` (calendar month, exact since outer test windows start Jan 1, same property `quarter_position` already relied on) and `week_position` (the rollout's own 1..horizon step count, since ISO week numbering has 52/53-week edge cases calendar month/quarter don't).
- `ml/src/dengue_ml/validation/conditional_residuals.py`:
  - New `aggregate_oof_to_quarterly` (quarterly twin of `aggregate_oof_to_monthly`) — feeds `assign_loFo_conditional_ci` for the redesigned plot's "previous year estimated cases" line at quarterly grain.
  - New `compute_horizon_bucketed_monthly_residual_quantile_table` / `compute_horizon_bucketed_weekly_residual_quantile_table` — siblings of the existing quarterly one, bucketed by `month_position` (1..12) / `week_position` (1..52) instead of `quarter_position` (1..4).
  - `apply_horizon_bucketed_quantile_table`'s position parameter renamed from `quarter_position` to `position` (purely a rename — it was already generic, just mislabeled); all existing positional callers unaffected.
- `ml/src/dengue_ml/forecasting/quarterly_aggregation.py` — new `aggregate_weekly_forecast_to_monthly` / `aggregate_weekly_history_to_monthly`, monthly twins of the existing quarterly functions. Existing quarterly function names/columns left untouched (the FastAPI app's `app/src/dengue_app/data.py` imports them directly).
- `ml/src/dengue_ml/reporting/plots.py`:
  - New `plot_forecast_year_over_year(period_labels, prev_actual, prev_oof, forecast, prev_year, forecast_year, outputs_dir, ...)` — blue dotted "{prev_year} Historical Data" (no CI), red dotted "{prev_year} Estimated Cases" (model's own LOFO-conditional CI), green solid "{forecast_year} Forecast" (horizon-aware CI where available), all on one period-of-year x-axis. `reveal_n` parameter drops forecast/prev_oof rows past a given x_pos for frame-by-frame reveal.
  - New `plot_forecast_year_over_year_frames` — renders one frame per period, fixing each city's y-axis across all frames (`_city_ylim`) so the GIF doesn't rescale, then assembles `forecast.gif` via Pillow (already an indirect dependency through matplotlib, no new dependency added).
  - New `_savefig_into` helper — the existing `_savefig`/`_resolve_fig_dir` always append `/figures` to whatever directory is passed, which doesn't fit the new nested `figures/forecast/{week,month,quarter}/` layout; `_savefig_into` writes directly into a fully-specified directory instead.
- `ml/scripts/generate_forecasts.py` — after the existing quarterly horizon-aware block, assembles prev-year-actual/prev-year-OOF/forecast frames at all three grains and calls the new plot + frame/GIF functions, writing into `run_dir/figures/forecast/{quarterly,monthly,weekly}/`. Loads `fold_predictions.csv` and `fold_predictions_ar.csv` directly (rather than relying on variables from earlier in the script) so the block degrades gracefully (prints a skip message) if either file, or `horizon_quantiles`, is missing.

**Why:** User wanted forecast granularity finer than quarterly for a more detailed view, and a year-over-year comparison plot that reads more like "this year vs last year" than a sliding timeline, plus a way to build an animated reveal of the forecast for a presentation.

**Assumptions & trade-offs:**
- `forecast_year`/`prev_year` are derived once from `weekly_forecast_df["forecast_week"].min().year` (and `- 1`), not from the actual calendar coverage of every series — correct as long as the forecast starts at/near a calendar year boundary, which matches current production usage (`MAX_RELIABLE_WEEK` runs through Dec 28).
- Forecast-side x_pos uses a per-city rank of forecast period (1..N, consistent with how the horizon-bucketed CI itself is keyed), while previous-year x_pos uses the literal calendar period (quarter/month/ISO week) — intentionally asymmetric since one is "steps into the rollout" and the other is "calendar position," and in practice both line up because production forecasts start on a year boundary.
- Monthly/weekly horizon-bucketed tables are computed on-the-fly in `generate_forecasts.py` from `fold_predictions_ar.csv` rather than being precomputed and stored in `final_model.pkl` (unlike the existing quarterly `horizon_quantiles`) — avoids touching `training/final_train.py` / requiring a retrain; the computation itself is cheap (groupby + quantile).
- Verified with synthetic smoke tests (small fabricated fold/forecast data) for all three grains rather than a full pipeline re-run — a real pipeline run (`run_nested_cv.py` → `run_autoregressive_cv.py` → ...) was already in progress on a SLURM compute node at the time, and the remaining steps are CPU-heavy multi-fold model training inappropriate to duplicate on the shared login node. The on-disk `fold_predictions_ar.csv` from the most recent completed run predates the `month_position`/`week_position` columns, so the new monthly/weekly deliverable will only appear after the next full `run_autoregressive_cv.py` run (the currently-running pipeline, or a future one).

---
