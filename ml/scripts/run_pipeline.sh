#!/bin/bash
#SBATCH --job-name=dengue_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --mem-per-cpu=8G

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_DIR="/cluster/scratch/gcardenal/dengue_forecasting_brazil"
SCRIPTS_DIR="${PROJECT_DIR}/ml/scripts"

cd "${PROJECT_DIR}" || { echo "ERROR: cannot cd to ${PROJECT_DIR}"; exit 1; }

# Create logs dir if needed
mkdir -p logs

# ── Shared run ID: all three steps write to the same results/ subfolder ───────
export DENGUE_RUN_ID="$(date +%Y%m%d_%H%M%S)"
echo "Run ID: ${DENGUE_RUN_ID}"
echo "Results will be written to: ${PROJECT_DIR}/ml/results/run_${DENGUE_RUN_ID}/"

# ── Activate the uv-managed virtual environment ───────────────────────────────
source "${PROJECT_DIR}/.venv/bin/activate"
echo "Python: $(which python)  ($(python --version))"
echo "Working dir: $(pwd)"
echo "Start: $(date)"
echo "─────────────────────────────────────────────────────────"

# ── Step 1: Nested cross-validation (heaviest step) ───────────────────────────
echo ""
echo ">>> STEP 1: Nested cross-validation"
python "${SCRIPTS_DIR}/run_nested_cv.py"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "ERROR: run_nested_cv.py failed with exit code ${STATUS}"
    exit ${STATUS}
fi
echo "<<< STEP 1 done at $(date)"

# ── Step 2: Autoregressive CV (horizon-aware CI calibration) -- second
# validation, separate from Step 1's nested CV. Needs Step 1's
# fold_metrics.csv/best_hyperparameters.csv/best_hyperparameters_clf.csv/
# selected_classifier.txt in the same run dir; reuses Step 1's already-tuned
# hyperparameters (no new search), parallelized across the 10 outer folds.
# Non-fatal: a failure here shouldn't block the production forecast in
# Steps 4-5, it just means the parallel horizon-aware deliverable is skipped.
echo ""
echo ">>> STEP 2: Autoregressive CV (horizon-aware CI calibration)"
python "${SCRIPTS_DIR}/run_autoregressive_cv.py"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "WARNING: run_autoregressive_cv.py failed with exit code ${STATUS} (continuing -- horizon-aware deliverable will be skipped)"
fi
echo "<<< STEP 2 done at $(date)"

# ── Step 3: Epidemic classifier CV + proxy comparison (evaluation-only; not
# wired into final_model/forecasts) -- needs Step 1's fold_predictions.csv in
# the same run dir, which is why it runs right after it. Non-fatal: a failure
# here shouldn't block the production forecast in Steps 4-5.
echo ""
echo ">>> STEP 3: Epidemic classifier CV + proxy comparison"
python "${SCRIPTS_DIR}/run_classifier_cv.py"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "WARNING: run_classifier_cv.py failed with exit code ${STATUS} (continuing -- evaluation-only step)"
fi
echo "<<< STEP 3 done at $(date)"

# ── Step 4: Train final model on all data ─────────────────────────────────────
echo ""
echo ">>> STEP 4: Train final model"
python "${SCRIPTS_DIR}/train_final_model.py"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "ERROR: train_final_model.py failed with exit code ${STATUS}"
    exit ${STATUS}
fi
echo "<<< STEP 4 done at $(date)"

# ── Step 5: Generate forecasts ────────────────────────────────────────────────
echo ""
echo ">>> STEP 5: Generate forecasts"
python "${SCRIPTS_DIR}/generate_forecasts.py"
STATUS=$?
if [ ${STATUS} -ne 0 ]; then
    echo "ERROR: generate_forecasts.py failed with exit code ${STATUS}"
    exit ${STATUS}
fi
echo "<<< STEP 5 done at $(date)"

echo ""
echo "─────────────────────────────────────────────────────────"
echo "Pipeline complete. Outputs in: ${PROJECT_DIR}/ml/results/run_${DENGUE_RUN_ID}/"
echo "End: $(date)"
