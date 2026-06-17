#!/usr/bin/env python
"""Run nested rolling cross-validation. Creates a new timestamped results folder."""
import os
from dengue_ml.run_dir import make_run_dir
from dengue_ml.training.train_pipeline import run_training_pipeline

if __name__ == "__main__":
    # Allow the shell script to supply a shared run ID so all three scripts
    # write to the same folder (DENGUE_RUN_ID env var set by run_pipeline.sh).
    run_dir = make_run_dir(run_id=os.environ.get("DENGUE_RUN_ID"))
    print(f"Run directory: {run_dir}")
    run_training_pipeline(outputs_dir=run_dir)
