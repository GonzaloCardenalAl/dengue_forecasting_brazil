"""Helpers for per-run output directories under ml/results/."""
import os
from datetime import datetime
from pathlib import Path

from dengue_ml.config import RESULTS_DIR, LATEST_RUN_FILE


def make_run_dir(run_id: str | None = None) -> Path:
    """
    Create a new timestamped directory under ml/results/ and record it in
    latest_run.txt so subsequent pipeline steps find the same folder.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_FILE.write_text(str(run_dir))
    return run_dir


def get_latest_run_dir() -> Path:
    """Return the run directory recorded by the most recent make_run_dir() call.

    DENGUE_RUN_DIR, if set, overrides this -- lets API/dashboard consumers
    point at an arbitrary run dir (e.g. a demo fixture) without touching
    latest_run.txt, which live pipeline runs depend on for their own
    later steps.
    """
    override = os.environ.get("DENGUE_RUN_DIR")
    if override is not None:
        run_dir = Path(override)
        if not run_dir.exists():
            raise FileNotFoundError(f"DENGUE_RUN_DIR={run_dir} does not exist.")
        return run_dir

    if not LATEST_RUN_FILE.exists():
        raise FileNotFoundError(
            f"{LATEST_RUN_FILE} not found — run run_nested_cv.py first."
        )
    run_dir = Path(LATEST_RUN_FILE.read_text().strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} no longer exists.")
    return run_dir
