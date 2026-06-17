"""Helpers for per-run output directories under ml/results/."""
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
    """Return the run directory recorded by the most recent make_run_dir() call."""
    if not LATEST_RUN_FILE.exists():
        raise FileNotFoundError(
            f"{LATEST_RUN_FILE} not found — run run_nested_cv.py first."
        )
    run_dir = Path(LATEST_RUN_FILE.read_text().strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} no longer exists.")
    return run_dir
