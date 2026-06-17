"""Loader for ml/configs/model_training.yaml — hyperparameter grids, distributions,
and search settings, kept out of code so they can be tuned without a code change."""
from functools import lru_cache
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "model_training.yaml"


@lru_cache(maxsize=1)
def load_training_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
