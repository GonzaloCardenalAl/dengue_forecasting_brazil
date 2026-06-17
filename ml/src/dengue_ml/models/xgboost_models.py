import numpy as np
import pandas as pd
import xgboost as xgb

from dengue_ml.config import RANDOM_SEED
from dengue_ml.training_config import load_training_config


def get_default_xgb_params() -> dict:
    params = dict(load_training_config()["xgboost"]["default_params"])
    params["random_state"] = RANDOM_SEED
    params["n_jobs"] = -1
    return params


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> xgb.XGBRegressor:
    """Train XGBoost on log1p(target). Returns fitted model."""
    p = {**get_default_xgb_params(), **(params or {})}
    model = xgb.XGBRegressor(**p)
    model.fit(X_train, y_train)
    return model


def predict_xgb(model: xgb.XGBRegressor, X_test: pd.DataFrame) -> np.ndarray:
    """Predict; output is in log1p scale — caller applies np.expm1."""
    return model.predict(X_test)
