import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from dengue_ml.config import RANDOM_SEED
from dengue_ml.training_config import load_training_config


def get_default_logreg_params() -> dict:
    params = dict(load_training_config()["logreg"]["default_params"])
    params["random_state"] = RANDOM_SEED
    return params


def get_default_xgb_clf_params() -> dict:
    params = dict(load_training_config()["xgb_classifier"]["default_params"])
    params["random_state"] = RANDOM_SEED
    params["n_jobs"] = -1
    return params


def train_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> LogisticRegression:
    p = {**get_default_logreg_params(), **(params or {})}
    model = LogisticRegression(**p)
    model.fit(X_train, y_train)
    return model


def predict_proba_logreg(model: LogisticRegression, X_test: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X_test)[:, 1]


def train_xgb_clf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict | None = None,
) -> xgb.XGBClassifier:
    """scale_pos_weight defaults to this fold's train-set class ratio (XGBoost
    has no class_weight="balanced" equivalent), overridable via params."""
    p = {**get_default_xgb_clf_params(), **(params or {})}
    if "scale_pos_weight" not in (params or {}):
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        p["scale_pos_weight"] = (n_neg / n_pos) if n_pos > 0 else 1.0
    model = xgb.XGBClassifier(**p)
    model.fit(X_train, y_train)
    return model


def predict_proba_xgb_clf(model: xgb.XGBClassifier, X_test: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X_test)[:, 1]
