import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from xrfm import xRFM

from dengue_ml.config import RANDOM_SEED
from dengue_ml.training_config import load_training_config

# Cap torch's intra-op thread pool to the SLURM allocation so a single xRFM
# .fit() call doesn't oversubscribe cores via nested torch/BLAS thread pools.
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))

# rfm_params passed to xRFM is a nested {'model': {...}, 'fit': {...}} dict
# (see xrfm.xRFM.__init__'s default_rfm_params). model_training.yaml's
# xrfm.default_params/param_distributions are flat, so we split them here.
_MODEL_KEYS = {"kernel", "bandwidth", "exponent", "diag", "bandwidth_mode"}
_FIT_KEYS   = {"reg", "iters", "early_stop_rfm", "get_agop_best_model", "return_best_params"}


def get_default_xrfm_params() -> dict:
    params = dict(load_training_config()["xrfm"]["default_params"])
    params["random_state"] = RANDOM_SEED
    return params


def _split_rfm_params(flat: dict) -> tuple[dict, dict]:
    """Split a flat hyperparameter dict into xRFM's nested rfm_params shape."""
    model_kwargs = {k: v for k, v in flat.items() if k in _MODEL_KEYS}
    fit_kwargs   = {k: v for k, v in flat.items() if k in _FIT_KEYS}
    return model_kwargs, fit_kwargs


class _ScaledXRFM:
    """
    xRFM's Laplace kernel operates on raw Euclidean distance with a single
    global bandwidth, so it's highly sensitive to feature scale (the xRFM
    README explicitly recommends standardizing numerical columns) — unlike
    XGBoost's split-based trees, which are scale-invariant. This wrapper
    keeps that scaling internal to train_xrfm/predict_xrfm so every other
    call site can keep treating xRFM like XGBoost (DataFrame in, no
    preprocessing concerns).
    """

    def __init__(self, model: xRFM, scaler: StandardScaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self.scaler.transform(X.to_numpy(dtype="float32"))
        # Clip to a fixed z-score range before the kernel ever sees it. The
        # Laplace kernel/AGOP metric has no defined ceiling for inputs far
        # outside the training distribution -- unlike XGBoost's trees, which
        # can only ever output a value seen in some training leaf. This
        # matters specifically during autoregressive rollouts: each step's
        # prediction is fed back as next step's lag feature, so if a
        # prediction drifts even slightly out of range, the next step's
        # input drifts further, and a kernel without an extrapolation
        # ceiling can compound that into an unbounded blowup (observed:
        # predictions reaching ~10M cases against actuals in the hundreds).
        # No-op for in-distribution data (normal nested CV, early rollout
        # steps), since real features essentially never reach |z| = 5.
        X_scaled = np.clip(X_scaled, -5.0, 5.0)
        return self.model.predict(X_scaled)

    @property
    def feature_importances_(self) -> np.ndarray:
        """
        RFM has no split-gain importance like XGBoost; the analogous quantity
        is the diagonal of the learned AGOP (Mahalanobis) matrix M, which
        measures how much the kernel stretches each (standardized) input
        direction. Averaged across leaf RFMs (config has get_agop_best_model
        always True -- see model_training.yaml) and normalized to sum to 1,
        same convention as XGBoost's feature_importances_, so the existing
        reporting code works unmodified.
        """
        agops = self.model.collect_best_agops()
        diags = [agop if agop.ndim == 1 else torch.diagonal(agop) for agop in agops]
        importances = torch.stack(diags).mean(dim=0).cpu().numpy()
        return importances / importances.sum()


def train_xrfm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict | None = None,
) -> _ScaledXRFM:
    """
    Train xRFM on log1p(target). Unlike XGBoost, xRFM's fit() API requires a
    held-out X_val/y_val (used internally for early stopping / best-iteration
    selection) — callers must carve one out of the training window before
    calling this (see nested_cv.py / final_train.py for the val-tail logic).

    Features are standardized (fit on X_train only, applied to X_train/X_val)
    before fitting — see _ScaledXRFM.
    """
    flat = {**get_default_xrfm_params(), **(params or {})}
    random_state = flat.pop("random_state", RANDOM_SEED)
    model_kwargs, fit_kwargs = _split_rfm_params(flat)

    scaler = StandardScaler()
    X_tr_scaled  = scaler.fit_transform(X_train.to_numpy(dtype="float32"))
    X_val_scaled = scaler.transform(X_val.to_numpy(dtype="float32"))

    model = xRFM(
        rfm_params={"model": model_kwargs, "fit": fit_kwargs},
        device="cpu",
        tuning_metric="mse",
        categorical_info=None,
        random_state=random_state,
        verbose=False,
    )
    model.fit(
        X_tr_scaled, y_train.to_numpy(dtype="float32"),
        X_val_scaled, y_val.to_numpy(dtype="float32"),
    )
    return _ScaledXRFM(model, scaler)


def predict_xrfm(model: _ScaledXRFM, X_test: pd.DataFrame) -> np.ndarray:
    """Predict; output is in log1p scale — caller applies np.expm1."""
    return np.asarray(model.predict(X_test)).ravel()
