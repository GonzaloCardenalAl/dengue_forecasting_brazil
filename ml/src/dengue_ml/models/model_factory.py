from dengue_ml.models.xgboost_models import get_default_xgb_params
from dengue_ml.models.xrfm_models import get_default_xrfm_params


def get_model_params(model_name: str, overrides: dict | None = None) -> dict:
    """Return default hyperparameters for the given model, with optional overrides."""
    if model_name.startswith("xgb"):
        params = get_default_xgb_params()
    elif model_name.startswith("xrfm"):
        params = get_default_xrfm_params()
    elif model_name == "sarima":
        params = {
            "order":         (1, 1, 1),
            "fourier_order": 2,
        }
    else:
        params = {}

    if overrides:
        params.update(overrides)
    return params
