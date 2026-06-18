import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from dengue_ml.config import TARGET, CITY_COL, FORECAST_HORIZON
from dengue_ml.validation.metrics import calculate_mae
from dengue_ml.training_config import load_training_config

_SEASONAL_PERIOD = load_training_config()["sarima"]["seasonal_period"]


def _iter_param_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def fit_sarima(
    series: pd.Series,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def forecast_sarima(
    fit_result,
    horizon: int = FORECAST_HORIZON,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (point_forecast, lower_95, upper_95) in log1p scale.
    Caller applies np.expm1 to convert to original scale.
    """
    pred = fit_result.get_forecast(steps=horizon)
    mean = pred.predicted_mean.values
    ci   = pred.conf_int(alpha=alpha).values  # shape (horizon, 2)
    return mean, ci[:, 0], ci[:, 1]


def tune_sarima(
    city_train_series: pd.Series,
    inner_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    param_grid: dict | None = None,
    city: str = "",
) -> tuple[tuple, tuple]:
    """
    Grid search over SARIMA parameters using inner rolling splits.
    Returns best (order, seasonal_order) by mean MAE across inner folds.
    city_train_series must be indexed by month_start.
    """
    if param_grid is None:
        param_grid = load_training_config()["sarima"]["param_grid"]

    best_mae    = float("inf")
    best_order  = (1, 1, 1)
    best_sorder = (0, 1, 0, _SEASONAL_PERIOD)

    for params in _iter_param_grid(param_grid):
        order  = (params["p"], params["d"],  params["q"])
        sorder = (params["P"], params["D"], params["Q"], _SEASONAL_PERIOD)

        fold_maes = []
        for inner_train_df, inner_val_df in inner_splits:
            try:
                train_s = (
                    inner_train_df[inner_train_df[CITY_COL] == city]
                    .set_index("month_start")[TARGET]
                    .sort_index()
                )
                val_s = (
                    inner_val_df[inner_val_df[CITY_COL] == city]
                    .set_index("month_start")[TARGET]
                    .sort_index()
                )
                if len(train_s) < 24 or len(val_s) == 0:
                    continue
                import numpy as np
                train_log = np.log1p(train_s)
                result = fit_sarima(train_log, order, sorder)
                preds_log, _, _ = forecast_sarima(result, horizon=len(val_s))
                preds = np.expm1(preds_log)
                fold_maes.append(calculate_mae(val_s.values, preds))
            except Exception:
                pass

        if fold_maes:
            mean_mae = float(np.mean(fold_maes))
            if mean_mae < best_mae:
                best_mae    = mean_mae
                best_order  = order
                best_sorder = sorder

    return best_order, best_sorder
