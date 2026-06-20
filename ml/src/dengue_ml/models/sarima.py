import itertools
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from dengue_ml.config import TARGET, CITY_COL, FORECAST_HORIZON
from dengue_ml.validation.metrics import calculate_mae
from dengue_ml.training_config import load_training_config


def _iter_param_grid(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def fourier_terms(dates: pd.DatetimeIndex, order: int) -> np.ndarray:
    """
    Exogenous Fourier regressors for the annual seasonal cycle, `order`
    harmonic pairs. Uses day-of-year/365.25 (not week-of-year mod 52) so the
    cycle doesn't drift against the ~52.18 weeks/year reality.
    """
    frac = dates.dayofyear.values / 365.25
    cols = []
    for k in range(1, order + 1):
        cols.append(np.sin(2 * np.pi * k * frac))
        cols.append(np.cos(2 * np.pi * k * frac))
    return np.column_stack(cols)


def fit_sarima(
    series: pd.Series,
    order: tuple[int, int, int],
    exog: np.ndarray | None = None,
) -> object:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series,
            exog=exog,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit(disp=False)


def forecast_sarima(
    fit_result,
    horizon: int = FORECAST_HORIZON,
    exog: np.ndarray | None = None,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (point_forecast, lower_95, upper_95) in log1p scale.
    Caller applies np.expm1 to convert to original scale.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = fit_result.get_forecast(steps=horizon, exog=exog)
        mean = pred.predicted_mean.values
        ci   = pred.conf_int(alpha=alpha).values  # shape (horizon, 2)
    return mean, ci[:, 0], ci[:, 1]


def tune_sarima(
    city_train_series: pd.Series,
    inner_splits: list[tuple[pd.DataFrame, pd.DataFrame]],
    param_grid: dict | None = None,
    city: str = "",
) -> tuple[tuple, int]:
    """
    Grid search over SARIMA (p, d, q) + Fourier harmonic order using inner
    rolling splits. Returns best (order, fourier_order) by mean MAE across
    inner folds. city_train_series must be indexed by week_start.
    """
    if param_grid is None:
        param_grid = load_training_config()["sarima"]["param_grid"]

    best_mae    = float("inf")
    best_order  = (1, 1, 1)
    best_fourier_order = 2

    for params in _iter_param_grid(param_grid):
        order = (params["p"], params["d"], params["q"])
        k     = params["fourier_order"]

        fold_maes = []
        for inner_train_df, inner_val_df in inner_splits:
            try:
                train_s = (
                    inner_train_df[inner_train_df[CITY_COL] == city]
                    .set_index("week_start")[TARGET]
                    .sort_index()
                )
                val_s = (
                    inner_val_df[inner_val_df[CITY_COL] == city]
                    .set_index("week_start")[TARGET]
                    .sort_index()
                )
                if len(train_s) < 104 or len(val_s) == 0:
                    continue
                train_log = np.log1p(train_s)
                train_exog = fourier_terms(train_s.index, k)
                val_exog   = fourier_terms(val_s.index, k)
                result = fit_sarima(train_log, order, exog=train_exog)
                preds_log, _, _ = forecast_sarima(result, horizon=len(val_s), exog=val_exog)
                preds = np.expm1(preds_log)
                fold_maes.append(calculate_mae(val_s.values, preds))
            except Exception:
                pass

        if fold_maes:
            mean_mae = float(np.mean(fold_maes))
            if mean_mae < best_mae:
                best_mae    = mean_mae
                best_order  = order
                best_fourier_order = k

    return best_order, best_fourier_order
