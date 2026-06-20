import numpy as np
import pandas as pd

from dengue_ml.features.feature_pipeline import _make_eval_window_deployment_realistic
from dengue_ml.features.forecast_proxies import climatological_val


def _weekly_df(dates: pd.DatetimeIndex, city: str, tempmed: float, nino34_anom) -> pd.DataFrame:
    return pd.DataFrame({
        "city_name": city,
        "week_start": dates,
        "casos_est": 1.0,
        "tempmed": tempmed,
        "humidmed": tempmed,  # reuse the same pattern for simplicity
        "nino34_anom": nino34_anom,
    })


def test_eval_window_climate_is_not_the_real_recorded_value():
    """
    A 1-step CV/hyperparameter-search eval split must never see the real
    tempmed/humidmed for the week it's forecasting -- the production forecast
    loop (forecasting/autoregressive.py) never has it either. Build a train
    history with a flat climate signal and an eval window with a sharply
    different real value, and confirm the substituted eval value is the
    train-history climatological estimate, not the injected real one.
    """
    train_dates = pd.date_range("2018-01-07", periods=104, freq="W-SUN")  # 2 years
    train_df = _weekly_df(train_dates, "Recife", tempmed=20.0, nino34_anom=0.1)

    eval_dates = pd.date_range("2020-01-05", periods=52, freq="W-SUN")
    eval_df = _weekly_df(eval_dates, "Recife", tempmed=99.0, nino34_anom=5.0)

    result = _make_eval_window_deployment_realistic(train_df, eval_df)

    # tempmed/humidmed are replaced with train_df's climatological (here flat
    # 20.0) value, not the injected 99.0 "real" eval value.
    assert (result["tempmed"] != 99.0).all()
    assert np.allclose(result["tempmed"], 20.0)
    assert np.allclose(result["humidmed"], 20.0)

    # nino34_anom is carried forward at train_df's last known value (0.1),
    # not the injected 5.0 "real" eval value, and is flat across the window.
    assert (result["nino34_anom"] != 5.0).all()
    assert np.allclose(result["nino34_anom"], 0.1)


def test_eval_window_climatological_value_follows_seasonal_pattern():
    """The substituted tempmed should track train history's own seasonal
    (week-of-year) pattern, not just a single flat average."""
    train_dates = pd.date_range("2018-01-07", periods=156, freq="W-SUN")  # 3 years
    woy = train_dates.isocalendar().week.to_numpy(dtype=float)
    seasonal_temp = 20.0 + 5.0 * np.sin(2 * np.pi * woy / 52.0)
    train_df = _weekly_df(train_dates, "Recife", tempmed=seasonal_temp, nino34_anom=0.0)

    eval_dates = pd.date_range("2021-01-03", periods=52, freq="W-SUN")
    eval_df = _weekly_df(eval_dates, "Recife", tempmed=-999.0, nino34_anom=0.0)

    result = _make_eval_window_deployment_realistic(train_df, eval_df)

    # Per-row ground truth via the same dependency the function under test
    # calls -- this isolates "does it call climatological_val correctly, per
    # row, against train_df" from re-deriving the seasonal math independently.
    expected = [climatological_val(train_df, "Recife", "tempmed", w) for w in eval_dates]
    np.testing.assert_allclose(result["tempmed"].to_numpy(), expected, atol=1e-9)
    # Genuinely seasonal, not a single flat value repeated.
    assert result["tempmed"].nunique() > 1


def test_eval_window_substitution_is_noop_without_climate_columns():
    train_df = pd.DataFrame({
        "city_name": ["Recife"], "week_start": [pd.Timestamp("2018-01-07")], "casos_est": [1.0],
    })
    eval_df = pd.DataFrame({
        "city_name": ["Recife"], "week_start": [pd.Timestamp("2020-01-05")], "casos_est": [1.0],
    })

    result = _make_eval_window_deployment_realistic(train_df, eval_df)

    pd.testing.assert_frame_equal(result, eval_df)
