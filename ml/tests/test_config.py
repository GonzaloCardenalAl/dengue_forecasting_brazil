import pandas as pd

from dengue_ml.config import compute_max_reliable_week


def test_compute_max_reliable_week_falls_back_a_quarter_when_latest_quarter_too_young():
    # now - 13wk = 2026-03-21, still inside Q1 2026 (which ends 2026-03-31) --
    # Q1 hasn't had its full 13-week convergence window yet, so the cutoff
    # must fall back to the end of Q4 2025, not just "13 weeks before now".
    now = pd.Timestamp("2026-06-20")

    result = compute_max_reliable_week(now=now)

    assert result == pd.Timestamp("2025-12-28")  # last Sunday on/before 2025-12-31
    assert result.dayofweek == 6  # Sunday
    # The forecast's first week (cutoff + 7 days) must be the very first
    # Sunday of the next quarter (Q1 2026) -- no partial leading quarter.
    first_forecast_week = result + pd.Timedelta(days=7)
    assert first_forecast_week == pd.Timestamp("2026-01-04")
    assert pd.Period(first_forecast_week, freq="Q") == pd.Period("2026Q1")


def test_compute_max_reliable_week_quarter_boundary_aligns_with_forecast_start():
    now = pd.Timestamp("2026-01-15")

    result = compute_max_reliable_week(now=now)

    assert result == pd.Timestamp("2025-09-28")  # last Sunday on/before Q3 2025's end
    first_forecast_week = result + pd.Timedelta(days=7)
    assert pd.Period(first_forecast_week, freq="Q") == pd.Period("2025Q4")
