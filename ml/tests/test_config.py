import pandas as pd

from dengue_ml.config import compute_max_reliable_week


def test_compute_max_reliable_week_snaps_to_sunday_and_lags_13_weeks():
    now = pd.Timestamp("2026-06-20")  # a Saturday

    result = compute_max_reliable_week(now=now)

    assert result.dayofweek == 6  # Sunday
    assert (now - result).days >= 13 * 7
    assert (now - result).days < 14 * 7


def test_compute_max_reliable_week_is_idempotent_on_a_sunday():
    now = pd.Timestamp("2026-06-21")  # already a Sunday

    result = compute_max_reliable_week(now=now)

    assert result.dayofweek == 6
    assert (now - result).days == 13 * 7
