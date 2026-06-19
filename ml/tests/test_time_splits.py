import pandas as pd

from dengue_ml.validation.time_splits import make_inner_splits, make_outer_splits


def _weekly_df(dates: pd.DatetimeIndex, cities: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"city_name": city, "week_start": d, "casos_est": 1.0}
            for d in dates
            for city in cities
        ]
    )


def test_make_outer_splits_train_test_boundary():
    dates = pd.date_range("2020-01-05", periods=156, freq="W-SUN")  # 3 years
    df = _weekly_df(dates, cities=["A", "B"])
    cutoff = pd.Timestamp("2021-12-31")

    splits = make_outer_splits(df, cutoffs=[cutoff])

    assert len(splits) == 1
    train_df, test_df = splits[0]
    window_end = cutoff + pd.DateOffset(years=1)
    assert (train_df["week_start"] <= cutoff).all()
    assert (test_df["week_start"] > cutoff).all()
    assert (test_df["week_start"] <= window_end).all()

    expected_weeks = [d for d in dates if cutoff < d <= window_end]
    assert sorted(test_df["week_start"].unique()) == expected_weeks


def test_make_outer_splits_drops_cutoffs_without_enough_future_data():
    dates = pd.date_range("2020-01-05", periods=53, freq="W-SUN")
    df = _weekly_df(dates, cities=["A"])
    # Only 1 week remains after this cutoff -- not enough for a full test window.
    cutoff = dates[-2]

    splits = make_outer_splits(df, cutoffs=[cutoff])

    assert splits == []


def test_make_outer_splits_includes_all_53_weeks_in_a_53_week_year():
    """
    Sunday-anchored weekly cadence produces 53 distinct week-starts in some
    calendar years (e.g. 2012, 2017, 2023 in the real InfoDengue data -- see
    config.py's OUTER_CUTOFFS comment) and 52 in most others, purely from
    where Jan 1 falls relative to the weekly grid -- not a fixed, predictable
    count. A Dec-31-to-Dec-31 date-arithmetic window must include every week
    that actually falls in that calendar year, 52 or 53, rather than
    silently truncating to a fixed count the way a row-slice would.
    """
    dates = pd.date_range("2010-01-03", periods=800, freq="W-SUN")
    year_of = pd.Series(dates).dt.year
    counts_by_year = year_of.groupby(year_of).size()
    year_53 = counts_by_year[counts_by_year == 53].index[0]
    assert year_53 > dates[0].year  # cutoff below needs a prior year to exist

    df = _weekly_df(dates, cities=["A"])
    cutoff = pd.Timestamp(f"{year_53 - 1}-12-31")

    splits = make_outer_splits(df, cutoffs=[cutoff])

    assert len(splits) == 1
    _, test_df = splits[0]
    test_weeks = test_df["week_start"].unique()
    assert len(test_weeks) == 53
    assert set(pd.Series(test_weeks).dt.year) == {year_53}


def test_make_inner_splits_produces_requested_number_of_folds():
    dates = pd.date_range("2015-01-04", periods=52 * 5, freq="W-SUN")
    df = _weekly_df(dates, cities=["A"])

    splits = make_inner_splits(df, n_splits=2)

    assert len(splits) == 2
    for inner_train, inner_val in splits:
        assert len(inner_val) >= 52
        assert inner_train["week_start"].max() < inner_val["week_start"].min()
