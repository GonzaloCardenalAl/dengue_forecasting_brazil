import pandas as pd

from dengue_ml.validation.time_splits import make_inner_splits, make_outer_splits


def _quarterly_df(start: str, periods: int, cities: list[str]) -> pd.DataFrame:
    quarters = pd.date_range(start=start, periods=periods, freq="QS")
    return pd.DataFrame(
        [
            {"city_name": city, "quarter_start": q, "casos_est": 1.0}
            for q in quarters
            for city in cities
        ]
    )


def test_make_outer_splits_train_test_boundary():
    df = _quarterly_df("2020-01-01", periods=12, cities=["A", "B"])
    cutoff = pd.Timestamp("2021-10-01")

    splits = make_outer_splits(df, cutoffs=[cutoff], horizon=4)

    assert len(splits) == 1
    train_df, test_df = splits[0]
    assert (train_df["quarter_start"] <= cutoff).all()
    assert (test_df["quarter_start"] > cutoff).all()
    assert sorted(test_df["quarter_start"].unique()) == list(
        pd.date_range("2022-01-01", periods=4, freq="QS")
    )


def test_make_outer_splits_drops_cutoffs_without_enough_future_data():
    df = _quarterly_df("2020-01-01", periods=8, cities=["A"])
    # Only 1 quarter remains after this cutoff, but horizon=4 is required.
    cutoff = pd.Timestamp("2021-07-01")

    splits = make_outer_splits(df, cutoffs=[cutoff], horizon=4)

    assert splits == []


def test_make_inner_splits_produces_requested_number_of_folds():
    df = _quarterly_df("2015-01-01", periods=28, cities=["A"])

    splits = make_inner_splits(df, horizon=4, n_splits=2)

    assert len(splits) == 2
    for inner_train, inner_val in splits:
        assert len(inner_val) == 4
        assert inner_train["quarter_start"].max() < inner_val["quarter_start"].min()
