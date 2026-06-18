import pandas as pd

from dengue_ml.validation.time_splits import make_inner_splits, make_outer_splits


def _monthly_df(start: str, periods: int, cities: list[str]) -> pd.DataFrame:
    months = pd.date_range(start=start, periods=periods, freq="MS")
    return pd.DataFrame(
        [
            {"city_name": city, "month_start": m, "casos_est": 1.0}
            for m in months
            for city in cities
        ]
    )


def test_make_outer_splits_train_test_boundary():
    df = _monthly_df("2020-01-01", periods=36, cities=["A", "B"])
    cutoff = pd.Timestamp("2021-12-01")

    splits = make_outer_splits(df, cutoffs=[cutoff], horizon=12)

    assert len(splits) == 1
    train_df, test_df = splits[0]
    assert (train_df["month_start"] <= cutoff).all()
    assert (test_df["month_start"] > cutoff).all()
    assert sorted(test_df["month_start"].unique()) == list(
        pd.date_range("2022-01-01", periods=12, freq="MS")
    )


def test_make_outer_splits_drops_cutoffs_without_enough_future_data():
    df = _monthly_df("2020-01-01", periods=13, cities=["A"])
    # Only 1 month remains after this cutoff, but horizon=12 is required.
    cutoff = pd.Timestamp("2020-12-01")

    splits = make_outer_splits(df, cutoffs=[cutoff], horizon=12)

    assert splits == []


def test_make_inner_splits_produces_requested_number_of_folds():
    df = _monthly_df("2015-01-01", periods=60, cities=["A"])

    splits = make_inner_splits(df, horizon=12, n_splits=2)

    assert len(splits) == 2
    for inner_train, inner_val in splits:
        assert len(inner_val) == 12
        assert inner_train["month_start"].max() < inner_val["month_start"].min()
