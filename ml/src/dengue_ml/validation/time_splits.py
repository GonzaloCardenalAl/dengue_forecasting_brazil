import pandas as pd

from dengue_ml.config import OUTER_CUTOFFS, N_INNER_FOLDS


def make_outer_splits(
    df: pd.DataFrame,
    cutoffs: list[pd.Timestamp] = OUTER_CUTOFFS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Return list of (train_df, test_df) for each outer fold.

    train_df: all rows with week_start <= cutoff
    test_df:  all rows in the following calendar year, i.e. in
              (cutoff, cutoff + 1 year] -- resolved via calendar-date
              arithmetic on the actual week_start timestamps, NOT a fixed
              row count, since calendar years have 52 or 53 weeks depending
              on the year (e.g. 2012/2017/2023 each have 53 in this data).
    """
    splits = []

    for cutoff in cutoffs:
        window_end = cutoff + pd.DateOffset(years=1)
        train_mask = df["week_start"] <= cutoff
        test_mask  = (df["week_start"] > cutoff) & (df["week_start"] <= window_end)

        # Require a full year of test data (>=52, not 53, so a 52-week year
        # isn't spuriously rejected); same role as the old horizon-count check.
        if test_mask.sum() < 52:
            continue  # not enough future data
        splits.append((df[train_mask].copy(), df[test_mask].copy()))

    return splits


def make_inner_splits(
    train_df: pd.DataFrame,
    n_splits: int = N_INNER_FOLDS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create rolling inner train/val splits from an outer training set.
    Each fold's validation window is exactly one calendar year, walking
    backward from the end of train_df for n_splits folds -- resolved via
    calendar-date arithmetic so 52- and 53-week years are both handled
    correctly with no off-by-one risk.
    """
    all_weeks   = sorted(train_df["week_start"].unique())
    train_start = all_weeks[0]
    train_end   = all_weeks[-1]

    min_train_end = train_start + pd.DateOffset(years=2)  # >=2 years of history floor
    n_cities = train_df["city_name"].nunique()
    splits = []

    for i in range(n_splits, 0, -1):
        val_end = train_end - pd.DateOffset(years=(i - 1))
        cutoff  = val_end - pd.DateOffset(years=1)

        if cutoff < min_train_end:
            continue

        inner_train = train_df[train_df["week_start"] <= cutoff].copy()
        inner_val   = train_df[
            (train_df["week_start"] > cutoff) & (train_df["week_start"] <= val_end)
        ].copy()

        # >=52, not ==, so a 52-week year isn't spuriously rejected.
        if len(inner_train) > 0 and len(inner_val) >= 52 * n_cities:
            splits.append((inner_train, inner_val))

    return splits
