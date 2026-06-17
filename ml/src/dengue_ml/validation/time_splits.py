import pandas as pd
from typing import Iterator

from dengue_ml.config import OUTER_CUTOFFS, FORECAST_HORIZON, N_INNER_FOLDS


def make_outer_splits(
    df: pd.DataFrame,
    cutoffs: list[pd.Timestamp] = OUTER_CUTOFFS,
    horizon: int = FORECAST_HORIZON,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Return list of (train_df, test_df) for each outer fold.

    train_df: all rows with quarter_start <= cutoff
    test_df:  the next `horizon` distinct quarters after cutoff (all cities)
    """
    splits = []
    all_quarters = sorted(df["quarter_start"].unique())

    for cutoff in cutoffs:
        train_mask = df["quarter_start"] <= cutoff
        # Find which quarters come after the cutoff
        test_quarters = [q for q in all_quarters if q > cutoff][:horizon]
        if len(test_quarters) < horizon:
            continue  # not enough future data
        test_mask = df["quarter_start"].isin(test_quarters)
        splits.append((df[train_mask].copy(), df[test_mask].copy()))

    return splits


def make_inner_splits(
    train_df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
    n_splits: int = N_INNER_FOLDS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create rolling inner train/val splits from an outer training set.
    Works backwards from the end of train_df to produce n_splits folds.
    """
    all_quarters = sorted(train_df["quarter_start"].unique())
    n_q = len(all_quarters)

    # We need at least horizon val quarters + some training
    min_train_q = max(horizon * 2, 8)  # at least 2 years of training in inner loop
    splits = []

    # Generate cutoff indices from the end, stepping by horizon
    for i in range(n_splits, 0, -1):
        val_end_idx   = n_q - (i - 1) * horizon - 1
        val_start_idx = val_end_idx - horizon + 1
        if val_start_idx <= min_train_q:
            continue
        cutoff_q   = all_quarters[val_start_idx - 1]
        val_quarters = all_quarters[val_start_idx: val_end_idx + 1]

        inner_train = train_df[train_df["quarter_start"] <= cutoff_q].copy()
        inner_val   = train_df[train_df["quarter_start"].isin(val_quarters)].copy()

        if len(inner_train) > 0 and len(inner_val) == horizon * train_df[
            "city_name"
        ].nunique():
            splits.append((inner_train, inner_val))

    return splits
