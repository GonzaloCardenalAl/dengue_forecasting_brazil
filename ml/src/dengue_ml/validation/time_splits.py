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

    train_df: all rows with month_start <= cutoff
    test_df:  the next `horizon` distinct months after cutoff (all cities)
    """
    splits = []
    all_months = sorted(df["month_start"].unique())

    for cutoff in cutoffs:
        train_mask = df["month_start"] <= cutoff
        # Find which months come after the cutoff
        test_months = [m for m in all_months if m > cutoff][:horizon]
        if len(test_months) < horizon:
            continue  # not enough future data
        test_mask = df["month_start"].isin(test_months)
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
    all_months = sorted(train_df["month_start"].unique())
    n_m = len(all_months)

    # We need at least horizon val months + some training
    min_train_m = max(horizon * 2, 24)  # at least 2 years of training in inner loop
    splits = []

    # Generate cutoff indices from the end, stepping by horizon
    for i in range(n_splits, 0, -1):
        val_end_idx   = n_m - (i - 1) * horizon - 1
        val_start_idx = val_end_idx - horizon + 1
        if val_start_idx <= min_train_m:
            continue
        cutoff_m   = all_months[val_start_idx - 1]
        val_months = all_months[val_start_idx: val_end_idx + 1]

        inner_train = train_df[train_df["month_start"] <= cutoff_m].copy()
        inner_val   = train_df[train_df["month_start"].isin(val_months)].copy()

        if len(inner_train) > 0 and len(inner_val) == horizon * train_df[
            "city_name"
        ].nunique():
            splits.append((inner_train, inner_val))

    return splits
