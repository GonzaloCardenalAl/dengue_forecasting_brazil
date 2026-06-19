import numpy as np
import pandas as pd

from dengue_ml.config import TARGET, CITY_COL


def seasonal_naive_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Seasonal naïve: ŷ(city, iso_week, iso_year) = y(city, iso_week, iso_year-1).
    Falls back to city-iso_week median across all training years if the prior
    year's week is missing (e.g. forecasting into a week-53 target when the
    prior year only had 52 weeks).

    Keyed by ISO (year, week) via `.dt.isocalendar()` rather than a
    `DateOffset(years=1)` + snap-to-period trick (which has no clean weekly
    equivalent, unlike the monthly case) -- isocalendar() also correctly
    reassigns late-Dec/early-Jan boundary weeks to their true ISO year.

    Returns test_df rows with an added 'predicted' column (original scale).
    """
    train = train_df.copy()
    train_iso = train["week_start"].dt.isocalendar()
    train["iso_year"], train["iso_week"] = train_iso["year"], train_iso["week"]

    test = test_df.copy()
    test_iso = test["week_start"].dt.isocalendar()
    test["iso_year"], test["iso_week"] = test_iso["year"], test_iso["week"]
    test["prior_iso_year"] = test["iso_year"] - 1

    # Lookup: (city, iso_year, iso_week) -> cases
    lookup = train.set_index([CITY_COL, "iso_year", "iso_week"])[TARGET]
    # Fallback: city-iso_week median over training
    medians = (
        train.groupby([CITY_COL, "iso_week"])[TARGET]
        .median()
        .rename("median_cases")
    )

    keys = list(zip(test[CITY_COL], test["prior_iso_year"], test["iso_week"]))
    preds = lookup.reindex(keys).to_numpy()

    fallback_keys = list(zip(test[CITY_COL], test["iso_week"]))
    fallback = medians.reindex(fallback_keys).to_numpy()
    preds = np.where(np.isnan(preds), fallback, preds)

    test["predicted"] = preds
    return test[[CITY_COL, "week_start", TARGET, "predicted"]]
