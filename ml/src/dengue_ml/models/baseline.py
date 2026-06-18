import numpy as np
import pandas as pd

from dengue_ml.config import TARGET, CITY_COL


def seasonal_naive_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Seasonal naïve: ŷ(city, M, year) = y(city, M, year-1).
    Falls back to city-month median across all training years if prior year is missing.
    Returns test_df rows with an added 'predicted' column (original scale).
    """
    train = train_df.copy()
    train["month"] = train["month_start"].dt.month

    # Build lookup: (city, month_start) → cases
    lookup = train.set_index([CITY_COL, "month_start"])[TARGET]

    test = test_df.copy()
    test["month"] = test["month_start"].dt.month
    test["year"]  = test["month_start"].dt.year

    # Fallback: city-month median over training
    medians = (
        train.groupby([CITY_COL, "month"])[TARGET]
        .median()
        .rename("median_cases")
    )

    preds = []
    for _, row in test.iterrows():
        city = row[CITY_COL]
        # Prior-year same month
        prior_m = row["month_start"] - pd.DateOffset(years=1)
        # Snap to month start
        prior_m = pd.Timestamp(prior_m.year, prior_m.month, 1)
        prior_m = pd.Period(prior_m, freq="M").to_timestamp()

        val = lookup.get((city, prior_m), np.nan)
        if np.isnan(val):
            val = medians.get((city, row["month"]), np.nan)
        preds.append(val)

    test["predicted"] = preds
    return test[[CITY_COL, "month_start", TARGET, "predicted"]]
