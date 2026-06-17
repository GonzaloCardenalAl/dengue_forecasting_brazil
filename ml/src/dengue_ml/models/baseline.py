import numpy as np
import pandas as pd

from dengue_ml.config import TARGET, CITY_COL


def seasonal_naive_forecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Seasonal naïve: ŷ(city, Q, year) = y(city, Q, year-1).
    Falls back to city-quarter median across all training years if prior year is missing.
    Returns test_df rows with an added 'predicted' column (original scale).
    """
    train = train_df.copy()
    train["quarter"] = train["quarter_start"].dt.quarter

    # Build lookup: (city, year, quarter) → cases
    lookup = train.set_index([CITY_COL, "quarter_start"])[TARGET]

    test = test_df.copy()
    test["quarter"] = test["quarter_start"].dt.quarter
    test["year"]    = test["quarter_start"].dt.year

    # Fallback: city-quarter median over training
    medians = (
        train.groupby([CITY_COL, "quarter"])[TARGET]
        .median()
        .rename("median_cases")
    )

    preds = []
    for _, row in test.iterrows():
        city = row[CITY_COL]
        # Prior-year same quarter
        prior_q = row["quarter_start"] - pd.DateOffset(years=1)
        # Snap to quarter start
        prior_q = pd.Timestamp(prior_q.year, prior_q.month, 1)
        prior_q = pd.Period(prior_q, freq="Q").to_timestamp()

        val = lookup.get((city, prior_q), np.nan)
        if np.isnan(val):
            val = medians.get((city, row["quarter"]), np.nan)
        preds.append(val)

    test["predicted"] = preds
    return test[[CITY_COL, "quarter_start", TARGET, "predicted"]]
