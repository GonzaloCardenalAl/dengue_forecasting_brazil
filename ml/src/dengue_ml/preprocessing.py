import pandas as pd
from functools import lru_cache
from pathlib import Path

from dengue_ml.config import (
    DATE_COL, CITY_COL, TARGET, MAX_RELIABLE_WEEK,
    DENGUE_FILE, SST_FILE, RONI_FILE,
)
from dengue_ml.data_loading import load_dengue_data, load_sst_data, load_roni_data


def _impute_climate_columns(
    df: pd.DataFrame, cols: tuple[str, ...] = ("tempmed", "umidmed")
) -> pd.DataFrame:
    """
    Climatological imputation: fill missing weekly climate values with that
    city's historical mean for the same ISO week-of-year.

    ~58/3340 weeks are missing, but 13 of those dates (the bulk: a Sep-Nov 2017
    block and a Nov-Dec 2020 block) are missing in ALL 4 cities simultaneously
    -- a nationwide weather-feed outage, not a city-specific sensor gap. So a
    cross-city donor can't fill them; same-week-of-year history can.
    """
    df = df.copy()
    week_of_year = df[DATE_COL].dt.isocalendar().week
    for col in cols:
        df[col] = df.groupby([CITY_COL, week_of_year])[col].transform(
            lambda s: s.fillna(s.mean())
        )
    return df


def prepare_sst_monthly(
    sst_df: pd.DataFrame, roni_df: pd.DataFrame
) -> pd.DataFrame:
    """Resample (already-monthly) SST and RONI onto a clean month-start index, then merge."""
    # Set date index for resampling
    sst = sst_df.set_index("date")[["nino34_anom"]].copy()
    roni = roni_df.set_index("date")[["roni"]].copy()

    sst_m  = sst.resample("MS").mean().reset_index().rename(columns={"date": "month_start"})
    roni_m = roni.resample("MS").mean().reset_index().rename(columns={"date": "month_start"})

    merged = pd.merge(sst_m, roni_m, on="month_start", how="outer").sort_values("month_start")
    return merged.reset_index(drop=True)


def merge_dengue_and_sst(
    dengue_w: pd.DataFrame, sst_m: pd.DataFrame
) -> pd.DataFrame:
    """
    Left-join weekly dengue data onto monthly SST (global signal, same for all
    cities). SST/RONI have no weekly source, so every week is bridged onto its
    containing calendar month's value (forward-fill-within-month) -- the
    simplest, least-assumption-heavy bridge; no invented intra-month
    interpolation.
    """
    df = dengue_w.copy()
    df["_month_key"] = pd.PeriodIndex(df["week_start"], freq="M").to_timestamp()
    merged = pd.merge(df, sst_m, left_on="_month_key", right_on="month_start", how="left")
    return merged.drop(columns=["_month_key", "month_start"])


def prepare_weekly_table(
    dengue_path: Path | str = DENGUE_FILE, apply_reliability_cutoff: bool = True
) -> pd.DataFrame:
    """
    Weekly InfoDengue panel at native resolution -- no aggregation needed
    (these columns already exist per-week in the raw data), just column
    selection plus the reporting-lag cutoff: weeks within ~13 weeks of the
    data pull are dropped as still-converging.

    apply_reliability_cutoff=False keeps the still-converging final week(s)
    -- for plotting only; training/forecasting must keep the default True.
    """
    df = load_dengue_data(dengue_path)
    df = _impute_climate_columns(df)
    df = df.copy()
    # InfoDengue leaves casos_est_max as NaN once a week's interval has fully
    # collapsed (converged) -- NaN there means "no separate max", not
    # "missing", so it should read as the point estimate, not be left blank.
    df["casos_est_max"] = df["casos_est_max"].fillna(df["casos_est"])

    df["week_start"] = df[DATE_COL]
    if apply_reliability_cutoff:
        df = df[df["week_start"] <= MAX_RELIABLE_WEEK].copy()

    df = df.sort_values([CITY_COL, "week_start"])
    # InfoDengue's own documented orange-alert criterion: Rt>1 sustained for
    # >=3 consecutive weeks. Empirically a much stronger leading indicator of
    # a real outbreak (75.5% recall, 57% precision against true nivel_inc==2
    # onsets) than the pre-epidemic alert level alone (100% recall, 11.7%
    # precision) -- worth exposing to the model as its own feature.
    # float (0.0/1.0), not bool: a bool column's .shift() in
    # add_weekly_lag_features introduces NaN for the first few rows per city,
    # which pandas can only represent by upcasting the whole column to
    # `object` dtype -- XGBoost then rejects it ("DataFrame.dtypes ... must
    # be int, float, bool or category").
    high_rt = df["p_rt1"] > 0.95
    df["sustained_rt"] = (
        high_rt.groupby(df[CITY_COL]).transform(lambda s: s.rolling(3).sum() >= 3)
    ).astype(float)

    df = df.rename(columns={"umidmed": "humidmed"})
    cols = [
        CITY_COL, "week_start", "casos_est", "casos_est_min", "casos_est_max",
        "p_inc100k", "tempmed", "humidmed", "transmissao", "receptivo",
        "nivel_inc", "pop", "Rt", "p_rt1", "sustained_rt",
    ]
    return df[cols].sort_values([CITY_COL, "week_start"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def get_monthly_sst_table(sst_path: Path | str = SST_FILE) -> pd.DataFrame:
    """
    Cached monthly Niño 3.4 series at its native resolution. SST is published
    with much less lag than dengue case counts, so the full series —
    including the most recent months — is usable as-is.
    """
    return load_sst_data(sst_path)


def prepare_model_table(
    dengue_path: Path | str = DENGUE_FILE,
    sst_path: Path | str = SST_FILE,
    roni_path: Path | str = RONI_FILE,
    apply_reliability_cutoff: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: raw CSVs → weekly model-ready table.

    apply_reliability_cutoff=False includes the still-converging final week(s)
    — for plotting only; training/forecasting must keep the default True.
    """
    sst_df  = load_sst_data(sst_path)
    roni_df = load_roni_data(roni_path)

    dengue_w = prepare_weekly_table(dengue_path, apply_reliability_cutoff)
    sst_m    = prepare_sst_monthly(sst_df, roni_df)
    table    = merge_dengue_and_sst(dengue_w, sst_m)

    return table.sort_values([CITY_COL, "week_start"]).reset_index(drop=True)
