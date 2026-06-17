import pandas as pd
from functools import lru_cache
from pathlib import Path

from dengue_ml.config import (
    DATE_COL, CITY_COL, TARGET, MAX_RELIABLE_QUARTER,
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


def aggregate_dengue_to_quarterly(
    df: pd.DataFrame, apply_reliability_cutoff: bool = True
) -> pd.DataFrame:
    """
    Convert weekly InfoDengue panel to quarterly city-level data.

    apply_reliability_cutoff=False keeps the still-converging final quarter
    (excluded from training/forecasting) — used only for plotting, to show its
    genuinely wide casos_est_min/casos_est_max credible interval.
    """
    df = _impute_climate_columns(df)
    df = df.copy()
    # InfoDengue leaves casos_est_max as NaN once a week's interval has fully
    # collapsed (converged) -- NaN there means "no separate max", not "missing".
    # Summing NaN-as-0 across a quarter deflates the max below the point
    # estimate for any quarter mixing collapsed and still-converging weeks.
    df["casos_est_max"] = df["casos_est_max"].fillna(df["casos_est"])

    # quarter_start = first day of the calendar quarter containing each week
    df["quarter_start"] = pd.PeriodIndex(df[DATE_COL], freq="Q").to_timestamp()

    agg = (
        df.groupby([CITY_COL, "quarter_start"], sort=True)
        .agg(
            casos_est=("casos_est", "sum"),
            casos_est_min=("casos_est_min", "sum"),
            casos_est_max=("casos_est_max", "sum"),
            p_inc100k=("p_inc100k", "mean"),
            tempmed=("tempmed", "mean"),
            humidmed=("umidmed", "mean"),
            transmissao=("transmissao", "mean"),
            receptivo=("receptivo", "mean"),
            nivel_inc=("nivel_inc", "max"),
            pop=("pop", "last"),  # time-varying; captures 2023 census step-change
        )
        .reset_index()
    )

    if apply_reliability_cutoff:
        # Drop unreliable (still-converging) recent quarter
        agg = agg[agg["quarter_start"] <= MAX_RELIABLE_QUARTER].copy()
    return agg.sort_values([CITY_COL, "quarter_start"]).reset_index(drop=True)


def prepare_sst_quarterly(
    sst_df: pd.DataFrame, roni_df: pd.DataFrame
) -> pd.DataFrame:
    """Resample monthly SST and RONI to quarterly means, then merge."""
    # Set date index for resampling
    sst = sst_df.set_index("date")[["nino34_anom"]].copy()
    roni = roni_df.set_index("date")[["roni"]].copy()

    sst_q  = sst.resample("QS").mean().reset_index().rename(columns={"date": "quarter_start"})
    roni_q = roni.resample("QS").mean().reset_index().rename(columns={"date": "quarter_start"})

    merged = pd.merge(sst_q, roni_q, on="quarter_start", how="outer").sort_values("quarter_start")
    return merged.reset_index(drop=True)


def merge_dengue_and_sst(
    dengue_q: pd.DataFrame, sst_q: pd.DataFrame
) -> pd.DataFrame:
    """Left-join quarterly dengue data onto quarterly SST (global signal, same for all cities)."""
    return pd.merge(dengue_q, sst_q, on="quarter_start", how="left")


def prepare_weekly_table(dengue_path: Path | str = DENGUE_FILE) -> pd.DataFrame:
    """
    Raw weekly city-level series (casos_est, tempmed, umidmed, and InfoDengue's
    own surveillance status fields), used for week-level lag features. Same
    reporting-lag cutoff as the quarterly table: weeks belonging to the
    still-converging final quarter are dropped.
    """
    df = load_dengue_data(dengue_path)
    df = _impute_climate_columns(df)
    df["quarter_start"] = pd.PeriodIndex(df[DATE_COL], freq="Q").to_timestamp()
    df = df[df["quarter_start"] <= MAX_RELIABLE_QUARTER].copy()

    df = df.sort_values([CITY_COL, DATE_COL])
    # InfoDengue's own documented orange-alert criterion: Rt>1 sustained for
    # >=3 consecutive weeks. Empirically a much stronger leading indicator of
    # a real outbreak (75.5% recall, 57% precision against true nivel_inc==2
    # onsets) than the pre-epidemic alert level alone (100% recall, 11.7%
    # precision) -- worth exposing to the model as its own feature.
    high_rt = df["p_rt1"] > 0.95
    df["sustained_rt"] = (
        high_rt.groupby(df[CITY_COL]).transform(lambda s: s.rolling(3).sum() >= 3)
    )

    cols = [
        CITY_COL, DATE_COL, "casos_est", "tempmed", "umidmed",
        "transmissao", "receptivo", "nivel_inc", "Rt", "p_rt1", "sustained_rt",
    ]
    return df[cols].sort_values([CITY_COL, DATE_COL]).reset_index(drop=True)


@lru_cache(maxsize=1)
def get_weekly_table() -> pd.DataFrame:
    """Cached weekly table — avoids re-reading the CSV on every build_features() call."""
    return prepare_weekly_table()


@lru_cache(maxsize=1)
def get_monthly_sst_table(sst_path: Path | str = SST_FILE) -> pd.DataFrame:
    """
    Cached monthly Niño 3.4 series at its native resolution (no quarterly
    resampling). SST is published with much less lag than dengue case counts,
    so the full series — including the most recent months — is usable as-is.
    """
    return load_sst_data(sst_path)


def prepare_model_table(
    dengue_path: Path | str = DENGUE_FILE,
    sst_path: Path | str = SST_FILE,
    roni_path: Path | str = RONI_FILE,
    apply_reliability_cutoff: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline: raw CSVs → quarterly model-ready table.

    apply_reliability_cutoff=False includes the still-converging final quarter
    — for plotting only; training/forecasting must keep the default True.
    """
    dengue_df = load_dengue_data(dengue_path)
    sst_df    = load_sst_data(sst_path)
    roni_df   = load_roni_data(roni_path)

    dengue_q = aggregate_dengue_to_quarterly(dengue_df, apply_reliability_cutoff)
    sst_q    = prepare_sst_quarterly(sst_df, roni_df)
    table    = merge_dengue_and_sst(dengue_q, sst_q)

    return table.sort_values([CITY_COL, "quarter_start"]).reset_index(drop=True)
