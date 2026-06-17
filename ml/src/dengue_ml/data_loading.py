import pandas as pd
from pathlib import Path


def load_dengue_data(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["data_iniSE"])
    df = df.rename(columns={"data_iniSE": "data_iniSE"})  # keep as-is
    return df


def load_sst_data(path: Path | str) -> pd.DataFrame:
    """Monthly SST indices (Niño 1+2, 3, 3.4, 4 and their anomalies)."""
    df = pd.read_csv(path)
    # Columns: YR  MON  NINO1+2  ANOM  NINO3  ANOM.1  NINO4  ANOM.2  NINO3.4  ANOM.3
    # Rename anomaly columns to descriptive names
    # CSV column order: YR MON NINO1+2 ANOM NINO3 ANOM.1 NINO4 ANOM.2 NINO3.4 ANOM.3
    df = df.rename(columns={
        "ANOM":   "nino12_anom",
        "ANOM.1": "nino3_anom",
        "ANOM.2": "nino4_anom",
        "ANOM.3": "nino34_anom",   # ← Niño 3.4 anomaly; primary ENSO index
        "YR":  "year",
        "MON": "month",
    })
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df.sort_values("date").reset_index(drop=True)


def load_roni_data(path: Path | str) -> pd.DataFrame:
    """RONI (Relative Oceanic Niño Index) data with overlapping 3-month seasons."""
    df = pd.read_csv(path)
    # Columns: SEAS  YR  ANOM
    # Map 3-month season to its centre month number
    _season_to_centre = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    df = df.rename(columns={"YR": "year", "SEAS": "season", "ANOM": "roni"})
    df["month"] = df["season"].map(_season_to_centre)
    # NDJ centre month 12 belongs to the year of the J at the end (year + 1 for NDJ)
    # But InfoDengue convention: year column is already the correct calendar year for the centre month
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
    return df[["date", "roni"]].sort_values("date").reset_index(drop=True)
