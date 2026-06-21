"""Refresh the raw InfoDengue CSV with the latest weekly data from the public
AlertaDengue API (https://info.dengue.mat.br/api/alertcity).

InfoDengue revises recent weeks upward as its nowcast model converges, so a
refresh must re-fetch a lookback window of already-stored weeks and REPLACE
them (upsert), not just append new ones.
"""
import pandas as pd
import requests

from dengue_ml.config import CITIES, DENGUE_FILE

API_BASE = "https://info.dengue.mat.br/api/alertcity"

# IBGE municipality geocodes for the 4 capitals tracked by this project.
# Confirmed live against the InfoDengue API (each returns the matching
# municipio_nome). Not exposed by the API itself -- it returns Localidade_id=0
# for all of these, so the mapping has to live here.
CITY_GEOCODES = {
    "Vitória":        (3205309, "ES"),
    "Belo Horizonte": (3106200, "MG"),
    "Rio de Janeiro": (3304557, "RJ"),
    "São Paulo":      (3550308, "SP"),
}
REGION = "Southeast"  # all 4 capitals above are in Brazil's Southeast region

CSV_COLUMNS = [
    "data_iniSE", "SE", "casos_est", "casos_est_min", "casos_est_max", "casos",
    "p_rt1", "p_inc100k", "Localidade_id", "nivel", "id", "versao_modelo",
    "tweet", "Rt", "pop", "tempmin", "umidmax", "receptivo", "transmissao",
    "nivel_inc", "umidmed", "umidmin", "tempmed", "tempmax", "casprov",
    "casprov_est", "casprov_est_min", "casprov_est_max", "casconf",
    "notif_accum_year", "city_name", "state", "region",
]
INT_COLS = [
    "SE", "casos_est_min", "casos", "Localidade_id", "nivel", "id",
    "receptivo", "nivel_inc", "notif_accum_year",
]
FLOAT_COLS = [
    "casos_est", "casos_est_max", "p_rt1", "p_inc100k", "tweet", "Rt", "pop",
    "tempmin", "umidmax", "transmissao", "umidmed", "umidmin", "tempmed",
    "tempmax", "casprov", "casprov_est", "casprov_est_min", "casprov_est_max",
    "casconf",
]


def determine_fetch_window(
    existing_df: pd.DataFrame, city: str, lookback_weeks: int = 20
) -> tuple[int, int, int, int]:
    """(ey_start, ew_start, ey_end, ew_end) for the API call.

    Steps back `lookback_weeks` from this city's last stored week to
    re-capture InfoDengue's revision window -- recent weeks get revised
    upward as the nowcast converges, so re-fetching them (and upserting) is
    required, not optional. Over-fetching already-converged weeks is
    harmless since upsert_rows() is idempotent.

    Raises ValueError if the city has no existing rows -- a full historical
    backfill is a separate, larger operation, out of scope for a refresh.
    """
    city_rows = existing_df[existing_df["city_name"] == city]
    if city_rows.empty:
        raise ValueError(
            f"No existing rows for '{city}' in {DENGUE_FILE} -- "
            "a full historical backfill is needed before incremental refresh can work."
        )
    last_date = pd.Timestamp(city_rows.loc[city_rows["SE"].idxmax(), "data_iniSE"])
    start = last_date - pd.Timedelta(weeks=lookback_weeks)
    now = pd.Timestamp.now()
    start_iso = start.isocalendar()
    now_iso = now.isocalendar()
    return (start_iso.year, start_iso.week, now_iso.year, now_iso.week)


def fetch_city_weeks(
    city: str, geocode: int, ey_start: int, ew_start: int, ey_end: int, ew_end: int
) -> pd.DataFrame:
    """Fetch one city's weekly rows from the InfoDengue API as a raw DataFrame
    (API's own field names, not yet mapped to the CSV schema)."""
    params = {
        "geocode": geocode,
        "disease": "dengue",
        "format": "json",
        "ey_start": ey_start,
        "ew_start": ew_start,
        "ey_end": ey_end,
        "ew_end": ew_end,
    }
    resp = requests.get(API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    records = resp.json()
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def map_api_response_to_csv_schema(df: pd.DataFrame, city: str) -> pd.DataFrame:
    """Map the InfoDengue API's JSON response onto the raw CSV's exact column
    set/order/dtypes.

    Differences confirmed live against the real API:
    - data_iniSE arrives as an epoch-millisecond int, not an ISO date string.
    - pop/tempmin/umidmax/umidmed/umidmin/tempmed/tempmax arrive as JSON
      strings, not numbers.
    - The API has municipio_nome instead of city_name/state/region.
    - casos_est_max can arrive as a plain JSON int once a week's interval has
      converged (no separate max), but the CSV column is float64 because NaN
      appears in not-yet-converged rows -- must cast, not just copy through.
    """
    if df.empty:
        return df

    df = df.copy()
    df["data_iniSE"] = pd.to_datetime(df["data_iniSE"], unit="ms").dt.strftime("%Y-%m-%d")
    df["city_name"] = city
    df["state"] = CITY_GEOCODES[city][1]
    df["region"] = REGION
    df = df.drop(columns=["municipio_nome"], errors="ignore")

    for col in INT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")
    for col in FLOAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df[CSV_COLUMNS]


def upsert_rows(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """Replace any existing (city_name, SE) rows that also appear in new_df
    (InfoDengue revised them) and append genuinely new ones -- never
    duplicate a (city_name, SE) pair."""
    if new_df.empty:
        return existing_df

    key_cols = ["city_name", "SE"]
    new_keys = pd.MultiIndex.from_frame(new_df[key_cols])
    existing_keys = pd.MultiIndex.from_frame(existing_df[key_cols])
    kept_existing = existing_df[~existing_keys.isin(new_keys)]

    merged = pd.concat([kept_existing, new_df], ignore_index=True)
    return merged.sort_values(["city_name", "data_iniSE"]).reset_index(drop=True)


def refresh_dengue_data(dry_run: bool = False) -> dict:
    """Fetch the latest weeks for all 4 capitals and upsert them into the raw
    CSV. Returns a JSON-safe summary dict; does not write to disk if
    dry_run=True."""
    existing_df = pd.read_csv(DENGUE_FILE)

    all_new = []
    per_city_fetched = {}
    for city in CITIES:
        geocode, _ = CITY_GEOCODES[city]
        ey_start, ew_start, ey_end, ew_end = determine_fetch_window(existing_df, city)
        raw = fetch_city_weeks(city, geocode, ey_start, ew_start, ey_end, ew_end)
        mapped = map_api_response_to_csv_schema(raw, city)
        per_city_fetched[city] = len(mapped)
        if not mapped.empty:
            all_new.append(mapped)

    new_df = pd.concat(all_new, ignore_index=True) if all_new else pd.DataFrame(columns=CSV_COLUMNS)
    updated_df = upsert_rows(existing_df, new_df)

    rows_before, rows_after = len(existing_df), len(updated_df)
    summary = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "rows_added": rows_after - rows_before,
        "weeks_fetched_per_city": per_city_fetched,
        "dry_run": dry_run,
    }

    if not dry_run:
        updated_df.to_csv(DENGUE_FILE, index=False)

    return summary
