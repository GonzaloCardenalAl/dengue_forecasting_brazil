import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query

from dengue_app import data
from dengue_app.city_coords import CITY_COORDS
from dengue_app.risk import RECOMMENDATIONS

load_dotenv()

app = FastAPI(title="DORA API", description="Dengue Outbreak Response Assistant")

ADMIN_TOKEN = os.environ.get("DENGUE_ADMIN_TOKEN")


def _check_admin_token(x_admin_token: str | None = Header(default=None)) -> None:
    if not ADMIN_TOKEN or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/admin/refresh")
def admin_refresh(_: None = Depends(_check_admin_token)) -> dict:
    """Fetch the latest InfoDengue data and regenerate the forecast using the
    currently-served model (no retraining)."""
    try:
        return data.refresh_and_reforecast()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def _validate_city(city: str | None) -> None:
    if city is not None and city not in CITY_COORDS:
        raise HTTPException(status_code=404, detail=f"Unknown city '{city}'. Known cities: {list(CITY_COORDS)}")


@app.get("/cities")
def cities() -> list[dict]:
    pop_by_city = data.load_city_population()
    return [
        {"city_name": name, **coords, "population": pop_by_city.get(name)}
        for name, coords in CITY_COORDS.items()
    ]


@app.get("/risk/recommendations")
def risk_recommendations() -> dict:
    return RECOMMENDATIONS


@app.get("/forecast/quarterly")
def forecast_quarterly(city: str | None = Query(default=None)) -> list[dict]:
    """Next 4 quarters per city: predicted_cases, 95% CI, epidemic probability, risk tier."""
    _validate_city(city)
    df = data.load_quarterly_forecast().rename(columns={"city": "city_name"})
    if city is not None:
        df = df[df["city_name"] == city]

    df["risk_tier"] = [
        data.risk_tier_for(row["city_name"], row["predicted_cases"], row["proxy_value"])
        for row in df.to_dict(orient="records")
    ]
    return data.records(df.sort_values(["city_name", "forecast_quarter"]))


@app.get("/backtest/quarterly")
def backtest_quarterly(
    city: str | None = Query(default=None),
    start: str | None = Query(default=None, description="ISO date, inclusive lower bound on quarter_start"),
    end: str | None = Query(default=None, description="ISO date, inclusive upper bound on quarter_start"),
) -> list[dict]:
    """Out-of-fold backtest: predicted vs actual cases per (city, quarter),
    bounded to whatever the nested CV folds cover. No prediction is returned
    for quarters outside that range."""
    _validate_city(city)
    df = data.load_backtest_quarterly()
    if city is not None:
        df = df[df["city_name"] == city]
    df = data.filter_by_date_range(df, "quarter_start", start, end)

    df["risk_tier"] = [
        data.risk_tier_for(row["city_name"], row["predicted_cases"], row["predicted_proba"])
        for row in df.to_dict(orient="records")
    ]
    return data.records(df.sort_values(["city_name", "quarter_start"]))


@app.get("/history/quarterly")
def history_quarterly(
    city: str | None = Query(default=None),
    start: str | None = Query(default=None, description="ISO date, inclusive lower bound on quarter_start"),
    end: str | None = Query(default=None, description="ISO date, inclusive upper bound on quarter_start"),
) -> list[dict]:
    """Actual quarterly case counts only -- the long-run context line behind
    the forecast/backtest charts."""
    _validate_city(city)
    df = data.load_history_quarterly()
    if city is not None:
        df = df[df["city_name"] == city]
    df = data.filter_by_date_range(df, "quarter_start", start, end)
    return data.records(df.sort_values(["city_name", "quarter_start"]))


@app.get("/history/seasonal-profile")
def history_seasonal_profile(city: str | None = Query(default=None)) -> list[dict]:
    """Per-season (Q4->Q1->Q2->Q3) summed incidence per city, for the annual
    quarterly-profile chart. is_epidemic_season flags seasons containing any
    nivel_inc==2 week."""
    _validate_city(city)
    return data.records(data.load_seasonal_profile(city))


@app.get("/history/monthly")
def history_monthly(city: str | None = Query(default=None)) -> list[dict]:
    """Actual monthly case counts -- higher-resolution twin of
    /history/quarterly, for the incidence bar chart."""
    _validate_city(city)
    df = data.load_history_monthly()
    if city is not None:
        df = df[df["city_name"] == city]
    return data.records(df.sort_values(["city_name", "month_start"]))


@app.get("/forecast/monthly")
def forecast_monthly(city: str | None = Query(default=None)) -> list[dict]:
    """Next ~12 months per city: predicted_cases (no CI -- see
    load_monthly_forecast). Higher-resolution twin of /forecast/quarterly,
    for the incidence bar chart."""
    _validate_city(city)
    df = data.load_monthly_forecast().rename(columns={"city": "city_name"})
    if city is not None:
        df = df[df["city_name"] == city]
    return data.records(df.sort_values(["city_name", "forecast_month"]))
