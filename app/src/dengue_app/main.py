from fastapi import FastAPI, HTTPException, Query

from dengue_app import data
from dengue_app.city_coords import CITY_COORDS
from dengue_app.risk import RECOMMENDATIONS

app = FastAPI(title="DORA API", description="Dengue Outbreak Response Assistant")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _validate_city(city: str | None) -> None:
    if city is not None and city not in CITY_COORDS:
        raise HTTPException(status_code=404, detail=f"Unknown city '{city}'. Known cities: {list(CITY_COORDS)}")


@app.get("/cities")
def cities() -> list[dict]:
    return [{"city_name": name, **coords} for name, coords in CITY_COORDS.items()]


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
