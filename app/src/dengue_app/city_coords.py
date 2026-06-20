"""Static lat/lon for the 4 forecasted cities -- not present anywhere in the
ml data (no GeoJSON/coordinates in the raw CSVs), so hardcoded here."""

CITY_COORDS = {
    "Vitória":        {"lat": -20.3155, "lon": -40.3128, "state": "ES"},
    "Belo Horizonte": {"lat": -19.9167, "lon": -43.9345, "state": "MG"},
    "Rio de Janeiro": {"lat": -22.9068, "lon": -43.1729, "state": "RJ"},
    "São Paulo":      {"lat": -23.5505, "lon": -46.6333, "state": "SP"},
}
