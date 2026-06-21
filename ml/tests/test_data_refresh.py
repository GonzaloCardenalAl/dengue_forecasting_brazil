import pandas as pd

from dengue_ml.data_refresh import map_api_response_to_csv_schema, upsert_rows


def test_upsert_replaces_overlapping_weeks_not_duplicates():
    existing = pd.DataFrame({
        "city_name": ["Rio de Janeiro", "Rio de Janeiro"],
        "SE": [202552, 202553],
        "data_iniSE": ["2025-12-21", "2025-12-28"],
        "casos_est": [100.0, 50.0],
    })
    new = pd.DataFrame({
        "city_name": ["Rio de Janeiro", "Rio de Janeiro"],
        "SE": [202553, 202601],  # 202553 overlaps (revised), 202601 is new
        "data_iniSE": ["2025-12-28", "2026-01-04"],
        "casos_est": [80.0, 30.0],  # revised value for 202553
    })

    result = upsert_rows(existing, new)

    assert len(result) == 3  # no duplicate for 202553
    revised = result[(result["city_name"] == "Rio de Janeiro") & (result["SE"] == 202553)]
    assert len(revised) == 1
    assert revised["casos_est"].iloc[0] == 80.0  # replaced, not the stale 50.0


def test_upsert_returns_existing_unchanged_when_new_is_empty():
    existing = pd.DataFrame({
        "city_name": ["Vitória"], "SE": [202552], "data_iniSE": ["2025-12-21"], "casos_est": [1.0],
    })
    new = pd.DataFrame(columns=existing.columns)

    result = upsert_rows(existing, new)

    pd.testing.assert_frame_equal(result.reset_index(drop=True), existing.reset_index(drop=True))


def test_map_api_response_casts_numeric_strings_and_renames_municipio():
    raw = pd.DataFrame([{
        "data_iniSE": 1768089600000, "SE": 202602, "casos_est": 276.0,
        "casos_est_min": 276, "casos_est_max": 276, "casos": 276,
        "municipio_nome": "Rio de Janeiro", "pop": "6625849",
        "tempmin": "24.05", "Localidade_id": 0, "nivel": 3, "id": 1,
        "versao_modelo": "2026-06-17", "tweet": None, "Rt": 1.26, "p_rt1": 0.99,
        "p_inc100k": 4.17, "umidmax": "82.45", "receptivo": 1, "transmissao": 1,
        "nivel_inc": 1, "umidmed": "61.16", "umidmin": "37.63", "tempmed": "28.74",
        "tempmax": "34.0", "casprov": 74, "casprov_est": None,
        "casprov_est_min": None, "casprov_est_max": None, "casconf": None,
        "notif_accum_year": 572,
    }])

    mapped = map_api_response_to_csv_schema(raw, city="Rio de Janeiro")

    assert "municipio_nome" not in mapped.columns
    assert mapped["pop"].dtype == "float64"
    assert mapped["tempmin"].dtype == "float64"
    assert mapped["SE"].dtype == "int64"
    assert mapped["city_name"].iloc[0] == "Rio de Janeiro"
    assert mapped["state"].iloc[0] == "RJ"
    assert mapped["region"].iloc[0] == "Southeast"
    assert mapped["data_iniSE"].iloc[0] == "2026-01-11"


def test_map_api_response_passes_through_empty_dataframe():
    empty = pd.DataFrame()
    assert map_api_response_to_csv_schema(empty, city="Vitória").empty
