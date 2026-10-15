"""HTTP API behaviour."""

from __future__ import annotations

BASE_REQUEST = {
    "lat": 12.9352,
    "lon": 77.6245,
    "roof_area_m2": 120,
    "roof_type": "rcc_flat",
    "open_area_m2": 40,
    "household_size": 4,
    "water_source": "cauvery_and_borewell",
    "has_borewell": True,
    "plot_area_m2": 223,
}


def test_health_reports_loaded_data(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["hoblis_loaded"] > 0
    assert body["zones_loaded"] == 8


def test_parameters_drive_the_form(client):
    body = client.get("/api/parameters").json()
    assert "rcc_flat" in body["roof_types"]
    assert "paved_concrete" in body["open_area_types"]
    assert body["engine_modes"] == ["auto", "rules", "ml"]
    assert body["defaults"]["lpcd"] == 135
    assert len(body["structure_labels"]) == 6


def test_hoblis_endpoint_lists_centroids(client):
    body = client.get("/api/hoblis").json()
    assert body["count"] == len(body["hoblis"])
    first = body["hoblis"][0]
    assert {"hobli", "taluk", "lat", "lon", "annual_rainfall_mm"} <= set(first)


def test_zones_endpoint(client):
    body = client.get("/api/zones").json()
    assert body["count"] == 8
    assert sum(z["wards"] for z in body["zones"]) == 198


def test_location_lookup_by_pin(client):
    body = client.get("/api/location", params={"lat": 12.9352, "lon": 77.6245}).json()
    assert body["district"] == "Bengaluru Urban"
    assert len(body["monthly_rainfall"]) == 12


def test_location_lookup_by_hobli(client):
    body = client.get("/api/location", params={"hobli": "Varthur"}).json()
    assert body["hobli"] == "Varthur"


def test_location_outside_bengaluru_is_a_400(client):
    response = client.get("/api/location", params={"lat": 28.6139, "lon": 77.2090})
    assert response.status_code == 400
    assert "outside the covered area" in response.json()["detail"]


def test_unknown_hobli_is_a_400(client):
    response = client.get("/api/location", params={"hobli": "Atlantis"})
    assert response.status_code == 400
    assert "unknown hobli" in response.json()["detail"]


def test_location_without_arguments_is_a_400(client):
    assert client.get("/api/location").status_code == 400


def test_rainfall_endpoint_returns_twelve_months(client):
    body = client.get("/api/rainfall", params={"hobli": "Kengeri"}).json()
    assert len(body["monthly"]) == 12
    assert sum(m["rainfall_mm"] for m in body["monthly"]) > 0
    assert body["data_grade"]


def test_model_endpoint_always_answers(client):
    body = client.get("/api/model").json()
    assert "available" in body
    if body["available"]:
        assert body["labels"]
        assert body["feature_importances"]
    else:
        assert "reason" in body


def test_recommend_returns_a_full_report(client):
    body = client.post("/api/recommend", json=BASE_REQUEST).json()
    assert body["recommendation"]["label"]
    assert body["sizing"]
    assert body["cost"]["total_capital_cost_inr"] > 0
    assert len(body["water_balance"]["monthly"]) == 12


def test_recommend_accepts_a_hobli_instead_of_a_pin(client):
    payload = {k: v for k, v in BASE_REQUEST.items() if k not in ("lat", "lon")}
    payload["hobli"] = "Anekal"
    body = client.post("/api/recommend", json=payload).json()
    assert body["site"]["hobli"] == "Anekal"


def test_recommend_needs_a_location(client):
    payload = {k: v for k, v in BASE_REQUEST.items() if k not in ("lat", "lon")}
    assert client.post("/api/recommend", json=payload).status_code == 422


def test_recommend_rejects_a_lone_latitude(client):
    payload = dict(BASE_REQUEST)
    del payload["lon"]
    assert client.post("/api/recommend", json=payload).status_code == 422


def test_recommend_rejects_a_bad_roof_type(client):
    payload = dict(BASE_REQUEST, roof_type="thatch")
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 422
    assert "roof_type" in response.text


def test_recommend_rejects_a_zero_roof(client):
    assert client.post("/api/recommend", json=dict(BASE_REQUEST, roof_area_m2=0)).status_code == 422


def test_recommend_rejects_an_empty_household(client):
    assert (
        client.post("/api/recommend", json=dict(BASE_REQUEST, household_size=0)).status_code == 422
    )


def test_recommend_rejects_an_unknown_engine(client):
    assert client.post("/api/recommend", json=dict(BASE_REQUEST, engine="vibes")).status_code == 422


def test_recommend_outside_bengaluru_is_a_400(client):
    payload = dict(BASE_REQUEST, lat=13.0827, lon=80.2707)  # Chennai
    response = client.post("/api/recommend", json=payload)
    assert response.status_code == 400


def test_every_engine_mode_is_accepted(client):
    for mode in ("auto", "rules", "ml"):
        body = client.post("/api/recommend", json=dict(BASE_REQUEST, engine=mode)).json()
        assert body["meta"]["engine_mode"] == mode
        assert body["meta"]["engine_used"] in {"rules", "ml"}


def test_the_web_ui_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Rainwater Harvesting" in response.text


def test_static_assets_are_served(client):
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200


def test_openapi_schema_is_available(client):
    schema = client.get("/openapi.json").json()
    assert "/api/recommend" in schema["paths"]
