"""Data integrity and location lookup."""

from __future__ import annotations

import pytest

from app.config import param
from app.datasets import (
    LocationOutOfRangeError,
    UnknownHobliError,
    haversine_km,
    list_hoblis,
    list_zones,
    load_hoblis,
    load_rainfall_monthly,
    site_context_for_hobli,
    site_context_for_point,
)

GW_CATEGORIES = {"safe", "semi_critical", "critical", "over_exploited"}


def test_every_hobli_has_twelve_months_of_rainfall():
    hoblis = load_hoblis()
    rainfall = load_rainfall_monthly()
    counts = rainfall.groupby("hobli_key").size()
    for key in hoblis["hobli_key"]:
        assert counts.get(key) == 12, f"{key} does not have 12 monthly rows"


def test_monthly_rainfall_sums_to_the_annual_total():
    hoblis = load_hoblis()
    rainfall = load_rainfall_monthly()
    totals = rainfall.groupby("hobli_key")["rainfall_mm"].sum()
    for _, row in hoblis.iterrows():
        annual = float(row["annual_rainfall_mm"])
        assert totals[row["hobli_key"]] == pytest.approx(annual, rel=0.01)


def test_hobli_values_are_physically_plausible():
    for _, row in load_hoblis().iterrows():
        assert 400 <= row["annual_rainfall_mm"] <= 2000
        assert 20 <= row["annual_rainy_days"] <= 120
        assert 0.5 <= row["infiltration_mm_per_hr"] <= 100
        assert 1 <= row["gw_depth_m"] <= 120
        assert row["gw_category"] in GW_CATEGORIES


def test_hobli_centroids_sit_inside_the_covered_box():
    box = param("bounding_box")
    for _, row in load_hoblis().iterrows():
        assert box["min_lat"] <= row["lat"] <= box["max_lat"]
        assert box["min_lon"] <= row["lon"] <= box["max_lon"]


def test_bbmp_zones_cover_198_wards():
    zones = list_zones()
    assert len(zones) == 8
    assert sum(z["wards"] for z in zones) == 198


def test_haversine_matches_a_known_distance():
    # Vidhana Soudha to Kempegowda airport: about 27 km as the crow flies, against
    # roughly 35 km by road.
    distance = haversine_km(12.9794, 77.5912, 13.1986, 77.7066)
    assert 26 < distance < 29


def test_haversine_matches_a_degree_of_longitude():
    # One degree of longitude at Bengaluru's latitude is about 108 km.
    assert haversine_km(12.9716, 77.5946, 12.9716, 78.5946) == pytest.approx(108, abs=1)


def test_point_lookup_finds_the_nearest_hobli():
    site = site_context_for_point(12.9352, 77.6245)
    assert site.district == "Bengaluru Urban"
    assert site.distance_to_centroid_km < 15
    assert len(site.monthly_rainfall) == 12


def test_point_lookup_attaches_the_bbmp_zone():
    site = site_context_for_point(12.9716, 77.5946)
    assert site.bbmp_zone in {z["zone"] for z in list_zones()}
    assert site.population_density_per_km2 > 0


def test_point_outside_bengaluru_is_rejected():
    with pytest.raises(LocationOutOfRangeError):
        site_context_for_point(19.0760, 72.8777)  # Mumbai


def test_hobli_lookup_is_case_insensitive():
    assert site_context_for_hobli("varthur").hobli == "Varthur"


def test_unknown_hobli_is_rejected():
    with pytest.raises(UnknownHobliError):
        site_context_for_hobli("Nowhere")


def test_hobli_list_is_complete():
    assert len(list_hoblis()) == len(load_hoblis())


def test_reference_data_is_flagged_as_an_estimate(jayanagar):
    assert jayanagar.data_grade == "reference_estimate"
    assert any("reference estimates" in note for note in jayanagar.notes)
