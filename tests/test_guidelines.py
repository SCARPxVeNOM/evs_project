"""Validation against standard rainwater harvesting design guidance.

This is the roadmap's "validation against standard RWH design guidelines" item.
Each test pins one number or rule that comes from outside this project - a BIS
standard, CGWB design practice, the Karnataka byelaw, or a published Bengaluru
rule of thumb - so that a change to the engine that drifts away from accepted
practice fails here rather than quietly shipping.

Sources are named in each test. Where a figure is a range rather than a single
value, the test asserts the range.
"""

from __future__ import annotations

import pytest

from app.config import param
from app.core.calculator import compute_harvest_potential, harvest_litres
from app.core.engine import HouseholdInputs, recommend
from app.core.sizing import minimum_pit_count, size_recharge_pit

# --------------------------------------------------------------------------- #
# The core estimate
# --------------------------------------------------------------------------- #


def test_harvest_formula_matches_the_published_relation():
    """IS 15797 / CGWB: harvest = rainfall x area x runoff coefficient."""
    assert harvest_litres(900, 150, 0.8) == pytest.approx(900 * 150 * 0.8)


def test_rcc_roof_runoff_coefficient_matches_cgwb():
    """CGWB's artificial-recharge manual gives 0.85 for an RCC roof."""
    assert param("runoff_coefficients", "roof", "rcc_flat") == pytest.approx(0.85, abs=0.02)


def test_tiled_roof_runoff_coefficient_matches_cgwb():
    """0.75 for tiled roofs in the same table."""
    assert param("runoff_coefficients", "roof", "tiled") == pytest.approx(0.75, abs=0.02)


def test_bengaluru_urban_hobli_mean_matches_the_ksndmc_district_normal():
    """
    KSNDMC puts the Bengaluru Urban district annual normal at 846 mm, so the mean
    of the district's hobli normals has to land on it.

    This is deliberately not checked against the IMD Bengaluru City normal of
    about 970 mm. That figure is a single observatory in the wet urban core, not
    a district-wide average, and the two differ by roughly 15% - conflating them
    is what made an earlier version of this test pass against invented rainfall.
    """
    from app.datasets import load_hoblis

    urban = load_hoblis()
    urban = urban[urban["district"] == "Bengaluru Urban"]
    assert urban["annual_rainfall_mm"].mean() == pytest.approx(846, abs=25)


def test_the_standard_bengaluru_house_yields_the_quoted_volume(jayanagar):
    """
    A 1200 sq ft (111 m2) RCC roof in Bengaluru is widely quoted as harvesting
    roughly 80,000 to 100,000 litres a year.

    That quote carries two assumptions: the IMD Bengaluru City normal of about
    970 mm, and gross runoff before any first-flush or filter loss. Checked on
    its own terms the engine reproduces it, so the assertion below validates the
    harvest relation rather than the rainfall table.
    """
    coeff = param("runoff_coefficients", "roof", "rcc_flat")
    assert 80_000 <= harvest_litres(970, 111.0, coeff) <= 100_000


def test_a_measured_hobli_normal_yields_less_than_the_popular_quote(jayanagar):
    """
    The same roof on this hobli's measured KSNDMC normal yields appreciably less
    than the folk figure, because the hobli is drier than the city observatory
    and because first-flush and filter losses are actually modelled. Being
    conservative here is the intended behaviour, so it is pinned.
    """
    harvest = compute_harvest_potential(jayanagar, 111.0, "rcc_flat", 0)
    assert harvest.annual_rainfall_mm < 970
    assert harvest.effective_annual_rainfall_mm < harvest.annual_rainfall_mm
    assert 55_000 <= harvest.annual_roof_litres <= 80_000


def test_first_flush_diversion_is_one_to_two_millimetres():
    """IS 15797 recommends diverting the first 1-2 mm of each event."""
    assert 1.0 <= param("system_losses", "first_flush_mm_per_event") <= 2.0


# --------------------------------------------------------------------------- #
# Demand
# --------------------------------------------------------------------------- #


def test_per_capita_demand_matches_is_1172():
    """IS 1172:1993 - 135 lpcd for towns with full flushing, 70 lpcd minimum."""
    assert param("demand", "bis_1172_full_flushing_lpcd") == 135
    assert param("demand", "bis_1172_minimum_lpcd") == 70
    assert param("demand", "default_lpcd") == 135


def test_non_potable_share_is_within_the_surveyed_range():
    """Household end-use surveys put flushing, washing and gardening at 50-65%."""
    assert 0.45 <= param("demand", "non_potable_share") <= 0.65


# --------------------------------------------------------------------------- #
# Storage sizing
# --------------------------------------------------------------------------- #


def test_storage_design_period_follows_domestic_practice():
    """CGWB and the BWSSB handbook size domestic storage for a 15-20 day spell."""
    assert 15 <= param("storage_design", "design_days") <= 20


def test_a_tank_is_never_larger_than_the_wettest_month_can_fill(jayanagar, typical_inputs):
    report = recommend(jayanagar, typical_inputs)
    storage = report["sizing"]["storage"]
    if storage and not storage["byelaw_minimum_litres"]:
        assert storage["required_litres"] <= storage["peak_month_harvest_litres"] + 1


def test_a_tank_never_holds_more_than_a_month_of_demand(jayanagar, typical_inputs):
    """Domestic storage beyond about a month of demand stops earning its cost."""
    report = recommend(jayanagar, typical_inputs)
    storage = report["sizing"]["storage"]
    demand = report["demand"]["daily_non_potable_litres"]
    assert storage["required_litres"] <= demand * 31


# --------------------------------------------------------------------------- #
# Recharge design
# --------------------------------------------------------------------------- #


def test_design_storm_matches_bengaluru_short_duration_rainfall():
    """The 1-hour, 2-year-return intensity for Bengaluru is around 50 mm/hr."""
    assert 40 <= param("recharge_design", "design_storm_mm_per_hr") <= 60


def test_filter_media_void_ratio_is_physically_sensible():
    """Graded boulder, gravel and coarse sand leave roughly a third as voids."""
    assert 0.25 <= param("recharge_design", "filter_media_void_ratio") <= 0.40


def test_cgwb_pit_density_of_one_per_hundred_square_metres_is_enforced():
    assert minimum_pit_count(100) == 1
    assert minimum_pit_count(300) == 3
    pit = size_recharge_pit(
        catchment_area_m2=300,
        runoff_coeff=0.85,
        infiltration_mm_per_hr=20,
        gw_depth_m=25,
        open_area_m2=120,
    )
    assert pit.feasible
    assert pit.pits >= 3


def test_every_recommended_pit_drains_within_two_days():
    """CGWB practice: a percolation pit must be empty before the next storm."""
    limit = param("recharge_design", "max_drain_down_hours")
    assert limit <= 48
    for infiltration in (6, 10, 15, 20, 30):
        pit = size_recharge_pit(
            catchment_area_m2=150,
            runoff_coeff=0.85,
            infiltration_mm_per_hr=infiltration,
            gw_depth_m=25,
            open_area_m2=80,
        )
        if pit.feasible:
            assert pit.drain_down_hours <= limit


def test_recharge_is_refused_where_the_water_table_is_near_the_surface():
    """
    Recharging an aquifer whose water level is within a few metres of the surface
    risks contaminating it, and there is no space in it to store the water.
    """
    assert param("suitability_thresholds", "min_gw_depth_for_recharge_m") >= 5
    pit = size_recharge_pit(
        catchment_area_m2=120,
        runoff_coeff=0.85,
        infiltration_mm_per_hr=25,
        gw_depth_m=3,
        open_area_m2=50,
    )
    assert not pit.feasible


def test_pit_dimensions_stay_within_hand_excavation_limits():
    """Domestic pits are dug by hand: up to 2 m across and 3 m deep."""
    assert max(param("recharge_design", "pit_diameters_m")) <= 2.0
    assert max(param("recharge_design", "pit_depths_m")) <= 3.0


# --------------------------------------------------------------------------- #
# The Karnataka byelaw
# --------------------------------------------------------------------------- #


def test_byelaw_thresholds_match_the_karnataka_regulation():
    """Mandatory at 2400 sq ft (223 m2), and at 1200 sq ft (111 m2) for new builds."""
    assert param("byelaw", "mandatory_plot_area_m2") == pytest.approx(223, abs=2)
    assert param("byelaw", "mandatory_new_building_plot_area_m2") == pytest.approx(111, abs=2)


def test_byelaw_minimum_provisions_match_the_regulation():
    """20 litres per m2 of roof for storage, 10 litres per m2 for recharge."""
    assert param("byelaw", "min_storage_litres_per_m2_roof") == 20
    assert param("byelaw", "min_recharge_litres_per_m2_roof") == 10


def test_a_mandated_plot_gets_a_compliant_system(jayanagar, typical_inputs):
    report = recommend(jayanagar, typical_inputs)
    check = report["byelaw_check"]
    assert check["applicable"] is True
    assert check["compliant"] is True
    assert (
        check["storage_provided_litres_per_m2_roof"] >= check["storage_minimum_litres_per_m2_roof"]
        or check["recharge_provided_litres_per_m2_roof"]
        >= check["recharge_minimum_litres_per_m2_roof"]
    )


def test_a_small_plot_is_not_forced_to_comply(jayanagar):
    report = recommend(
        jayanagar,
        HouseholdInputs(roof_area_m2=60, open_area_m2=20, plot_area_m2=90, engine="rules"),
    )
    assert report["byelaw_check"]["applicable"] is False


def test_a_new_building_faces_the_lower_threshold(jayanagar):
    inputs = HouseholdInputs(
        roof_area_m2=90, open_area_m2=30, plot_area_m2=150, new_building=True, engine="rules"
    )
    assert recommend(jayanagar, inputs)["byelaw_check"]["applicable"] is True


def test_byelaw_applicability_is_unknown_without_a_plot_area(jayanagar):
    inputs = HouseholdInputs(roof_area_m2=120, open_area_m2=40, engine="rules")
    check = recommend(jayanagar, inputs)["byelaw_check"]
    assert check["applicable"] is None
    assert check["compliant"] is None


# --------------------------------------------------------------------------- #
# End-to-end sanity against known Bengaluru numbers
# --------------------------------------------------------------------------- #


def test_structure_life_matches_normal_practice():
    assert 15 <= param("cost_model", "structure_life_years") <= 30


def test_cost_of_a_typical_domestic_system_is_in_the_market_range(jayanagar, typical_inputs):
    """
    A sump-plus-recharge system for a 1200 sq ft roof is quoted in Bengaluru at
    roughly Rs 75,000 to Rs 2,50,000 depending on sump size and site.
    """
    report = recommend(jayanagar, typical_inputs)
    total = report["cost"]["total_capital_cost_inr"]
    assert 75_000 <= total <= 250_000


def test_reported_saving_never_exceeds_the_water_actually_harvested(jayanagar, typical_inputs):
    report = recommend(jayanagar, typical_inputs)
    assert (
        report["savings"]["annual_water_saved_litres"]
        <= report["harvest_potential"]["annual_roof_harvest_litres"] + 1
    )


def test_reported_saving_never_exceeds_non_potable_demand(jayanagar, typical_inputs):
    report = recommend(jayanagar, typical_inputs)
    assert (
        report["savings"]["annual_water_saved_litres"]
        <= report["demand"]["annual_non_potable_litres"] + 1
    )
