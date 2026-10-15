"""Sizing of tanks, pits and borewell recharge structures."""

from __future__ import annotations

import math

import pytest

from app.config import param
from app.core.sizing import (
    cylinder_volume_m3,
    infiltrating_area_m2,
    minimum_pit_count,
    next_standard_tank,
    size_borewell_recharge,
    size_recharge_pit,
    size_storage,
)

# -------------------------------------------------------------------- storage


def test_tank_is_rounded_up_to_a_commercial_size():
    sizes = param("storage_design", "standard_tank_sizes_litres")
    assert next_standard_tank(4_100) == 5_000
    assert next_standard_tank(5_000) == 5_000
    assert next_standard_tank(1) == sizes[0]


def test_very_large_requirements_round_up_in_25_m3_steps():
    assert next_standard_tank(120_000) == 125_000


def test_tank_covers_the_dry_spell_when_rain_allows():
    sizing = size_storage(
        daily_non_potable_litres=300,
        peak_month_harvest_litres=50_000,
        roof_area_m2=100,
        design_days=20,
    )
    assert sizing.demand_based_litres == 6_000
    assert sizing.required_litres == 6_000
    assert sizing.recommended_litres == 7_500
    assert "20 days" in sizing.basis


def test_tank_is_capped_by_the_wettest_month():
    sizing = size_storage(
        daily_non_potable_litres=1_000,
        peak_month_harvest_litres=5_000,
        roof_area_m2=100,
        design_days=20,
    )
    assert sizing.required_litres == 5_000
    assert "wettest month" in sizing.basis


def test_design_days_are_clamped_to_the_allowed_window():
    low = size_storage(300, 500_000, 100, design_days=1)
    high = size_storage(300, 500_000, 100, design_days=365)
    assert low.design_days == param("storage_design", "design_days_min")
    assert high.design_days == param("storage_design", "design_days_max")


def test_byelaw_floor_raises_a_small_tank():
    sizing = size_storage(
        daily_non_potable_litres=20,
        peak_month_harvest_litres=50_000,
        roof_area_m2=200,
        design_days=20,
        byelaw_applies=True,
    )
    assert sizing.byelaw_minimum_litres == 4_000
    assert sizing.required_litres >= 4_000
    assert "statutory minimum" in sizing.basis


def test_material_switches_from_plastic_to_masonry_with_size():
    assert "HDPE" in size_storage(300, 500_000, 100).material
    assert "RCC" in size_storage(5_000, 500_000, 400).material


def test_an_existing_sump_reduces_what_has_to_be_built():
    sizing = size_storage(300, 50_000, 100, design_days=20, existing_litres=3_000)
    assert sizing.recommended_litres == 7_500
    assert sizing.additional_litres == 4_500
    assert sizing.existing_litres == 3_000


def test_a_large_enough_existing_sump_needs_no_new_tank():
    sizing = size_storage(300, 50_000, 100, design_days=20, existing_litres=20_000)
    assert sizing.additional_litres == 0
    assert sizing.recommended_litres == 20_000
    assert "existing" in sizing.basis
    assert "reused" in sizing.material


# ---------------------------------------------------------------- pit geometry


def test_cylinder_volume_is_geometric():
    assert cylinder_volume_m3(2.0, 3.0) == pytest.approx(math.pi * 1.0 * 3.0)


def test_infiltrating_area_credits_floor_and_part_of_the_walls():
    floor_only = infiltrating_area_m2(1.5, 3.0, 0.0)
    with_sides = infiltrating_area_m2(1.5, 3.0, 0.5)
    assert floor_only == pytest.approx(math.pi * 0.75**2)
    assert with_sides == pytest.approx(floor_only + 0.5 * math.pi * 1.5 * 3.0)


def test_cgwb_rule_of_thumb_one_pit_per_hundred_square_metres():
    assert minimum_pit_count(80) == 1
    assert minimum_pit_count(100) == 1
    assert minimum_pit_count(101) == 2
    assert minimum_pit_count(250) == 3


# --------------------------------------------------------------- recharge pits


def _pit(**overrides):
    args = {
        "catchment_area_m2": 120.0,
        "runoff_coeff": 0.85,
        "infiltration_mm_per_hr": 20.0,
        "gw_depth_m": 25.0,
        "open_area_m2": 40.0,
    }
    args.update(overrides)
    return size_recharge_pit(**args)


def test_a_normal_site_gets_a_feasible_pit():
    pit = _pit()
    assert pit.feasible
    assert pit.pits >= 1
    assert pit.effective_storage_litres >= pit.storage_needed_litres


def test_pit_empties_within_the_allowed_window():
    pit = _pit()
    assert pit.drain_down_hours <= param("recharge_design", "max_drain_down_hours")


def test_impermeable_soil_rules_out_a_pit():
    pit = _pit(infiltration_mm_per_hr=3.0)
    assert not pit.feasible
    assert "infiltration" in pit.reason


def test_a_shallow_water_table_rules_out_a_pit():
    pit = _pit(gw_depth_m=4.0)
    assert not pit.feasible
    assert "shallow" in pit.reason or "water table" in pit.reason


def test_a_very_deep_water_table_rules_out_a_shallow_pit():
    pit = _pit(gw_depth_m=90.0)
    assert not pit.feasible
    assert "too deep" in pit.reason


def test_no_open_ground_rules_out_a_pit():
    pit = _pit(open_area_m2=1.0)
    assert not pit.feasible
    assert "open ground" in pit.reason


def test_pit_capacity_grows_with_roof_area():
    small = _pit(catchment_area_m2=60, open_area_m2=60)
    large = _pit(catchment_area_m2=400, open_area_m2=200)
    assert large.total_gross_volume_m3 > small.total_gross_volume_m3
    assert large.pits >= small.pits


def test_only_half_the_design_storm_has_to_be_retained():
    pit = _pit()
    capture = param("recharge_design", "design_capture_fraction")
    full_storm = param("recharge_design", "design_storm_mm_per_hr") * 120 * 0.85
    assert pit.design_storm_runoff_litres == pytest.approx(full_storm * capture)


def test_pit_dimensions_come_from_the_standard_list():
    pit = _pit()
    assert pit.diameter_m in param("recharge_design", "pit_diameters_m")
    assert pit.depth_m in param("recharge_design", "pit_depths_m")


# ---------------------------------------------------------- borewell recharge


def _bore(**overrides):
    args = {
        "catchment_area_m2": 120.0,
        "runoff_coeff": 0.85,
        "gw_depth_m": 30.0,
        "has_borewell": True,
    }
    args.update(overrides)
    return size_borewell_recharge(**args)


def test_borewell_recharge_needs_a_borewell():
    assert not _bore(has_borewell=False).feasible


def test_borewell_recharge_needs_a_deep_enough_water_table():
    minimum = param("recharge_design", "borewell_recharge", "min_gw_depth_m")
    assert not _bore(gw_depth_m=minimum - 5).feasible
    assert _bore(gw_depth_m=minimum + 5).feasible


def test_chamber_holds_the_required_settling_volume():
    bore = _bore()
    assert bore.feasible
    assert bore.provided_volume_m3 >= bore.required_volume_m3


def test_one_unit_serves_up_to_two_hundred_square_metres():
    per_unit = param("recharge_design", "borewell_recharge", "max_roof_area_per_unit_m2")
    assert _bore(catchment_area_m2=per_unit).units == 1
    assert _bore(catchment_area_m2=per_unit * 2 + 1).units >= 3


def test_an_oversized_roof_is_turned_away():
    bore = _bore(catchment_area_m2=5000)
    assert not bore.feasible
    assert "more than" in bore.reason
