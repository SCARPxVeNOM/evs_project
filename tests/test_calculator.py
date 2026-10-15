"""The harvest calculator."""

from __future__ import annotations

import pytest

from app.config import param, runoff_coefficient
from app.core.calculator import (
    compute_harvest_potential,
    design_storm_runoff_litres,
    effective_rainfall_mm,
    harvest_litres,
)


def test_core_formula_is_rainfall_times_area_times_coefficient():
    # 1 mm over 1 m2 is 1 litre, so 1000 mm on 100 m2 at 0.85 is 85,000 litres.
    assert harvest_litres(1000, 100, 0.85) == pytest.approx(85_000)


def test_harvest_scales_linearly_with_area():
    assert harvest_litres(900, 200, 0.8) == pytest.approx(2 * harvest_litres(900, 100, 0.8))


def test_negative_or_zero_inputs_yield_nothing():
    assert harvest_litres(0, 100, 0.85) == 0
    assert harvest_litres(900, 0, 0.85) == 0
    assert harvest_litres(-50, 100, 0.85) == 0


def test_first_flush_removes_one_mm_per_rainy_day():
    assert effective_rainfall_mm(100, 10, 1.0) == pytest.approx(90)


def test_first_flush_cannot_push_rainfall_below_zero():
    assert effective_rainfall_mm(3, 10, 1.0) == 0


def test_runoff_coefficients_sit_in_the_published_range():
    for roof in param("runoff_coefficients", "roof").values():
        assert 0.70 <= roof <= 0.95
    for surface in param("runoff_coefficients", "open_area").values():
        assert 0.10 <= surface <= 0.80


def test_metal_sheet_sheds_more_than_tile():
    assert runoff_coefficient("gi_metal_sheet") > runoff_coefficient("tiled")


def test_unknown_surface_is_rejected():
    with pytest.raises(ValueError, match="unknown roof surface"):
        runoff_coefficient("thatch")


def test_monthly_harvest_sums_to_the_annual_figure(jayanagar):
    harvest = compute_harvest_potential(jayanagar, 100, "rcc_flat", 0)
    assert sum(m.roof_litres for m in harvest.monthly) == pytest.approx(
        harvest.annual_roof_litres
    )
    assert len(harvest.monthly) == 12


def test_september_is_the_wettest_month_in_bengaluru(jayanagar):
    harvest = compute_harvest_potential(jayanagar, 100, "rcc_flat", 0)
    assert harvest.peak_month.month_name == "Sep"


def test_four_wettest_months_carry_most_of_the_year(jayanagar):
    harvest = compute_harvest_potential(jayanagar, 100, "rcc_flat", 0)
    assert harvest.monsoon_litres / harvest.annual_roof_litres > 0.6


def test_open_area_runoff_is_tracked_separately(jayanagar):
    with_open = compute_harvest_potential(jayanagar, 100, "rcc_flat", 50, "paved_concrete")
    without = compute_harvest_potential(jayanagar, 100, "rcc_flat", 0)
    assert with_open.annual_roof_litres == pytest.approx(without.annual_roof_litres)
    assert with_open.annual_open_litres > 0
    assert with_open.annual_total_litres > with_open.annual_roof_litres


def test_system_losses_reduce_the_raw_estimate(jayanagar):
    harvest = compute_harvest_potential(jayanagar, 100, "rcc_flat", 0)
    raw = harvest_litres(jayanagar.annual_rainfall_mm, 100, harvest.roof_runoff_coeff)
    assert harvest.annual_roof_litres < raw
    assert harvest.annual_roof_litres > 0.8 * raw


def test_design_storm_runoff_uses_the_hourly_intensity():
    intensity = param("recharge_design", "design_storm_mm_per_hr")
    duration = param("recharge_design", "design_storm_duration_hr")
    assert design_storm_runoff_litres(100, 0.85) == pytest.approx(
        intensity * duration * 100 * 0.85
    )
