"""Household demand and the monthly water balance."""

from __future__ import annotations

import pytest

from app.config import param
from app.core.balance import simulate_balance
from app.core.calculator import compute_harvest_potential
from app.core.demand import compute_demand, resolve_lpcd

# --------------------------------------------------------------------- demand


def test_default_demand_follows_the_is_1172_norm():
    demand = compute_demand(persons=4, house_type="independent")
    assert demand.lpcd == pytest.approx(param("demand", "bis_1172_full_flushing_lpcd"))
    assert demand.daily_total_litres == pytest.approx(4 * 135)


def test_a_stated_figure_overrides_the_norm():
    demand = compute_demand(persons=4, daily_use_lpcd=90)
    assert demand.lpcd == 90
    assert "stated" in demand.lpcd_source


def test_metered_consumption_is_used_when_no_figure_is_stated():
    lpcd, source = resolve_lpcd(None, "independent", 4, monthly_bill_litres=18_000)
    assert lpcd == pytest.approx(150, rel=0.01)
    assert "monthly consumption" in source


def test_villas_are_assumed_to_use_more_than_flats():
    villa = compute_demand(persons=4, house_type="villa")
    flat = compute_demand(persons=4, house_type="apartment")
    assert villa.lpcd > flat.lpcd


def test_stated_use_is_clamped_to_a_sane_range():
    assert compute_demand(persons=2, daily_use_lpcd=5000).lpcd == param("demand", "lpcd_max")
    assert compute_demand(persons=2, daily_use_lpcd=1).lpcd == param("demand", "lpcd_min")


def test_only_part_of_demand_is_credited_to_rainwater():
    demand = compute_demand(persons=4)
    assert demand.non_potable_share == param("demand", "non_potable_share")
    assert demand.daily_non_potable_litres < demand.daily_total_litres


def test_the_year_has_365_days_of_demand():
    demand = compute_demand(persons=1, daily_use_lpcd=100)
    assert demand.annual_total_litres == pytest.approx(36_500)
    assert sum(m.days for m in demand.monthly) == 365


def test_an_empty_household_is_rejected():
    with pytest.raises(ValueError, match="at least 1 person"):
        compute_demand(persons=0)


# -------------------------------------------------------------------- balance


def _balance(site, tank_litres, roof_area=120, persons=4, recharge=True):
    harvest = compute_harvest_potential(site, roof_area, "rcc_flat", 40, "paved_concrete")
    demand = compute_demand(persons=persons)
    return harvest, simulate_balance(harvest, demand, tank_litres, recharge)


def test_water_is_conserved_over_the_year(jayanagar):
    _, balance = _balance(jayanagar, 10_000)
    opening = balance.monthly[0].opening_storage_litres
    closing = balance.monthly[-1].closing_storage_litres
    accounted = (
        balance.annual_supplied_litres + balance.annual_overflow_litres + closing - opening
    )
    assert accounted == pytest.approx(balance.annual_harvest_litres, rel=1e-6)


def test_storage_never_exceeds_the_tank(jayanagar):
    _, balance = _balance(jayanagar, 5_000)
    for month in balance.monthly:
        assert month.closing_storage_litres <= 5_000 + 1e-6
        assert month.closing_storage_litres >= 0


def test_a_bigger_tank_supplies_at_least_as_much(jayanagar):
    _, small = _balance(jayanagar, 2_000)
    _, large = _balance(jayanagar, 20_000)
    assert large.annual_supplied_litres >= small.annual_supplied_litres
    assert large.annual_overflow_litres <= small.annual_overflow_litres


def test_demand_is_drawn_before_the_tank_spills(jayanagar):
    """Yield before spillage: a month cannot overflow while demand is unmet."""
    _, balance = _balance(jayanagar, 5_000)
    for month in balance.monthly:
        if month.overflow_litres > 0:
            assert month.deficit_litres == pytest.approx(0, abs=1e-6)


def test_reliability_cannot_exceed_one(jayanagar):
    for tank in (1_000, 10_000, 500_000):
        _, balance = _balance(jayanagar, tank)
        assert 0 <= balance.reliability_fraction <= 1


def test_supply_never_exceeds_demand_in_any_month(jayanagar):
    _, balance = _balance(jayanagar, 50_000)
    for month in balance.monthly:
        assert month.supplied_litres <= month.demand_litres + 1e-6


def test_recharge_counts_overflow_and_open_area_runoff(jayanagar):
    harvest, balance = _balance(jayanagar, 5_000, recharge=True)
    assert balance.annual_recharge_litres == pytest.approx(
        balance.annual_overflow_litres + harvest.annual_open_litres
    )


def test_nothing_is_recharged_without_a_recharge_structure(jayanagar):
    _, balance = _balance(jayanagar, 5_000, recharge=False)
    assert balance.annual_recharge_litres == 0


def test_a_tiny_tank_still_supplies_something(jayanagar):
    _, balance = _balance(jayanagar, 1_000)
    assert balance.annual_supplied_litres > 0
    assert balance.annual_overflow_litres > 0


def test_no_tank_supplies_nothing(jayanagar):
    _, balance = _balance(jayanagar, 0)
    assert balance.annual_supplied_litres == pytest.approx(0, abs=1e-6)
