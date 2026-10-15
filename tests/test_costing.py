"""Cost build-up and the savings calculation."""

from __future__ import annotations

import pytest

from app.config import param
from app.core.costing import (
    downpipe_count,
    estimate_cost,
    estimate_savings,
    gutter_length_m,
    marginal_piped_rate_per_litre,
)
from app.core.recommender import DecisionContext, decide_structure
from app.core.sizing import size_borewell_recharge, size_recharge_pit, size_storage


def _decision(**overrides):
    args = {
        "roof_area_m2": 120.0,
        "roof_runoff_coeff": 0.85,
        "open_area_m2": 40.0,
        "infiltration_mm_per_hr": 20.0,
        "gw_depth_m": 25.0,
        "gw_category": "over_exploited",
        "annual_rainfall_mm": 950.0,
        "annual_roof_harvest_litres": 82_000.0,
        "annual_non_potable_demand_litres": 108_000.0,
        "daily_non_potable_demand_litres": 297.0,
        "persons": 4,
        "house_type": "independent",
        "water_source": "cauvery_and_borewell",
        "has_borewell": True,
    }
    args.update(overrides)
    return decide_structure(DecisionContext(**args))


def _full_estimate(**overrides):
    decision = _decision(**overrides)
    storage = size_storage(297.0, 16_000.0, 120.0) if decision.includes_storage else None
    pit = (
        size_recharge_pit(120.0, 0.85, 20.0, 25.0, 40.0, 30_000)
        if decision.includes_pit
        else None
    )
    bore = (
        size_borewell_recharge(120.0, 0.85, 25.0, True, 30_000)
        if decision.includes_borewell
        else None
    )
    return decision, estimate_cost(decision, 120.0, storage, pit, bore)


def test_gutter_length_scales_with_the_square_root_of_area():
    assert gutter_length_m(100) < gutter_length_m(400)
    assert gutter_length_m(400) < 4 * gutter_length_m(100)


def test_one_downpipe_per_fifty_square_metres():
    assert downpipe_count(40) == 1
    assert downpipe_count(50) == 1
    assert downpipe_count(51) == 2
    assert downpipe_count(200) == 4


def test_cost_lines_sum_to_the_subtotal():
    _, cost = _full_estimate()
    assert sum(line.amount_inr for line in cost.lines) == pytest.approx(cost.subtotal_inr)


def test_total_adds_labour_and_contingency_on_top():
    _, cost = _full_estimate()
    assert cost.total_inr == pytest.approx(
        cost.subtotal_inr + cost.labour_inr + cost.contingency_inr
    )
    assert cost.total_inr > cost.subtotal_inr


def test_every_line_has_a_description_and_a_positive_amount():
    _, cost = _full_estimate()
    assert cost.lines
    for line in cost.lines:
        assert line.description
        assert line.amount_inr > 0
        assert line.quantity > 0


def test_the_quoted_range_brackets_the_total():
    _, cost = _full_estimate()
    assert cost.range_low_inr < cost.total_inr < cost.range_high_inr


def test_maintenance_has_a_floor():
    _, cost = _full_estimate()
    assert cost.annual_om_inr >= param("cost_model", "min_annual_om_inr")


def test_a_storage_only_system_has_no_pit_line():
    decision, cost = _full_estimate(infiltration_mm_per_hr=2.0, has_borewell=False)
    assert not decision.includes_recharge
    items = {line.item for line in cost.lines}
    assert "recharge_pit" not in items
    assert "borewell_recharge_unit" not in items


def test_nothing_to_build_costs_nothing():
    decision = _decision(roof_area_m2=5, annual_roof_harvest_litres=3_000)
    cost = estimate_cost(decision, 5.0, None, None, None)
    assert cost.total_inr == 0
    assert cost.lines == []


def test_a_bigger_tank_costs_more():
    decision = _decision()
    small = estimate_cost(decision, 120.0, size_storage(100, 16_000, 120), None, None)
    large = estimate_cost(decision, 120.0, size_storage(2_000, 160_000, 120), None, None)
    assert large.total_inr > small.total_inr


# ------------------------------------------------------------------- savings


def test_piped_rate_follows_the_telescopic_slab():
    low, _ = marginal_piped_rate_per_litre(6_000)
    mid, _ = marginal_piped_rate_per_litre(20_000)
    high, _ = marginal_piped_rate_per_litre(40_000)
    assert low < mid < high


def test_piped_rate_includes_the_sanitary_cess():
    rate, basis = marginal_piped_rate_per_litre(6_000)
    cess = param("water_rates", "sanitary_cess_fraction")
    assert rate == pytest.approx(7.0 * (1 + cess) / 1000)
    assert "cess" in basis


def _savings(source, supplied=56_000, capital=150_000, om=4_500):
    return estimate_savings(
        annual_supplied_litres=supplied,
        annual_recharge_litres=30_000,
        monthly_total_demand_litres=16_400,
        water_source=source,
        capital_cost_inr=capital,
        annual_om_inr=om,
    )


def test_tanker_water_is_valued_above_piped_water():
    assert _savings("tanker").effective_rate_per_litre_inr > _savings(
        "cauvery"
    ).effective_rate_per_litre_inr


def test_net_saving_is_gross_less_upkeep():
    savings = _savings("tanker")
    assert savings.annual_net_saving_inr == pytest.approx(
        savings.annual_gross_saving_inr - 4_500
    )


def test_a_tanker_household_at_least_saves_money():
    savings = _savings("tanker")
    assert savings.annual_net_saving_inr > 0
    assert savings.payback_years is not None


def test_reusing_an_existing_sump_brings_the_payback_inside_the_structure_life():
    """
    The common Bengaluru retrofit: the house already has a sump, so only the
    gutters, filter, plumbing and a recharge pit are new. At tanker rates that
    recovers its cost well inside the structure's life.
    """
    savings = _savings("tanker", capital=45_000, om=1_500)
    assert savings.pays_back_within_life
    assert "pays for itself" in savings.note


def test_a_subsidised_piped_household_may_never_pay_back():
    savings = _savings("cauvery", supplied=20_000, om=5_000)
    assert savings.annual_net_saving_inr <= 0
    assert savings.payback_years is None
    assert not savings.pays_back_within_life
    assert "does not pay for itself" in savings.note


def test_tanker_loads_avoided_is_reported_in_whole_loads():
    savings = _savings("tanker", supplied=60_000)
    load = param("water_rates", "tanker_load_litres")
    assert savings.tanker_loads_avoided == pytest.approx(60_000 / load)


def test_a_local_tanker_price_can_be_supplied():
    cheap = estimate_savings(56_000, 0, 16_400, "tanker", 150_000, 4_500, 400.0)
    dear = estimate_savings(56_000, 0, 16_400, "tanker", 150_000, 4_500, 1_800.0)
    assert dear.annual_gross_saving_inr > cheap.annual_gross_saving_inr
