"""The end-to-end report assembled by the engine."""

from __future__ import annotations

import pytest

from app.core.engine import HouseholdInputs, recommend

TOP_LEVEL_KEYS = {
    "meta",
    "inputs",
    "site",
    "harvest_potential",
    "demand",
    "recommendation",
    "ml",
    "sizing",
    "water_balance",
    "dry_year_outlook",
    "cost",
    "savings",
    "byelaw_check",
    "summary",
    "maintenance",
    "caveats",
}


def test_report_has_every_section(typical_report):
    assert set(typical_report) == TOP_LEVEL_KEYS


def test_report_is_json_serialisable(typical_report):
    import json

    json.loads(json.dumps(typical_report))


def test_summary_maintenance_and_caveats_are_populated(typical_report):
    assert typical_report["summary"]
    assert typical_report["maintenance"]
    assert typical_report["caveats"]
    assert all(isinstance(line, str) and line for line in typical_report["summary"])


def test_sizing_matches_the_chosen_components(typical_report):
    components = typical_report["recommendation"]["components"]
    sizing = typical_report["sizing"]
    assert (sizing["storage"] is not None) == ("storage_tank" in components)
    assert (sizing["recharge_pit"] is not None) == ("recharge_pit" in components)
    assert (sizing["borewell_recharge"] is not None) == ("borewell_recharge" in components)


def test_tank_capacity_feeds_the_balance(typical_report):
    assert typical_report["water_balance"]["tank_capacity_litres"] == (
        typical_report["sizing"]["storage"]["recommended_litres"]
    )


def test_a_report_is_produced_for_every_hobli():
    from app.datasets import list_hoblis, site_context_for_hobli

    for entry in list_hoblis():
        site = site_context_for_hobli(entry["hobli"])
        report = recommend(
            site, HouseholdInputs(roof_area_m2=120, open_area_m2=40, engine="rules")
        )
        assert report["recommendation"]["label"]
        assert report["harvest_potential"]["annual_roof_harvest_litres"] > 0


def test_a_tiny_roof_produces_no_cost_or_saving(jayanagar):
    report = recommend(jayanagar, HouseholdInputs(roof_area_m2=5, engine="rules"))
    assert report["recommendation"]["label"] == "not_recommended"
    assert report["cost"]["total_capital_cost_inr"] == 0
    assert report["savings"] is None


def test_an_apartment_is_handled(jayanagar):
    report = recommend(
        jayanagar,
        HouseholdInputs(
            roof_area_m2=800,
            open_area_m2=60,
            household_size=80,
            house_type="apartment",
            water_source="cauvery_and_borewell",
            has_borewell=True,
            plot_area_m2=2000,
            engine="rules",
        ),
    )
    assert report["sizing"]["storage"]["recommended_litres"] > 20_000
    assert report["cost"]["total_capital_cost_inr"] > 0


def test_zero_roof_area_is_rejected(jayanagar):
    with pytest.raises(ValueError, match="roof area"):
        recommend(jayanagar, HouseholdInputs(roof_area_m2=0))


def test_an_unknown_engine_mode_is_rejected(jayanagar):
    with pytest.raises(ValueError, match="engine must be one of"):
        recommend(jayanagar, HouseholdInputs(roof_area_m2=100, engine="magic"))


def test_rules_mode_does_not_consult_the_model(jayanagar):
    report = recommend(jayanagar, HouseholdInputs(roof_area_m2=120, engine="rules"))
    assert report["ml"]["used"] == "rules"
    assert "prediction" not in report["ml"]


def test_the_report_records_which_engine_decided(typical_report):
    assert typical_report["meta"]["engine_used"] in {"rules", "ml"}
    assert typical_report["meta"]["engine_mode"] == "rules"


def test_a_stated_water_use_changes_the_outcome(jayanagar):
    frugal = recommend(
        jayanagar, HouseholdInputs(roof_area_m2=120, daily_use_lpcd=70, engine="rules")
    )
    heavy = recommend(
        jayanagar, HouseholdInputs(roof_area_m2=120, daily_use_lpcd=200, engine="rules")
    )
    assert (
        frugal["demand"]["annual_non_potable_litres"]
        < heavy["demand"]["annual_non_potable_litres"]
    )
    assert frugal["water_balance"]["reliability_pct"] > heavy["water_balance"]["reliability_pct"]


def test_results_are_reproducible(jayanagar, typical_inputs):
    first = recommend(jayanagar, typical_inputs)
    second = recommend(jayanagar, typical_inputs)
    assert first["recommendation"] == second["recommendation"]
    assert first["cost"]["total_capital_cost_inr"] == second["cost"]["total_capital_cost_inr"]
