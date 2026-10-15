"""The rule engine's structure choice."""

from __future__ import annotations

import pytest

from app.core.recommender import (
    ALL_LABELS,
    BOREWELL_RECHARGE,
    LABEL_TITLES,
    NOT_RECOMMENDED,
    RECHARGE_PIT,
    STORAGE_AND_BOREWELL,
    STORAGE_AND_PIT,
    STORAGE_TANK,
    DecisionContext,
    components_for_label,
    decide_structure,
)


def context(**overrides) -> DecisionContext:
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
        "slope_pct": 2.0,
        "population_density_per_km2": 8_000.0,
    }
    args.update(overrides)
    return DecisionContext(**args)


def test_every_label_has_a_title():
    for label in ALL_LABELS:
        assert LABEL_TITLES[label]


def test_decision_always_returns_a_known_label():
    decision = decide_structure(context())
    assert decision.label in ALL_LABELS
    assert decision.components == components_for_label(decision.label)


def test_a_typical_bengaluru_house_gets_storage_plus_recharge():
    decision = decide_structure(context())
    assert decision.label == STORAGE_AND_PIT
    assert decision.includes_storage and decision.includes_pit
    assert decision.reasons


def test_a_tiny_roof_is_turned_away():
    decision = decide_structure(context(roof_area_m2=6, annual_roof_harvest_litres=4_000))
    assert decision.label == NOT_RECOMMENDED
    assert decision.components == []
    assert "too small" in decision.reasons[0] or "barrel" in decision.reasons[0]


def test_impermeable_soil_leaves_storage_only_without_a_bore():
    decision = decide_structure(
        context(infiltration_mm_per_hr=2.0, has_borewell=False, water_source="cauvery")
    )
    assert decision.label == STORAGE_TANK
    assert not decision.includes_recharge
    assert any(r["option"] == RECHARGE_PIT for r in decision.rejected)


def test_impermeable_soil_with_a_deep_bore_prefers_borewell_recharge():
    decision = decide_structure(
        context(infiltration_mm_per_hr=2.0, gw_depth_m=40.0, has_borewell=True)
    )
    assert decision.label == STORAGE_AND_BOREWELL
    assert decision.includes_borewell


def test_a_shallow_water_table_blocks_all_recharge():
    decision = decide_structure(context(gw_depth_m=4.0, infiltration_mm_per_hr=20.0))
    assert not decision.includes_recharge
    assert decision.label == STORAGE_TANK


def test_no_open_ground_pushes_recharge_to_the_borewell():
    decision = decide_structure(context(open_area_m2=0.5, gw_depth_m=35.0, has_borewell=True))
    assert decision.includes_borewell
    assert not decision.includes_pit


def test_a_tanker_dependent_household_always_gets_storage():
    decision = decide_structure(
        context(water_source="tanker", annual_roof_harvest_litres=5_000, roof_area_m2=30)
    )
    assert decision.includes_storage


def test_storage_is_dropped_when_the_harvest_is_trivial_against_demand():
    decision = decide_structure(
        context(
            roof_area_m2=20,
            annual_roof_harvest_litres=14_000,
            annual_non_potable_demand_litres=400_000,
            daily_non_potable_demand_litres=1_100,
            persons=15,
            water_source="cauvery",
        )
    )
    assert decision.label in {RECHARGE_PIT, BOREWELL_RECHARGE}
    assert any(r["option"] == STORAGE_TANK for r in decision.rejected)


def test_a_deep_bore_outscores_a_pit():
    decision = decide_structure(context(gw_depth_m=45.0, has_borewell=True))
    assert decision.borewell_score > decision.pit_score
    assert decision.includes_borewell


def test_free_draining_soil_without_a_bore_chooses_a_pit():
    decision = decide_structure(context(infiltration_mm_per_hr=30.0, has_borewell=False))
    assert decision.includes_pit
    assert not decision.includes_borewell


def test_a_steep_site_is_penalised_for_a_pit():
    gentle = decide_structure(context(slope_pct=2.0))
    steep = decide_structure(context(slope_pct=20.0))
    assert steep.pit_score < gentle.pit_score


def test_an_overexploited_aquifer_scores_recharge_higher_than_a_safe_one():
    over = decide_structure(context(gw_category="over_exploited"))
    safe = decide_structure(context(gw_category="safe"))
    assert over.pit_score > safe.pit_score


def test_confidence_stays_in_range():
    for overrides in ({}, {"gw_depth_m": 4.0}, {"roof_area_m2": 5}, {"open_area_m2": 0}):
        decision = decide_structure(context(**overrides))
        assert 0.0 <= decision.confidence <= 1.0


def test_rejected_options_always_carry_a_reason():
    decision = decide_structure(context(infiltration_mm_per_hr=2.0, has_borewell=False))
    assert decision.rejected
    for entry in decision.rejected:
        assert entry["why"]
        assert entry["option"] in ALL_LABELS


def test_decisions_are_deterministic():
    first = decide_structure(context())
    second = decide_structure(context())
    assert first.label == second.label
    assert first.reasons == second.reasons


def test_components_for_label_round_trips():
    for label in ALL_LABELS:
        components = components_for_label(label)
        if label != NOT_RECOMMENDED:
            assert components
        assert all(c in {STORAGE_TANK, RECHARGE_PIT, BOREWELL_RECHARGE} for c in components)


def test_demand_coverage_handles_zero_demand():
    assert context(annual_non_potable_demand_litres=0).demand_coverage == 0.0


@pytest.mark.parametrize("source", ["cauvery", "borewell", "tanker", "cauvery_and_borewell"])
def test_all_water_sources_produce_a_decision(source):
    assert decide_structure(context(water_source=source)).label in ALL_LABELS
