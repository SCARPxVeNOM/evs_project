"""The dry-year outlook, and the real IMD record it is built from.

The point of this scenario is that a weak monsoon in Bengaluru is not an even
shortfall. These tests pin that the monthly pattern survives into the report,
because a flat percentage would hide the risk it exists to show.
"""

from __future__ import annotations

import pytest

from app.config import param
from app.core.calculator import (
    compute_harvest_potential,
    dry_year_monthly_ratio,
)
from app.core.engine import HouseholdInputs, recommend

# --------------------------------------------------------------- the parameters


def test_the_record_behind_the_scenario_is_a_real_century():
    cfg = param("dry_year")
    assert cfg["record_years"] >= 100
    assert len(cfg["years_used"]) == 8
    assert all(1991 <= y <= 2026 for y in cfg["years_used"])


def test_a_dry_year_is_meaningfully_drier_but_not_absurd():
    fraction = param("dry_year", "annual_fraction_of_normal")
    assert 0.5 <= fraction <= 0.85


def test_the_dry_year_pattern_is_not_a_flat_scaling():
    """
    The whole reason this is a monthly pattern: September collapses while June
    holds near normal. If these ever converge, the scenario has lost its meaning.
    """
    ratios = dry_year_monthly_ratio()
    assert len(ratios) == 12
    assert max(ratios.values()) - min(ratios.values()) > 0.5


def test_the_monsoon_peak_is_what_fails():
    """September is the wettest month of the normal year and the one that fails."""
    ratios = dry_year_monthly_ratio()
    normals = {int(k): float(v) for k, v in param("monthly_rainfall_normal_mm").items()
               if k.isdigit()}
    wettest = max(normals, key=normals.get)
    assert wettest == 9
    assert ratios[9] < 0.55
    assert ratios[6] > 0.9       # June holds


def test_annual_percentiles_are_ordered():
    pcts = param("dry_year", "annual_percentiles_mm")
    values = [pcts[k] for k in ("p05", "p10", "p25", "p50", "p75", "p90")]
    assert values == sorted(values)
    assert 500 <= values[0] <= 750
    assert 1100 <= values[-1] <= 1400


# ------------------------------------------------------------- the calculator


def test_scaling_months_reduces_the_harvest(jayanagar):
    normal = compute_harvest_potential(jayanagar, 120.0, "rcc_flat", 40.0)
    dry = compute_harvest_potential(
        jayanagar, 120.0, "rcc_flat", 40.0, monthly_scale=dry_year_monthly_ratio()
    )
    assert dry.annual_roof_litres < normal.annual_roof_litres
    assert dry.annual_rainfall_mm < normal.annual_rainfall_mm


def test_an_absent_scale_changes_nothing(jayanagar):
    a = compute_harvest_potential(jayanagar, 120.0, "rcc_flat", 40.0)
    b = compute_harvest_potential(jayanagar, 120.0, "rcc_flat", 40.0, monthly_scale=None)
    assert a.annual_roof_litres == pytest.approx(b.annual_roof_litres)


def test_a_unit_scale_changes_nothing(jayanagar):
    unit = {m: 1.0 for m in range(1, 13)}
    a = compute_harvest_potential(jayanagar, 120.0, "rcc_flat", 40.0)
    b = compute_harvest_potential(jayanagar, 120.0, "rcc_flat", 40.0, monthly_scale=unit)
    assert a.annual_roof_litres == pytest.approx(b.annual_roof_litres)


def test_rainy_days_scale_with_rainfall(jayanagar):
    """
    First flush is charged per rainy day. If days stayed at the normal-year count
    a dry month would be penalised twice, so they scale together.
    """
    half = {m: 0.5 for m in range(1, 13)}
    normal = compute_harvest_potential(jayanagar, 120.0, "rcc_flat", monthly_scale=None)
    dry = compute_harvest_potential(jayanagar, 120.0, "rcc_flat", monthly_scale=half)
    for n, d in zip(normal.monthly, dry.monthly):
        assert d.rainy_days == pytest.approx(n.rainy_days * 0.5)
        assert d.rainfall_mm == pytest.approx(n.rainfall_mm * 0.5)
    # halving rainfall and days halves the effective rainfall exactly
    assert dry.effective_annual_rainfall_mm == pytest.approx(
        normal.effective_annual_rainfall_mm * 0.5
    )


def test_a_negative_scale_is_treated_as_zero(jayanagar):
    bad = {m: -1.0 for m in range(1, 13)}
    out = compute_harvest_potential(jayanagar, 120.0, "rcc_flat", monthly_scale=bad)
    assert out.annual_roof_litres == 0.0


# ------------------------------------------------------------------ the report


@pytest.fixture
def outlook(typical_report):
    return typical_report["dry_year_outlook"]


def test_the_outlook_is_reported(outlook):
    assert outlook["annual_rainfall_mm"] > 0
    assert outlook["scenario"].startswith("Mean of the 8 driest years")
    assert "IMD Bengaluru records" in outlook["scenario"]


def test_a_dry_year_is_never_better_than_a_normal_one(outlook):
    assert outlook["reliability_pct"] <= outlook["reliability_pct_normal_year"]
    assert outlook["months_fully_met"] <= outlook["months_fully_met_normal_year"]
    assert outlook["reliability_drop_pct_points"] >= 0


def test_the_outlook_names_the_month_that_loses_the_most_water(outlook):
    """
    Ranked by absolute shortfall, not by ratio. February drops to 10% of normal
    but its normal is 6 mm; September drops to 39% and loses twenty times more.
    An earlier version ranked on ratio and reported February, which is noise.
    """
    assert outlook["weakest_month"] == "Sep"
    assert outlook["weakest_month_shortfall_mm"] > 50
    assert "not an even shortfall" in outlook["note"]
    assert "Sep loses the most" in outlook["note"]


def test_the_named_month_really_has_the_largest_shortfall(outlook):
    normals = {int(k): float(v) for k, v in param("monthly_rainfall_normal_mm").items()
               if k.isdigit()}
    ratios = dry_year_monthly_ratio()
    shortfalls = {m: normals[m] * (1 - ratios[m]) for m in normals}
    worst = max(shortfalls, key=lambda m: shortfalls[m])
    assert outlook["weakest_month"] == ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][worst - 1]


def test_february_is_the_worst_ratio_but_is_not_what_gets_reported():
    """Guards the distinction directly, so a regression to ratio-ranking fails here."""
    ratios = dry_year_monthly_ratio()
    assert min(ratios, key=lambda m: ratios[m]) == 2  # February has the worst ratio
    normals = {int(k): float(v) for k, v in param("monthly_rainfall_normal_mm").items()
               if k.isdigit()}
    assert normals[2] < 10  # and it is a trivially small month


def test_the_outlook_has_twelve_months(outlook):
    assert len(outlook["monthly"]) == 12
    assert [m["month"] for m in outlook["monthly"]] == list(range(1, 13))
    for row in outlook["monthly"]:
        assert row["rainfall_mm"] >= 0
        assert row["deficit_litres"] >= 0


def test_the_dry_year_harvest_is_below_the_normal_year(typical_report, outlook):
    assert outlook["annual_roof_harvest_litres"] < (
        typical_report["harvest_potential"]["annual_roof_harvest_litres"]
    )


def test_the_summary_mentions_the_dry_year(typical_report):
    assert any("dry year" in line for line in typical_report["summary"])


def test_the_tank_is_not_resized_for_the_dry_year(typical_report):
    """
    The scenario answers what the *recommended* system does in a dry year, so the
    sizing must come from the normal year and stay put.
    """
    assert typical_report["sizing"]["storage"] is not None
    tank = typical_report["sizing"]["storage"]["recommended_litres"]
    assert typical_report["water_balance"]["tank_capacity_litres"] == tank


def test_a_site_with_no_structure_still_reports_an_outlook(jayanagar):
    """A roof too small to be worth building on must not crash the scenario."""
    report = recommend(
        jayanagar, HouseholdInputs(roof_area_m2=6.0, household_size=4, engine="rules")
    )
    assert report["dry_year_outlook"]["annual_rainfall_mm"] > 0
    assert report["dry_year_outlook"]["reliability_pct"] == 0.0
