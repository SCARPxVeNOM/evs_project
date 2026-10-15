"""Puts the pieces together into one recommendation report.

Flow, matching the brief:

    inputs -> harvest potential -> structure recommendation -> sizing and cost
           -> demand-supply comparison -> water saving and recharge

Everything the UI and the API return comes out of :func:`recommend`, which is a
pure function of a :class:`~app.datasets.SiteContext` and a set of household
inputs. No HTTP, no file IO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app import __version__
from app.config import MONTH_NAMES, param
from app.core import ml as ml_layer
from app.core.balance import simulate_balance
from app.core.calculator import compute_harvest_potential, dry_year_monthly_ratio
from app.core.demand import compute_demand
from app.core.costing import estimate_cost, estimate_savings
from app.core.recommender import (
    ALL_LABELS,
    LABEL_TITLES,
    NOT_RECOMMENDED,
    DecisionContext,
    StructureDecision,
    components_for_label,
    decide_structure,
)
from app.core.sizing import (
    BorewellRechargeSizing,
    RechargePitSizing,
    StorageSizing,
    size_borewell_recharge,
    size_recharge_pit,
    size_storage,
)
from app.datasets import SiteContext

ENGINE_MODES = ("auto", "rules", "ml")


@dataclass
class HouseholdInputs:
    """What the household tells us about the site."""

    roof_area_m2: float
    roof_type: str = "rcc_flat"
    open_area_m2: float = 0.0
    open_area_type: str = "paved_concrete"
    household_size: int = 4
    daily_use_lpcd: float | None = None
    house_type: str = "independent"
    water_source: str = "cauvery"
    has_borewell: bool = False
    plot_area_m2: float | None = None
    slope_pct: float = 2.0
    monthly_consumption_litres: float | None = None
    existing_storage_litres: float = 0.0
    storage_design_days: int | None = None
    tanker_cost_per_load_inr: float | None = None
    new_building: bool = False
    engine: str = "auto"

    def as_dict(self) -> dict[str, Any]:
        return {
            "roof_area_m2": self.roof_area_m2,
            "roof_type": self.roof_type,
            "open_area_m2": self.open_area_m2,
            "open_area_type": self.open_area_type,
            "household_size": self.household_size,
            "daily_use_lpcd": self.daily_use_lpcd,
            "house_type": self.house_type,
            "water_source": self.water_source,
            "has_borewell": self.has_borewell,
            "plot_area_m2": self.plot_area_m2,
            "slope_pct": self.slope_pct,
            "monthly_consumption_litres": self.monthly_consumption_litres,
            "existing_storage_litres": self.existing_storage_litres,
            "storage_design_days": self.storage_design_days,
            "new_building": self.new_building,
            "engine": self.engine,
        }


@dataclass
class ByelawCheck:
    applicable: bool | None
    reason: str
    storage_provided_litres_per_m2: float
    recharge_provided_litres_per_m2: float
    storage_minimum_litres_per_m2: float
    recharge_minimum_litres_per_m2: float
    compliant: bool | None
    reference: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "applicable": self.applicable,
            "reason": self.reason,
            "storage_provided_litres_per_m2_roof": round(self.storage_provided_litres_per_m2, 1),
            "recharge_provided_litres_per_m2_roof": round(self.recharge_provided_litres_per_m2, 1),
            "storage_minimum_litres_per_m2_roof": self.storage_minimum_litres_per_m2,
            "recharge_minimum_litres_per_m2_roof": self.recharge_minimum_litres_per_m2,
            "compliant": self.compliant,
            "reference": self.reference,
        }


def _byelaw_check(
    inputs: HouseholdInputs,
    decision: StructureDecision,
    storage: StorageSizing | None,
    pit: RechargePitSizing | None,
    borewell: BorewellRechargeSizing | None,
) -> ByelawCheck:
    cfg = param("byelaw")
    roof = max(inputs.roof_area_m2, 1.0)

    storage_provided = (storage.recommended_litres / roof) if storage else 0.0
    recharge_litres = 0.0
    if pit is not None and pit.feasible and decision.includes_pit:
        recharge_litres += pit.effective_storage_litres
    if borewell is not None and borewell.feasible and decision.includes_borewell:
        recharge_litres += borewell.provided_volume_m3 * 1000.0
    recharge_provided = recharge_litres / roof

    storage_min = float(cfg["min_storage_litres_per_m2_roof"])
    recharge_min = float(cfg["min_recharge_litres_per_m2_roof"])

    if inputs.plot_area_m2 is None:
        applicable: bool | None = None
        reason = (
            "Plot area was not given, so applicability could not be checked. The Karnataka "
            f"regulation applies to sites of {cfg['mandatory_plot_area_m2']:g} m2 "
            f"({cfg['mandatory_plot_area_m2'] * 10.764:.0f} sq ft) and above, and to new "
            f"buildings on sites of {cfg['mandatory_new_building_plot_area_m2']:g} m2 and above."
        )
        compliant: bool | None = None
    else:
        threshold = float(
            cfg["mandatory_new_building_plot_area_m2"]
            if inputs.new_building
            else cfg["mandatory_plot_area_m2"]
        )
        applicable = inputs.plot_area_m2 >= threshold
        kind = "a new building" if inputs.new_building else "an existing building"
        if applicable:
            reason = (
                f"The plot is {inputs.plot_area_m2:g} m2, at or above the {threshold:g} m2 "
                f"threshold for {kind}, so rainwater harvesting is mandatory."
            )
            compliant = (storage_provided >= storage_min) or (recharge_provided >= recharge_min)
        else:
            reason = (
                f"The plot is {inputs.plot_area_m2:g} m2, below the {threshold:g} m2 threshold "
                f"for {kind}, so rainwater harvesting is voluntary here."
            )
            compliant = None

    return ByelawCheck(
        applicable=applicable,
        reason=reason,
        storage_provided_litres_per_m2=storage_provided,
        recharge_provided_litres_per_m2=recharge_provided,
        storage_minimum_litres_per_m2=storage_min,
        recharge_minimum_litres_per_m2=recharge_min,
        compliant=compliant,
        reference=str(cfg["reference"]),
    )


def _dry_year_outlook(
    site: SiteContext,
    inputs: HouseholdInputs,
    decision: StructureDecision,
    tank_litres: float,
    normal_balance,
) -> dict[str, Any]:
    """What the recommended system delivers in a dry year rather than an average one.

    The tank is deliberately *not* resized. The question a household needs answered
    is what the structure they are about to build does when the monsoon is weak, so
    the normal-year design is held fixed and only the rainfall changes.
    """
    ratios = dry_year_monthly_ratio()
    cfg = param("dry_year")

    dry_harvest = compute_harvest_potential(
        site=site,
        roof_area_m2=inputs.roof_area_m2,
        roof_type=inputs.roof_type,
        open_area_m2=inputs.open_area_m2,
        open_area_type=inputs.open_area_type,
        monthly_scale=ratios,
    )
    dry_demand = compute_demand(
        persons=inputs.household_size,
        daily_use_lpcd=inputs.daily_use_lpcd,
        house_type=inputs.house_type,
        monthly_bill_litres=inputs.monthly_consumption_litres,
    )
    dry_balance = simulate_balance(
        harvest=dry_harvest,
        demand=dry_demand,
        tank_capacity_litres=tank_litres,
        recharge_enabled=decision.includes_recharge,
    )

    normal_reliability = normal_balance.reliability_fraction
    dry_reliability = dry_balance.reliability_fraction

    # The month that matters is the one losing the most water, not the one with
    # the smallest ratio. February drops to 10% of normal, but its normal is 6 mm;
    # September only drops to 39% and loses twenty times as much.
    shortfalls = {
        normal.month: normal.rainfall_mm - dry.rainfall_mm
        for normal, dry in zip(site.monthly_rainfall, dry_harvest.monthly)
    }
    worst_month = max(shortfalls, key=lambda m: shortfalls[m])
    worst_ratio = ratios.get(worst_month, 1.0)

    return {
        "scenario": (
            f"Mean of the {len(cfg['years_used'])} driest years since 1991 "
            f"({', '.join(str(y) for y in cfg['years_used'])}), from "
            f"{cfg['record_years']} years of IMD Bengaluru records."
        ),
        "annual_rainfall_mm": round(dry_harvest.annual_rainfall_mm, 1),
        "annual_fraction_of_normal": cfg["annual_fraction_of_normal"],
        "annual_roof_harvest_litres": round(dry_harvest.annual_roof_litres),
        "annual_supplied_litres": round(dry_balance.annual_supplied_litres),
        "annual_deficit_litres": round(dry_balance.annual_deficit_litres),
        "reliability_pct": round(dry_reliability * 100, 1),
        "reliability_pct_normal_year": round(normal_reliability * 100, 1),
        "reliability_drop_pct_points": round((normal_reliability - dry_reliability) * 100, 1),
        "months_fully_met": dry_balance.months_fully_met,
        "months_fully_met_normal_year": normal_balance.months_fully_met,
        "weakest_month": MONTH_NAMES[worst_month - 1],
        "weakest_month_ratio": worst_ratio,
        "weakest_month_shortfall_mm": round(shortfalls[worst_month], 1),
        "annual_percentiles_mm": cfg["annual_percentiles_mm"],
        "monthly": [
            {
                "month": m.month,
                "month_name": m.month_name,
                "ratio_to_normal": ratios.get(m.month, 1.0),
                "rainfall_mm": round(m.rainfall_mm, 1),
                "harvest_litres": round(m.roof_litres),
                "supplied_litres": round(b.supplied_litres),
                "deficit_litres": round(b.deficit_litres),
            }
            for m, b in zip(dry_harvest.monthly, dry_balance.monthly)
        ],
        "note": (
            "A dry year in Bengaluru is not an even shortfall. "
            f"{MONTH_NAMES[worst_month - 1]} loses the most - about "
            f"{shortfalls[worst_month]:.0f} mm, falling to {worst_ratio:.0%} of normal - "
            "and that is exactly when a tank should be filling for the dry months ahead."
        ),
    }


def _maintenance_notes(decision: StructureDecision) -> list[str]:
    notes = [
        "Fit a leaf screen on every gutter and clean it before the pre-monsoon showers in April "
        "and again in September.",
        "Check that the first-flush diverter empties itself after each spell; a blocked diverter "
        "sends dirty water into the system.",
    ]
    if decision.includes_storage:
        notes += [
            "Wash out the filter media once a year, before the monsoon.",
            "Clean and disinfect the tank annually, and keep it covered and mosquito-proofed.",
            "Label the rainwater taps clearly - the water is fit for flushing, washing and "
            "gardening, not for drinking without further treatment.",
        ]
    if decision.includes_pit:
        notes += [
            "Desilt the pit's settling chamber every year before the monsoon and replace the top "
            "sand layer when water starts standing in the pit for more than a day.",
        ]
    if decision.includes_borewell:
        notes += [
            "Never let unfiltered runoff into the borewell - the filter chamber must be desilted "
            "before every monsoon or the bore will choke.",
            "Keep the recharge connection above the pump's delivery line so the bore can still be "
            "pumped while it is recharging.",
        ]
    return notes


def _caveats(site: SiteContext, decision: StructureDecision) -> list[str]:
    caveats = [
        "These are estimates for planning, not an engineering design. A site visit, a soil "
        "percolation test and a licensed plumber's drawing are still needed before building.",
        "The headline figures use long-period average rainfall. A weak monsoon delivers less, "
        "which the dry-year outlook quantifies; a cloudburst can deliver a month's rain in a "
        "day, which nothing here models.",
        "Costs are indicative Bengaluru market rates and move with material prices and site "
        "access.",
        "The supply-demand comparison runs on monthly rainfall, which treats rain falling on any "
        "day of a month as available through the whole month. That flatters small tanks; a daily "
        "simulation would show more of a wet month overflowing.",
        "The dry-year pattern comes from the Bengaluru city record, so it is the same shape for "
        "every hobli. A weak monsoon is not uniform across the district, and this does not "
        "capture which parts of the city fare worst.",
    ]
    caveats.extend(site.notes)
    if decision.label == NOT_RECOMMENDED:
        caveats.append(
            "No structure was recommended, so no cost or saving is reported."
        )
    return caveats


def _summary_sentences(
    site: SiteContext,
    decision: StructureDecision,
    harvest,
    demand,
    balance,
    savings,
    dry_year: dict[str, Any] | None = None,
) -> list[str]:
    if decision.label == NOT_RECOMMENDED:
        return [
            f"At {site.hobli} the roof you described is too small for a worthwhile system.",
        ]

    lines = [
        f"A {harvest.roof_area_m2:g} m2 {harvest.roof_type.replace('_', ' ')} roof in "
        f"{site.hobli} can yield about {harvest.annual_roof_litres / 1000:,.0f} kL of rainwater a "
        f"year from {site.annual_rainfall_mm:g} mm of rain.",
        f"Recommended: {decision.title.lower()}.",
    ]
    if balance.annual_demand_litres > 0:
        lines.append(
            f"That covers about {balance.reliability_fraction:.0%} of the household's non-potable "
            f"demand - {balance.annual_supplied_litres / 1000:,.0f} kL a year - and "
            f"{balance.months_fully_met} of 12 months are met in full."
        )
    if savings is not None:
        if savings.pays_back_within_life:
            lines.append(
                f"Net saving is roughly Rs {savings.annual_net_saving_inr:,.0f} a year after "
                f"maintenance, recovering the cost in about {savings.payback_years:.1f} years."
            )
        elif savings.annual_net_saving_inr > 0:
            lines.append(
                f"Net saving is roughly Rs {savings.annual_net_saving_inr:,.0f} a year after "
                "maintenance, which does not recover the cost within the structure's life."
            )
        else:
            lines.append(
                "At the rate this household pays for water, the saving does not cover the yearly "
                "upkeep - the gain here is dry-season security and recharge, not money."
            )
    if dry_year is not None and balance.annual_demand_litres > 0:
        lines.append(
            f"In a dry year like the recent ones on record - about "
            f"{dry_year['annual_fraction_of_normal']:.0%} of normal rainfall - the same system "
            f"would cover {dry_year['reliability_pct']:.0f}% instead of "
            f"{dry_year['reliability_pct_normal_year']:.0f}%, with "
            f"{dry_year['months_fully_met']} of 12 months met in full."
        )
    if balance.annual_recharge_litres > 0:
        lines.append(
            f"About {balance.annual_recharge_litres / 1000:,.0f} kL a year goes back into the "
            f"ground, where the aquifer is classed as {site.gw_category.replace('_', '-')}."
        )
    return lines


def _resolve_with_ml(
    ctx: DecisionContext,
    rule_decision: StructureDecision,
    mode: str,
) -> tuple[StructureDecision, dict[str, Any]]:
    """Run the classifier alongside the rules and decide which label to use."""
    info: dict[str, Any] = {"mode": mode, "used": "rules"}

    if mode == "rules":
        info["note"] = "The classifier was not consulted; engine=rules was requested."
        return rule_decision, info

    model = ml_layer.load_model()
    if model is None:
        info["note"] = (
            "No trained classifier is available, so the rule engine was used. "
            "Run 'python -m ml.train' to build one."
        )
        return rule_decision, info

    prediction = model.predict(ctx)
    info["prediction"] = prediction.as_dict()
    info["rule_label"] = rule_decision.label
    info["agrees_with_rules"] = prediction.label == rule_decision.label

    if mode == "auto":
        info["note"] = (
            "The classifier agreed with the rule engine."
            if info["agrees_with_rules"]
            else (
                f"The classifier suggested '{LABEL_TITLES.get(prediction.label, prediction.label)}' "
                "while the rules chose differently. The rules decide, and the disagreement is "
                "reported here."
            )
        )
        return rule_decision, info

    # mode == "ml": follow the model, but only where the components are buildable.
    if prediction.label not in ALL_LABELS:
        info["note"] = "The classifier returned an unknown label, so the rules were used."
        return rule_decision, info

    wanted = components_for_label(prediction.label)
    pit_ok = rule_decision.pit_sizing is not None and rule_decision.pit_sizing.feasible
    bore_ok = rule_decision.borewell_sizing is not None and rule_decision.borewell_sizing.feasible
    blocked = [
        c
        for c in wanted
        if (c == "recharge_pit" and not pit_ok) or (c == "borewell_recharge" and not bore_ok)
    ]
    if blocked:
        info["used"] = "rules"
        info["note"] = (
            f"The classifier suggested '{prediction.label}', but that needs a structure this site "
            "cannot take, so the rule engine's choice was kept."
        )
        return rule_decision, info

    info["used"] = "ml"
    info["note"] = (
        "engine=ml was requested, so the classifier's label was used. Sizing, cost and the water "
        "balance are still computed by the deterministic engine."
    )
    ml_decision = StructureDecision(
        label=prediction.label,
        title=LABEL_TITLES[prediction.label],
        components=wanted,
        storage_score=rule_decision.storage_score,
        recharge_score=rule_decision.recharge_score,
        pit_score=rule_decision.pit_score,
        borewell_score=rule_decision.borewell_score,
        confidence=prediction.probability,
        reasons=[
            f"Chosen by the {prediction.model_name} classifier with "
            f"{prediction.probability:.0%} confidence."
        ]
        + rule_decision.reasons,
        rejected=rule_decision.rejected,
        pit_sizing=rule_decision.pit_sizing,
        borewell_sizing=rule_decision.borewell_sizing,
    )
    return ml_decision, info


def recommend(site: SiteContext, inputs: HouseholdInputs) -> dict[str, Any]:
    """Produce the full recommendation report for a site and household."""
    if inputs.engine not in ENGINE_MODES:
        raise ValueError(f"engine must be one of {ENGINE_MODES}, got {inputs.engine!r}")
    if inputs.roof_area_m2 <= 0:
        raise ValueError("roof area must be greater than zero")

    harvest = compute_harvest_potential(
        site=site,
        roof_area_m2=inputs.roof_area_m2,
        roof_type=inputs.roof_type,
        open_area_m2=inputs.open_area_m2,
        open_area_type=inputs.open_area_type,
    )
    demand = compute_demand(
        persons=inputs.household_size,
        daily_use_lpcd=inputs.daily_use_lpcd,
        house_type=inputs.house_type,
        monthly_bill_litres=inputs.monthly_consumption_litres,
    )

    ctx = DecisionContext(
        roof_area_m2=inputs.roof_area_m2,
        roof_runoff_coeff=harvest.roof_runoff_coeff,
        open_area_m2=inputs.open_area_m2,
        infiltration_mm_per_hr=site.infiltration_mm_per_hr,
        gw_depth_m=site.gw_depth_m,
        gw_category=site.gw_category,
        annual_rainfall_mm=site.annual_rainfall_mm,
        annual_roof_harvest_litres=harvest.annual_roof_litres,
        annual_non_potable_demand_litres=demand.annual_non_potable_litres,
        daily_non_potable_demand_litres=demand.daily_non_potable_litres,
        persons=inputs.household_size,
        house_type=inputs.house_type,
        water_source=inputs.water_source,
        has_borewell=inputs.has_borewell,
        slope_pct=inputs.slope_pct,
        population_density_per_km2=(
            site.population_density_per_km2
            if site.population_density_per_km2 is not None
            else 8000.0
        ),
        plot_area_m2=inputs.plot_area_m2,
    )

    rule_decision = decide_structure(ctx)
    decision, ml_info = _resolve_with_ml(ctx, rule_decision, inputs.engine)

    byelaw_applies = (
        inputs.plot_area_m2 is not None
        and inputs.plot_area_m2
        >= float(
            param(
                "byelaw",
                "mandatory_new_building_plot_area_m2"
                if inputs.new_building
                else "mandatory_plot_area_m2",
            )
        )
    )

    storage: StorageSizing | None = None
    if decision.includes_storage:
        storage = size_storage(
            daily_non_potable_litres=demand.daily_non_potable_litres,
            peak_month_harvest_litres=harvest.peak_month.roof_litres,
            roof_area_m2=inputs.roof_area_m2,
            design_days=inputs.storage_design_days,
            byelaw_applies=byelaw_applies,
            existing_litres=inputs.existing_storage_litres,
        )

    tank_litres = float(storage.recommended_litres) if storage else 0.0
    balance = simulate_balance(
        harvest=harvest,
        demand=demand,
        tank_capacity_litres=tank_litres,
        recharge_enabled=decision.includes_recharge,
    )

    pit: RechargePitSizing | None = None
    if decision.includes_pit:
        pit = size_recharge_pit(
            catchment_area_m2=inputs.roof_area_m2,
            runoff_coeff=harvest.roof_runoff_coeff,
            infiltration_mm_per_hr=site.infiltration_mm_per_hr,
            gw_depth_m=site.gw_depth_m,
            open_area_m2=inputs.open_area_m2,
            annual_volume_litres=balance.annual_recharge_litres,
        )

    borewell: BorewellRechargeSizing | None = None
    if decision.includes_borewell:
        borewell = size_borewell_recharge(
            catchment_area_m2=inputs.roof_area_m2,
            runoff_coeff=harvest.roof_runoff_coeff,
            gw_depth_m=site.gw_depth_m,
            has_borewell=inputs.has_borewell,
            annual_volume_litres=balance.annual_recharge_litres,
        )

    cost = estimate_cost(
        decision=decision,
        roof_area_m2=inputs.roof_area_m2,
        storage=storage,
        pit=pit,
        borewell=borewell,
    )

    savings = None
    if decision.components:
        savings = estimate_savings(
            annual_supplied_litres=balance.annual_supplied_litres,
            annual_recharge_litres=balance.annual_recharge_litres,
            monthly_total_demand_litres=demand.annual_total_litres / 12.0,
            water_source=inputs.water_source,
            capital_cost_inr=cost.total_inr,
            annual_om_inr=cost.annual_om_inr,
            tanker_cost_per_load_inr=inputs.tanker_cost_per_load_inr,
        )

    byelaw = _byelaw_check(inputs, decision, storage, pit, borewell)
    dry_year = _dry_year_outlook(site, inputs, decision, tank_litres, balance)

    return {
        "meta": {
            "app_version": __version__,
            "engine_mode": inputs.engine,
            "engine_used": ml_info.get("used", "rules"),
            "price_base_year": param("price_base_year"),
            "data_grade": site.data_grade,
        },
        "inputs": inputs.as_dict(),
        "site": site.as_dict(),
        "harvest_potential": harvest.as_dict(),
        "demand": demand.as_dict(),
        "recommendation": decision.as_dict(),
        "ml": ml_info,
        "sizing": {
            "storage": storage.as_dict() if storage else None,
            "recharge_pit": pit.as_dict() if pit else None,
            "borewell_recharge": borewell.as_dict() if borewell else None,
        },
        "water_balance": balance.as_dict(),
        "dry_year_outlook": dry_year,
        "cost": cost.as_dict(),
        "savings": savings.as_dict() if savings else None,
        "byelaw_check": byelaw.as_dict(),
        "summary": _summary_sentences(
            site, decision, harvest, demand, balance, savings, dry_year
        ),
        "maintenance": _maintenance_notes(decision),
        "caveats": _caveats(site, decision),
    }
