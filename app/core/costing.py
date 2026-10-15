"""Cost of the recommended system, and what it saves.

Capital cost is built up item by item from the unit rates in ``data/costs.csv``,
with labour and contingency added as percentages. Quantities come from the sizing
step; gutter and downpipe lengths are inferred from roof area by treating the roof
as square and draining it to two sides.

Savings are valued at the rate of the water the household stops buying. That rate
depends on where their water comes from: tanker water is dear, piped Cauvery water
is charged on a telescopic slab so only the top slab the household occupies is
actually avoided, and borewell water costs only the electricity to pump it. A
share of displaced demand is assumed to have been tanker water even for piped and
borewell households, because Bengaluru households buy tankers in the dry months.

All rates are indicative Bengaluru market figures for the base year in
``data/parameters.json`` and should be refreshed before being quoted to anyone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from app.config import param
from app.core.recommender import StructureDecision
from app.core.sizing import BorewellRechargeSizing, RechargePitSizing, StorageSizing
from app.datasets import unit_cost


@dataclass(frozen=True)
class CostLine:
    item: str
    description: str
    quantity: float
    unit: str
    rate_inr: float
    amount_inr: float

    def as_dict(self) -> dict:
        return {
            "item": self.item,
            "description": self.description,
            "quantity": round(self.quantity, 2),
            "unit": self.unit,
            "rate_inr": round(self.rate_inr, 2),
            "amount_inr": round(self.amount_inr),
        }


@dataclass
class CostEstimate:
    lines: list[CostLine] = field(default_factory=list)
    subtotal_inr: float = 0.0
    labour_inr: float = 0.0
    contingency_inr: float = 0.0
    total_inr: float = 0.0
    annual_om_inr: float = 0.0
    range_low_inr: float = 0.0
    range_high_inr: float = 0.0

    def as_dict(self) -> dict:
        return {
            "lines": [line.as_dict() for line in self.lines],
            "subtotal_inr": round(self.subtotal_inr),
            "plumbing_and_labour_inr": round(self.labour_inr),
            "contingency_inr": round(self.contingency_inr),
            "total_capital_cost_inr": round(self.total_inr),
            "indicative_range_inr": [round(self.range_low_inr), round(self.range_high_inr)],
            "annual_maintenance_inr": round(self.annual_om_inr),
        }


@dataclass
class SavingsEstimate:
    annual_water_saved_litres: float
    annual_recharge_litres: float
    effective_rate_per_litre_inr: float
    rate_basis: str
    annual_gross_saving_inr: float
    annual_net_saving_inr: float
    tanker_loads_avoided: float
    payback_years: float | None
    lifetime_net_benefit_inr: float
    structure_life_years: int
    pays_back_within_life: bool
    note: str

    def as_dict(self) -> dict:
        return {
            "annual_water_saved_litres": round(self.annual_water_saved_litres),
            "annual_water_saved_kl": round(self.annual_water_saved_litres / 1000.0, 1),
            "annual_recharge_litres": round(self.annual_recharge_litres),
            "effective_rate_per_litre_inr": round(self.effective_rate_per_litre_inr, 4),
            "rate_basis": self.rate_basis,
            "annual_gross_saving_inr": round(self.annual_gross_saving_inr),
            "annual_net_saving_inr": round(self.annual_net_saving_inr),
            "tanker_loads_avoided_per_year": round(self.tanker_loads_avoided, 1),
            "payback_years": round(self.payback_years, 1) if self.payback_years else None,
            "pays_back_within_life": self.pays_back_within_life,
            "structure_life_years": self.structure_life_years,
            "lifetime_net_benefit_inr": round(self.lifetime_net_benefit_inr),
            "note": self.note,
        }


def gutter_length_m(roof_area_m2: float) -> float:
    """Gutter run plus downpipes for a roof treated as square, draining two sides."""
    side = math.sqrt(max(roof_area_m2, 1.0))
    downpipes = max(1, int(math.ceil(roof_area_m2 / 50.0)))
    return 2 * side + 4.0 * downpipes


def downpipe_count(roof_area_m2: float) -> int:
    return max(1, int(math.ceil(roof_area_m2 / 50.0)))


def estimate_cost(
    decision: StructureDecision,
    roof_area_m2: float,
    storage: StorageSizing | None,
    pit: RechargePitSizing | None,
    borewell: BorewellRechargeSizing | None,
) -> CostEstimate:
    """Itemised capital cost for the recommended components."""
    estimate = CostEstimate()

    if not decision.components:
        return estimate

    lines: list[CostLine] = []

    length = gutter_length_m(roof_area_m2)
    rate = unit_cost("gutter_downpipe")
    lines.append(
        CostLine(
            "gutter_downpipe",
            "PVC gutters and downpipes along the eaves",
            length,
            "m",
            rate,
            length * rate,
        )
    )

    pipes = downpipe_count(roof_area_m2)
    rate = unit_cost("first_flush_diverter")
    lines.append(
        CostLine(
            "first_flush_diverter",
            f"First-flush diverter on each of {pipes} downpipe(s)",
            pipes,
            "no.",
            rate,
            pipes * rate,
        )
    )

    if decision.includes_storage and storage is not None:
        rate = unit_cost("rainwater_filter")
        lines.append(
            CostLine("rainwater_filter", "Inline rainwater filter ahead of storage", 1, "no.", rate, rate)
        )

        # Only new capacity is charged for. A household that already has a sump
        # pays for the connection, not for the tank.
        litres = storage.additional_litres
        if litres > 0:
            is_plastic = storage.material.startswith("HDPE")
            item = "plastic_tank" if is_plastic else "rcc_sump"
            rate = unit_cost(item)
            description = f"{storage.material} of {litres:,} litres"
            if storage.existing_litres > 0:
                description += f", added to the existing {storage.existing_litres:,.0f} litres"
            lines.append(CostLine(item, description, litres, "litre", rate, litres * rate))

        if litres >= 3000 and storage.existing_litres <= 0:
            rate = unit_cost("pump_set")
            lines.append(CostLine("pump_set", "Pump to lift stored water for use", 1, "no.", rate, rate))

        rate = unit_cost("plumbing_distribution")
        lines.append(
            CostLine(
                "plumbing_distribution",
                "Separate line to flushing and washing points",
                1,
                "no.",
                rate,
                rate,
            )
        )

    if decision.includes_recharge:
        rate = unit_cost("desilting_chamber")
        lines.append(
            CostLine("desilting_chamber", "Silt trap ahead of the recharge structure", 1, "no.", rate, rate)
        )

    if decision.includes_pit and pit is not None and pit.feasible:
        volume = pit.total_gross_volume_m3
        rate = unit_cost("recharge_pit_per_m3")
        lines.append(
            CostLine(
                "recharge_pit",
                f"{pit.pits} recharge pit(s) of {pit.diameter_m} m dia x {pit.depth_m} m deep, "
                "lined and filled with filter media",
                volume,
                "m3",
                rate,
                volume * rate,
            )
        )

    if decision.includes_borewell and borewell is not None and borewell.feasible:
        rate = unit_cost("borewell_recharge_unit")
        lines.append(
            CostLine(
                "borewell_recharge_unit",
                f"{borewell.units} desilting-cum-filter chamber(s) feeding the borewell casing",
                borewell.units,
                "no.",
                rate,
                borewell.units * rate,
            )
        )

    subtotal = sum(line.amount_inr for line in lines)
    labour = subtotal * float(param("cost_model", "plumbing_labour_fraction"))
    contingency = (subtotal + labour) * float(param("cost_model", "contingency_fraction"))
    total = subtotal + labour + contingency
    om = max(
        float(param("cost_model", "min_annual_om_inr")),
        total * float(param("cost_model", "annual_om_fraction")),
    )

    estimate.lines = lines
    estimate.subtotal_inr = subtotal
    estimate.labour_inr = labour
    estimate.contingency_inr = contingency
    estimate.total_inr = total
    estimate.annual_om_inr = om
    estimate.range_low_inr = total * 0.85
    estimate.range_high_inr = total * 1.25
    return estimate


def marginal_piped_rate_per_litre(monthly_consumption_litres: float) -> tuple[float, str]:
    """BWSSB telescopic slab rate the household is actually paying at the margin."""
    slabs = param("water_rates", "bwssb_domestic_slabs")
    cess = float(param("water_rates", "sanitary_cess_fraction"))
    kl = monthly_consumption_litres / 1000.0

    rate_per_kl = float(slabs[-1]["rate_per_kl"])
    slab_label = f"above {slabs[-2]['up_to_kl']} kL"
    for slab in slabs:
        limit = slab["up_to_kl"]
        if limit is None or kl <= float(limit):
            rate_per_kl = float(slab["rate_per_kl"])
            slab_label = f"up to {limit} kL" if limit is not None else "top slab"
            break

    per_litre = rate_per_kl * (1 + cess) / 1000.0
    basis = (
        f"BWSSB domestic slab {slab_label} at Rs {rate_per_kl:g}/kL plus {cess:.0%} sanitary cess, "
        f"for a household drawing about {kl:.1f} kL a month"
    )
    return per_litre, basis


def estimate_savings(
    annual_supplied_litres: float,
    annual_recharge_litres: float,
    monthly_total_demand_litres: float,
    water_source: str,
    capital_cost_inr: float,
    annual_om_inr: float,
    tanker_cost_per_load_inr: float | None = None,
) -> SavingsEstimate:
    """Value the water the system displaces, and how long it takes to pay back."""
    rates = param("water_rates")
    load_litres = float(rates["tanker_load_litres"])
    load_cost = float(
        tanker_cost_per_load_inr
        if tanker_cost_per_load_inr is not None
        else rates["tanker_cost_per_load_inr"]
    )
    tanker_per_litre = load_cost / load_litres
    borewell_per_litre = float(rates["borewell_pumping_cost_per_litre_inr"])
    dependency = float(rates["assumed_tanker_dependency"].get(water_source, 0.1))

    piped_per_litre, piped_basis = marginal_piped_rate_per_litre(monthly_total_demand_litres)

    if water_source == "tanker":
        base_per_litre, base_basis = tanker_per_litre, f"tanker water at Rs {load_cost:g} per {load_litres:g} L load"
    elif water_source == "borewell":
        base_per_litre, base_basis = borewell_per_litre, f"borewell pumping at Rs {borewell_per_litre:g}/L"
    else:
        base_per_litre, base_basis = piped_per_litre, piped_basis

    effective = dependency * tanker_per_litre + (1 - dependency) * base_per_litre
    basis = base_basis
    if 0 < dependency < 1:
        basis += (
            f"; {dependency:.0%} of displaced demand assumed to have been tanker water "
            f"at Rs {load_cost:g} per load"
        )

    gross = annual_supplied_litres * effective
    net = gross - annual_om_inr
    life = int(param("cost_model", "structure_life_years"))
    payback = capital_cost_inr / net if net > 0 and capital_cost_inr > 0 else None
    pays_back = payback is not None and payback <= life

    if net <= 0:
        note = (
            "The water this system saves is worth less than its yearly upkeep at the rate the "
            "household currently pays, so it does not pay for itself in money. That is the normal "
            "result for a piped Cauvery connection, whose tariff is heavily subsidised. The case "
            "for building it rests on dry-season security, on the recharge it returns to an "
            "over-drawn aquifer, and on the byelaw where it applies - not on the bill."
        )
    elif not pays_back:
        note = (
            f"At current rates the system recovers its cost in about {payback:.0f} years, longer "
            f"than the {life}-year life assumed here. The water-security and recharge benefits "
            "matter more than the monetary return."
        )
    else:
        note = (
            f"The system pays for itself in about {payback:.1f} years and keeps saving for the "
            f"rest of its {life}-year life."
        )

    return SavingsEstimate(
        annual_water_saved_litres=annual_supplied_litres,
        annual_recharge_litres=annual_recharge_litres,
        effective_rate_per_litre_inr=effective,
        rate_basis=basis,
        annual_gross_saving_inr=gross,
        annual_net_saving_inr=net,
        tanker_loads_avoided=annual_supplied_litres * dependency / load_litres,
        payback_years=payback,
        lifetime_net_benefit_inr=net * life - capital_cost_inr,
        structure_life_years=life,
        pays_back_within_life=pays_back,
        note=note,
    )
