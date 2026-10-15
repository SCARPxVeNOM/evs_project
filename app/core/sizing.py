"""Sizing of storage tanks, recharge pits and borewell recharge structures.

**Storage.** A domestic tank is sized to bridge a dry spell - twenty days of
non-potable demand by default - but capped by what the wettest month can actually
deliver, because a tank larger than the peak month's harvest can never fill. The
result is then rounded up to the next commercially available size.

**Recharge pits.** Sized for a design storm rather than for annual volume. The pit
has to take a set share of the runoff from a 50 mm/hour hour of rain - half of it
by default, the rest going to the stormwater drain through an overflow, which is
how domestic pits are actually built. Credit is given for the water that
infiltrates while the storm lasts, through the pit floor and through the wetted
part of its walls. Whatever is left has to sit in the voids of the filter media,
and must drain away within two days so the pit is ready for the next storm. Pit
count also respects the CGWB rule of thumb of roughly one pit per 100 m^2 of roof.

**Borewell recharge.** A desilting-cum-filter chamber feeding the borewell casing.
Sized to hold ten minutes of design-storm runoff so silt settles before the water
enters the bore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import param
from app.core.calculator import design_storm_runoff_litres


def cylinder_volume_m3(diameter_m: float, depth_m: float) -> float:
    return math.pi * (diameter_m / 2.0) ** 2 * depth_m


def circle_area_m2(diameter_m: float) -> float:
    return math.pi * (diameter_m / 2.0) ** 2


def infiltrating_area_m2(diameter_m: float, depth_m: float, side_credit: float) -> float:
    """Pit floor plus the credited fraction of its wetted side wall."""
    floor = circle_area_m2(diameter_m)
    side = math.pi * diameter_m * depth_m
    return floor + side_credit * side


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StorageSizing:
    recommended_litres: int
    required_litres: float
    demand_based_litres: float
    peak_month_litres: float
    byelaw_minimum_litres: float
    design_days: int
    material: str
    basis: str
    existing_litres: float = 0.0
    additional_litres: int = 0

    def as_dict(self) -> dict:
        return {
            "recommended_litres": self.recommended_litres,
            "recommended_m3": round(self.recommended_litres / 1000.0, 2),
            "existing_litres": round(self.existing_litres),
            "additional_litres": self.additional_litres,
            "required_litres": round(self.required_litres),
            "demand_based_litres": round(self.demand_based_litres),
            "peak_month_harvest_litres": round(self.peak_month_litres),
            "byelaw_minimum_litres": round(self.byelaw_minimum_litres),
            "design_days": self.design_days,
            "material": self.material,
            "basis": self.basis,
        }


def next_standard_tank(litres: float) -> int:
    sizes = [int(s) for s in param("storage_design", "standard_tank_sizes_litres")]
    for size in sizes:
        if litres <= size:
            return size
    # Beyond the largest listed size, round up to the next 25 m^3.
    return int(math.ceil(litres / 25000.0) * 25000)


def size_storage(
    daily_non_potable_litres: float,
    peak_month_harvest_litres: float,
    roof_area_m2: float,
    design_days: int | None = None,
    byelaw_applies: bool = False,
    existing_litres: float = 0.0,
) -> StorageSizing:
    days = int(design_days if design_days is not None else param("storage_design", "design_days"))
    days = min(
        max(days, int(param("storage_design", "design_days_min"))),
        int(param("storage_design", "design_days_max")),
    )

    demand_based = daily_non_potable_litres * days
    byelaw_min = (
        roof_area_m2 * float(param("byelaw", "min_storage_litres_per_m2_roof"))
        if byelaw_applies
        else 0.0
    )

    if demand_based <= peak_month_harvest_litres:
        required = demand_based
        basis = f"{days} days of non-potable demand"
    else:
        required = peak_month_harvest_litres
        basis = "capped at the wettest month's harvest, which is less than the dry-spell demand"

    if byelaw_min > required:
        required = byelaw_min
        basis = (
            f"raised to the statutory minimum of "
            f"{param('byelaw', 'min_storage_litres_per_m2_roof')} litres per m2 of roof area"
        )

    required = max(required, float(param("storage_design", "min_useful_tank_litres")))
    existing = max(0.0, existing_litres)

    if existing >= required:
        # The sump already on site is big enough; connect the roof to it.
        recommended = int(existing)
        additional = 0
        basis = (
            f"the existing {existing:,.0f} litre sump already covers the "
            f"{required:,.0f} litres needed, so no new tank is required"
        )
    else:
        recommended = next_standard_tank(required)
        additional = int(max(0, recommended - existing))

    plastic_limit = int(param("storage_design", "plastic_tank_max_litres"))
    sized_on = additional if additional > 0 else recommended
    material = "HDPE moulded tank" if sized_on <= plastic_limit else "masonry/RCC sump"
    if additional == 0:
        material = "existing sump, reused"

    return StorageSizing(
        recommended_litres=recommended,
        required_litres=required,
        demand_based_litres=demand_based,
        peak_month_litres=peak_month_harvest_litres,
        byelaw_minimum_litres=byelaw_min,
        design_days=days,
        material=material,
        basis=basis,
        existing_litres=existing,
        additional_litres=additional,
    )


# --------------------------------------------------------------------------- #
# Recharge pits
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RechargePitSizing:
    feasible: bool
    reason: str
    pits: int = 0
    diameter_m: float = 0.0
    depth_m: float = 0.0
    gross_volume_m3_each: float = 0.0
    total_gross_volume_m3: float = 0.0
    effective_storage_litres: float = 0.0
    design_storm_runoff_litres: float = 0.0
    infiltration_during_storm_litres: float = 0.0
    storage_needed_litres: float = 0.0
    drain_down_hours: float = 0.0
    annual_recharge_litres: float = 0.0

    def as_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "pits": self.pits,
            "diameter_m": self.diameter_m,
            "depth_m": self.depth_m,
            "gross_volume_m3_each": round(self.gross_volume_m3_each, 2),
            "total_gross_volume_m3": round(self.total_gross_volume_m3, 2),
            "effective_storage_litres": round(self.effective_storage_litres),
            "design_storm_runoff_litres": round(self.design_storm_runoff_litres),
            "infiltration_during_storm_litres": round(self.infiltration_during_storm_litres),
            "storage_needed_litres": round(self.storage_needed_litres),
            "drain_down_hours": round(self.drain_down_hours, 1),
            "annual_recharge_litres": round(self.annual_recharge_litres),
        }


def minimum_pit_count(roof_area_m2: float) -> int:
    per_pit = float(param("recharge_design", "max_roof_area_per_pit_m2"))
    return max(1, int(math.ceil(roof_area_m2 / per_pit)))


def size_recharge_pit(
    catchment_area_m2: float,
    runoff_coeff: float,
    infiltration_mm_per_hr: float,
    gw_depth_m: float,
    open_area_m2: float,
    annual_volume_litres: float = 0.0,
) -> RechargePitSizing:
    """Smallest standard pit arrangement that handles the design storm."""
    cfg = param("recharge_design")
    thresholds = param("suitability_thresholds")
    capture = float(cfg["design_capture_fraction"])
    storm_runoff = design_storm_runoff_litres(catchment_area_m2, runoff_coeff) * capture

    if infiltration_mm_per_hr < float(thresholds["min_infiltration_mm_per_hr"]):
        return RechargePitSizing(
            feasible=False,
            reason=(
                f"soil infiltration of {infiltration_mm_per_hr:g} mm/hr is below the "
                f"{thresholds['min_infiltration_mm_per_hr']:g} mm/hr needed for a percolation pit "
                "to empty between storms"
            ),
            design_storm_runoff_litres=storm_runoff,
        )
    if gw_depth_m < float(thresholds["min_gw_depth_for_recharge_m"]):
        return RechargePitSizing(
            feasible=False,
            reason=(
                f"the water table is only {gw_depth_m:g} m below ground; recharging into a "
                "shallow aquifer risks contaminating it and offers little storage space"
            ),
            design_storm_runoff_litres=storm_runoff,
        )
    if gw_depth_m > float(thresholds["max_gw_depth_for_pit_m"]):
        return RechargePitSizing(
            feasible=False,
            reason=(
                f"at {gw_depth_m:g} m the water table is too deep for a shallow percolation pit "
                "to reach; recharge through the borewell instead"
            ),
            design_storm_runoff_litres=storm_runoff,
        )

    min_open_per_pit = float(cfg["min_open_area_per_pit_m2"])
    if open_area_m2 < min_open_per_pit:
        return RechargePitSizing(
            feasible=False,
            reason=(
                f"at least {min_open_per_pit:g} m2 of open ground is needed for a recharge pit; "
                f"only {open_area_m2:g} m2 is available"
            ),
            design_storm_runoff_litres=storm_runoff,
        )

    void_ratio = float(cfg["filter_media_void_ratio"])
    duration = float(cfg["design_storm_duration_hr"])
    max_drain = float(cfg["max_drain_down_hours"])
    side_credit = float(cfg["side_infiltration_credit"])
    pits_by_space = max(1, int(open_area_m2 // min_open_per_pit))
    pit_floor = minimum_pit_count(catchment_area_m2)
    max_pits = min(int(cfg["max_pits"]), max(pits_by_space, pit_floor))

    best: RechargePitSizing | None = None
    for pits in range(pit_floor, max_pits + 1):
        for depth in sorted(float(d) for d in cfg["pit_depths_m"]):
            for diameter in sorted(float(d) for d in cfg["pit_diameters_m"]):
                infiltrating_area = infiltrating_area_m2(diameter, depth, side_credit) * pits
                infiltration_per_hr = infiltration_mm_per_hr * infiltrating_area
                infiltration = infiltration_per_hr * duration
                needed = max(0.0, storm_runoff - infiltration)
                gross_each = cylinder_volume_m3(diameter, depth)
                effective = gross_each * pits * void_ratio * 1000.0
                if effective < needed:
                    continue
                drain_hours = (
                    needed / infiltration_per_hr if infiltration_per_hr > 0 else float("inf")
                )
                if drain_hours > max_drain:
                    continue
                candidate = RechargePitSizing(
                    feasible=True,
                    reason=(
                        f"sized to retain {capture:.0%} of the runoff from a "
                        f"{cfg['design_storm_mm_per_hr']:g} mm/hr design storm, with floor and "
                        "side-wall infiltration credited"
                    ),
                    pits=pits,
                    diameter_m=diameter,
                    depth_m=depth,
                    gross_volume_m3_each=gross_each,
                    total_gross_volume_m3=gross_each * pits,
                    effective_storage_litres=effective,
                    design_storm_runoff_litres=storm_runoff,
                    infiltration_during_storm_litres=infiltration,
                    storage_needed_litres=needed,
                    drain_down_hours=drain_hours,
                    annual_recharge_litres=annual_volume_litres,
                )
                if best is None or candidate.total_gross_volume_m3 < best.total_gross_volume_m3:
                    best = candidate
        if best is not None:
            break  # fewest pits that work

    if best is None:
        return RechargePitSizing(
            feasible=False,
            reason=(
                "no standard pit arrangement within the available open area can absorb the design "
                "storm; split the roof between a storage tank and borewell recharge, or have a "
                "recharge trench designed for the site"
            ),
            design_storm_runoff_litres=storm_runoff,
        )
    return best


# --------------------------------------------------------------------------- #
# Borewell recharge
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BorewellRechargeSizing:
    feasible: bool
    reason: str
    units: int = 0
    chamber_diameter_m: float = 0.0
    chamber_depth_m: float = 0.0
    chamber_volume_m3_each: float = 0.0
    required_volume_m3: float = 0.0
    provided_volume_m3: float = 0.0
    retention_minutes: float = 0.0
    annual_recharge_litres: float = 0.0

    def as_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "reason": self.reason,
            "units": self.units,
            "chamber_diameter_m": self.chamber_diameter_m,
            "chamber_depth_m": self.chamber_depth_m,
            "chamber_volume_m3_each": round(self.chamber_volume_m3_each, 2),
            "required_volume_m3": round(self.required_volume_m3, 2),
            "provided_volume_m3": round(self.provided_volume_m3, 2),
            "retention_minutes": self.retention_minutes,
            "annual_recharge_litres": round(self.annual_recharge_litres),
        }


def size_borewell_recharge(
    catchment_area_m2: float,
    runoff_coeff: float,
    gw_depth_m: float,
    has_borewell: bool,
    annual_volume_litres: float = 0.0,
) -> BorewellRechargeSizing:
    cfg = param("recharge_design", "borewell_recharge")
    storm_runoff = design_storm_runoff_litres(catchment_area_m2, runoff_coeff)

    if not has_borewell:
        return BorewellRechargeSizing(
            feasible=False, reason="the household has no borewell to recharge"
        )
    if gw_depth_m < float(cfg["min_gw_depth_m"]):
        return BorewellRechargeSizing(
            feasible=False,
            reason=(
                f"the water table at {gw_depth_m:g} m is shallower than the "
                f"{cfg['min_gw_depth_m']:g} m below which borewell recharge is worthwhile"
            ),
        )

    retention = float(cfg["retention_minutes"])
    duration_min = float(param("recharge_design", "design_storm_duration_hr")) * 60.0
    required_m3 = storm_runoff * (retention / duration_min) / 1000.0

    diameter = float(cfg["chamber_diameter_m"])
    depth = float(cfg["chamber_depth_m"])
    volume_each = cylinder_volume_m3(diameter, depth)

    by_area = int(math.ceil(catchment_area_m2 / float(cfg["max_roof_area_per_unit_m2"])))
    by_volume = int(math.ceil(required_m3 / volume_each)) if volume_each > 0 else 1
    units = max(1, by_area, by_volume)
    max_units = int(cfg["max_units"])

    if units > max_units:
        return BorewellRechargeSizing(
            feasible=False,
            reason=(
                f"this roof would need {units} borewell recharge units, more than the {max_units} "
                "that are practical; combine storage with recharge pits instead"
            ),
            required_volume_m3=required_m3,
        )

    return BorewellRechargeSizing(
        feasible=True,
        reason=(
            f"{units} desilting-cum-filter chamber(s) holding ten minutes of design-storm runoff "
            "ahead of the borewell casing"
        ),
        units=units,
        chamber_diameter_m=diameter,
        chamber_depth_m=depth,
        chamber_volume_m3_each=volume_each,
        required_volume_m3=required_m3,
        provided_volume_m3=volume_each * units,
        retention_minutes=retention,
        annual_recharge_litres=annual_volume_litres,
    )
