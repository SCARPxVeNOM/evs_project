"""Harvestable rainwater from a roof and the open area around it.

The core relation, straight from the project brief and from IS 15797:

    harvest (litres) = rainfall (mm) x catchment area (m^2) x runoff coefficient

One millimetre of rain on one square metre is one litre, so no unit factor is
needed. Two corrections are applied on top of it:

* **First flush.** The first millimetre or so of every rain event carries roof
  dust and droppings and is diverted. With ``d`` rainy days in a month that is
  ``d x first_flush_mm`` of rainfall that never reaches the tank.
* **System efficiency.** Gutter spillage, filter rejection and tank evaporation.
  The stored stream loses more than the recharge stream, which is piped straight
  into the ground, so the two carry different factors.

The roof stream and the open-area stream are tracked separately: roof water can
be stored and used indoors, open-area runoff is fit only for recharge.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import param, runoff_coefficient
from app.datasets import SiteContext


def harvest_litres(
    rainfall_mm: float,
    area_m2: float,
    runoff_coeff: float,
    efficiency: float = 1.0,
) -> float:
    """The core estimate. Negative inputs are treated as zero."""
    if rainfall_mm <= 0 or area_m2 <= 0 or runoff_coeff <= 0:
        return 0.0
    return rainfall_mm * area_m2 * runoff_coeff * efficiency


def effective_rainfall_mm(rainfall_mm: float, rainy_days: float, first_flush_mm: float) -> float:
    """Rainfall left after the first flush of each event is diverted."""
    return max(0.0, rainfall_mm - rainy_days * first_flush_mm)


@dataclass(frozen=True)
class MonthlyHarvest:
    month: int
    month_name: str
    rainfall_mm: float
    rainy_days: float
    effective_rainfall_mm: float
    roof_litres: float
    open_area_litres: float

    @property
    def total_litres(self) -> float:
        return self.roof_litres + self.open_area_litres


@dataclass(frozen=True)
class HarvestPotential:
    roof_area_m2: float
    roof_type: str
    roof_runoff_coeff: float
    open_area_m2: float
    open_area_type: str
    open_runoff_coeff: float
    monthly: list[MonthlyHarvest]
    annual_rainfall_mm: float
    effective_annual_rainfall_mm: float

    @property
    def annual_roof_litres(self) -> float:
        return sum(m.roof_litres for m in self.monthly)

    @property
    def annual_open_litres(self) -> float:
        return sum(m.open_area_litres for m in self.monthly)

    @property
    def annual_total_litres(self) -> float:
        return self.annual_roof_litres + self.annual_open_litres

    @property
    def peak_month(self) -> MonthlyHarvest:
        return max(self.monthly, key=lambda m: m.roof_litres)

    @property
    def monsoon_litres(self) -> float:
        """Roof harvest over the four wettest months - Jun to Oct in Bengaluru."""
        ranked = sorted(self.monthly, key=lambda m: m.roof_litres, reverse=True)
        return sum(m.roof_litres for m in ranked[:4])

    def as_dict(self) -> dict:
        return {
            "roof_area_m2": self.roof_area_m2,
            "roof_type": self.roof_type,
            "roof_runoff_coefficient": self.roof_runoff_coeff,
            "open_area_m2": self.open_area_m2,
            "open_area_type": self.open_area_type,
            "open_area_runoff_coefficient": self.open_runoff_coeff,
            "annual_rainfall_mm": round(self.annual_rainfall_mm, 1),
            "effective_annual_rainfall_mm": round(self.effective_annual_rainfall_mm, 1),
            "annual_roof_harvest_litres": round(self.annual_roof_litres),
            "annual_open_area_runoff_litres": round(self.annual_open_litres),
            "annual_total_litres": round(self.annual_total_litres),
            "peak_month": self.peak_month.month_name,
            "peak_month_litres": round(self.peak_month.roof_litres),
            "monthly": [
                {
                    "month": m.month,
                    "month_name": m.month_name,
                    "rainfall_mm": round(m.rainfall_mm, 1),
                    "rainy_days": round(m.rainy_days, 1),
                    "effective_rainfall_mm": round(m.effective_rainfall_mm, 1),
                    "roof_litres": round(m.roof_litres),
                    "open_area_litres": round(m.open_area_litres),
                }
                for m in self.monthly
            ],
        }


def dry_year_monthly_ratio() -> dict[int, float]:
    """Per-month rainfall ratio of a dry year to the normal, from the IMD record.

    Derived by ``scripts/ingest_city_monthly.py`` from the eight driest years since
    1991. It is a month-by-month pattern rather than one percentage because a dry
    year in Bengaluru is not an even shortfall - the September peak is what fails.
    """
    ratios = param("dry_year", "monthly_ratio")
    return {int(k): float(v) for k, v in ratios.items() if k.isdigit()}


def compute_harvest_potential(
    site: SiteContext,
    roof_area_m2: float,
    roof_type: str,
    open_area_m2: float = 0.0,
    open_area_type: str = "paved_concrete",
    monthly_scale: dict[int, float] | None = None,
) -> HarvestPotential:
    """Month-by-month harvestable volume for a site.

    ``monthly_scale`` multiplies each month's rainfall, which is how the dry-year
    scenario is built. Rainy days are scaled by the same factor, holding the mean
    depth per rainy day constant: the record behind the dry-year ratios carries
    rainfall only, with no rainy-day counts, so there is nothing to derive a
    different relation from. Scaling days matters because the first-flush loss is
    charged per rainy day, and leaving it at the normal-year count would penalise
    a dry month twice.
    """
    roof_coeff = runoff_coefficient(roof_type, "roof")
    open_coeff = runoff_coefficient(open_area_type, "open_area")
    first_flush = float(param("system_losses", "first_flush_mm_per_event"))
    eff_storage = float(param("system_losses", "system_efficiency_storage"))
    eff_recharge = float(param("system_losses", "system_efficiency_recharge"))

    monthly: list[MonthlyHarvest] = []
    for row in site.monthly_rainfall:
        scale = 1.0 if monthly_scale is None else max(0.0, float(monthly_scale.get(row.month, 1.0)))
        rainfall = row.rainfall_mm * scale
        rainy_days = row.rainy_days * scale
        effective = effective_rainfall_mm(rainfall, rainy_days, first_flush)
        monthly.append(
            MonthlyHarvest(
                month=row.month,
                month_name=row.month_name,
                rainfall_mm=rainfall,
                rainy_days=rainy_days,
                effective_rainfall_mm=effective,
                roof_litres=harvest_litres(effective, roof_area_m2, roof_coeff, eff_storage),
                open_area_litres=harvest_litres(effective, open_area_m2, open_coeff, eff_recharge),
            )
        )

    return HarvestPotential(
        roof_area_m2=roof_area_m2,
        roof_type=roof_type,
        roof_runoff_coeff=roof_coeff,
        open_area_m2=open_area_m2,
        open_area_type=open_area_type,
        open_runoff_coeff=open_coeff,
        monthly=monthly,
        annual_rainfall_mm=sum(m.rainfall_mm for m in monthly),
        effective_annual_rainfall_mm=sum(m.effective_rainfall_mm for m in monthly),
    )


def design_storm_runoff_litres(area_m2: float, runoff_coeff: float) -> float:
    """Runoff from the design storm - what a recharge structure must swallow at once."""
    intensity = float(param("recharge_design", "design_storm_mm_per_hr"))
    duration = float(param("recharge_design", "design_storm_duration_hr"))
    return harvest_litres(intensity * duration, area_m2, runoff_coeff)
