"""Household water demand.

Demand is split into a potable part (drinking, cooking) and a non-potable part
(flushing, clothes and utensil washing, floor cleaning, gardening, vehicle
washing). Only the non-potable part is credited to rainwater, because filtered
rooftop rainwater is fit for those uses without further treatment. If a household
goes on to treat it for drinking, the saving is larger than this model reports.

Default per-capita demand follows IS 1172:1993 - 135 litres per capita per day
for a town with full flushing systems - adjusted for house type when the
household does not state its own usage.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import MONTH_NAMES, param

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


@dataclass(frozen=True)
class MonthlyDemand:
    month: int
    month_name: str
    days: int
    total_litres: float
    non_potable_litres: float


@dataclass(frozen=True)
class HouseholdDemand:
    persons: int
    lpcd: float
    lpcd_source: str
    non_potable_share: float
    monthly: list[MonthlyDemand]

    @property
    def daily_total_litres(self) -> float:
        return self.persons * self.lpcd

    @property
    def daily_non_potable_litres(self) -> float:
        return self.daily_total_litres * self.non_potable_share

    @property
    def annual_total_litres(self) -> float:
        return sum(m.total_litres for m in self.monthly)

    @property
    def annual_non_potable_litres(self) -> float:
        return sum(m.non_potable_litres for m in self.monthly)

    def as_dict(self) -> dict:
        return {
            "persons": self.persons,
            "lpcd": round(self.lpcd, 1),
            "lpcd_source": self.lpcd_source,
            "non_potable_share": self.non_potable_share,
            "daily_total_litres": round(self.daily_total_litres),
            "daily_non_potable_litres": round(self.daily_non_potable_litres),
            "annual_total_litres": round(self.annual_total_litres),
            "annual_non_potable_litres": round(self.annual_non_potable_litres),
            "monthly": [
                {
                    "month": m.month,
                    "month_name": m.month_name,
                    "days": m.days,
                    "total_litres": round(m.total_litres),
                    "non_potable_litres": round(m.non_potable_litres),
                }
                for m in self.monthly
            ],
        }


def resolve_lpcd(
    daily_use_lpcd: float | None,
    house_type: str,
    persons: int,
    monthly_bill_litres: float | None = None,
) -> tuple[float, str]:
    """Pick a per-capita demand figure and say where it came from.

    Preference order: the household's own figure, then its metered consumption,
    then the IS 1172 norm adjusted for house type.
    """
    lo = float(param("demand", "lpcd_min"))
    hi = float(param("demand", "lpcd_max"))

    if daily_use_lpcd is not None and daily_use_lpcd > 0:
        return min(max(float(daily_use_lpcd), lo), hi), "stated by household"

    if monthly_bill_litres is not None and monthly_bill_litres > 0 and persons > 0:
        derived = monthly_bill_litres / 30.0 / persons
        if lo <= derived <= hi:
            return derived, "derived from stated monthly consumption"

    base = float(param("demand", "default_lpcd"))
    adjust = param("demand", "house_type_lpcd_adjust")
    factor = float(adjust.get(house_type, 1.0))
    return min(max(base * factor, lo), hi), f"IS 1172:1993 norm adjusted for {house_type} housing"


def compute_demand(
    persons: int,
    daily_use_lpcd: float | None = None,
    house_type: str = "independent",
    non_potable_share: float | None = None,
    monthly_bill_litres: float | None = None,
) -> HouseholdDemand:
    """Monthly demand profile for a household."""
    if persons < 1:
        raise ValueError("household size must be at least 1 person")

    lpcd, source = resolve_lpcd(daily_use_lpcd, house_type, persons, monthly_bill_litres)
    share = (
        float(non_potable_share)
        if non_potable_share is not None
        else float(param("demand", "non_potable_share"))
    )
    share = min(max(share, 0.0), 1.0)

    daily = persons * lpcd
    monthly = [
        MonthlyDemand(
            month=i + 1,
            month_name=MONTH_NAMES[i],
            days=days,
            total_litres=daily * days,
            non_potable_litres=daily * days * share,
        )
        for i, days in enumerate(DAYS_IN_MONTH)
    ]
    return HouseholdDemand(
        persons=persons,
        lpcd=lpcd,
        lpcd_source=source,
        non_potable_share=share,
        monthly=monthly,
    )
