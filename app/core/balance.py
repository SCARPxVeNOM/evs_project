"""Month-by-month water balance: what the system actually delivers.

Annual harvest on its own overstates the benefit. A tank that holds 10,000 litres
cannot deliver 90,000 litres of September rain to a household that uses 400 litres
a day - most of it overflows. So the system is simulated month by month, carrying
storage forward.

The order of operations within a month matters. This model draws the month's
demand first and spills only what is still left over:

    available = opening + harvest
    supplied  = min(demand, available)
    closing   = min(available - supplied, capacity)
    overflow  = max(0, available - supplied - capacity)

That is the yield-before-spillage convention, and it is the right one at a monthly
time step: rain that falls on the 3rd is genuinely available for use on the 20th,
so a tank should not be charged with overflowing water it would have used. Taking
the spill first - yield after spillage - would understate the yield badly at this
resolution.

Overflow is a loss to the household but a gain to the aquifer whenever a recharge
structure is part of the recommendation.

One known limitation: at a monthly step, rain falling on any day of the month is
treated as available on every other day of it, which flatters small tanks. A tank
of 1,000 litres will appear to deliver a good share of a wet month that a daily
simulation would show overflowing. Daily rainfall records would fix this, and the
loader is ready for them; the monthly figures bundled here are not.

The simulation runs for three years with the tank starting empty and reports the
final year, by which point the carry-over between December and January has
settled.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.calculator import HarvestPotential
from app.core.demand import HouseholdDemand

SETTLING_CYCLES = 3


@dataclass(frozen=True)
class MonthlyBalance:
    month: int
    month_name: str
    harvest_litres: float
    demand_litres: float
    opening_storage_litres: float
    supplied_litres: float
    deficit_litres: float
    overflow_litres: float
    closing_storage_litres: float

    @property
    def met_fully(self) -> bool:
        return self.deficit_litres <= 1.0


@dataclass(frozen=True)
class WaterBalance:
    tank_capacity_litres: float
    monthly: list[MonthlyBalance]
    open_area_runoff_litres: float
    recharge_enabled: bool

    @property
    def annual_harvest_litres(self) -> float:
        return sum(m.harvest_litres for m in self.monthly)

    @property
    def annual_demand_litres(self) -> float:
        return sum(m.demand_litres for m in self.monthly)

    @property
    def annual_supplied_litres(self) -> float:
        return sum(m.supplied_litres for m in self.monthly)

    @property
    def annual_deficit_litres(self) -> float:
        return sum(m.deficit_litres for m in self.monthly)

    @property
    def annual_overflow_litres(self) -> float:
        return sum(m.overflow_litres for m in self.monthly)

    @property
    def reliability_fraction(self) -> float:
        """Share of non-potable demand met by rainwater across the year."""
        demand = self.annual_demand_litres
        return self.annual_supplied_litres / demand if demand > 0 else 0.0

    @property
    def months_fully_met(self) -> int:
        return sum(1 for m in self.monthly if m.met_fully)

    @property
    def annual_recharge_litres(self) -> float:
        """Water put back into the ground: tank overflow plus open-area runoff."""
        if not self.recharge_enabled:
            return 0.0
        return self.annual_overflow_litres + self.open_area_runoff_litres

    def as_dict(self) -> dict:
        return {
            "tank_capacity_litres": round(self.tank_capacity_litres),
            "annual_harvest_litres": round(self.annual_harvest_litres),
            "annual_non_potable_demand_litres": round(self.annual_demand_litres),
            "annual_supplied_litres": round(self.annual_supplied_litres),
            "annual_deficit_litres": round(self.annual_deficit_litres),
            "annual_overflow_litres": round(self.annual_overflow_litres),
            "annual_recharge_litres": round(self.annual_recharge_litres),
            "reliability_pct": round(self.reliability_fraction * 100, 1),
            "months_fully_met": self.months_fully_met,
            "monthly": [
                {
                    "month": m.month,
                    "month_name": m.month_name,
                    "harvest_litres": round(m.harvest_litres),
                    "demand_litres": round(m.demand_litres),
                    "opening_storage_litres": round(m.opening_storage_litres),
                    "supplied_litres": round(m.supplied_litres),
                    "deficit_litres": round(m.deficit_litres),
                    "overflow_litres": round(m.overflow_litres),
                    "closing_storage_litres": round(m.closing_storage_litres),
                }
                for m in self.monthly
            ],
        }


def simulate_balance(
    harvest: HarvestPotential,
    demand: HouseholdDemand,
    tank_capacity_litres: float,
    recharge_enabled: bool,
) -> WaterBalance:
    """Run the monthly balance and return the settled year."""
    harvest_by_month = {m.month: m.roof_litres for m in harvest.monthly}
    demand_by_month = {m.month: m.non_potable_litres for m in demand.monthly}

    storage = 0.0
    rows: list[MonthlyBalance] = []

    for cycle in range(SETTLING_CYCLES):
        rows = []
        for month in range(1, 13):
            inflow = harvest_by_month.get(month, 0.0)
            need = demand_by_month.get(month, 0.0)
            opening = storage

            available = opening + inflow
            # With no tank there is nothing to draw from: the roof water runs
            # straight past, to a recharge structure or to the drain.
            supplied = min(need, available) if tank_capacity_litres > 0 else 0.0
            deficit = need - supplied
            remaining = available - supplied
            closing = min(remaining, tank_capacity_litres)
            overflow = remaining - closing

            rows.append(
                MonthlyBalance(
                    month=month,
                    month_name=harvest.monthly[month - 1].month_name,
                    harvest_litres=inflow,
                    demand_litres=need,
                    opening_storage_litres=opening,
                    supplied_litres=supplied,
                    deficit_litres=deficit,
                    overflow_litres=overflow,
                    closing_storage_litres=closing,
                )
            )
            storage = closing

    return WaterBalance(
        tank_capacity_litres=tank_capacity_litres,
        monthly=rows,
        open_area_runoff_litres=harvest.annual_open_litres,
        recharge_enabled=recharge_enabled,
    )
