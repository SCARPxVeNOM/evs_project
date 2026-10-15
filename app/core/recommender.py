"""The rule engine: which rainwater harvesting structure suits a site.

This is the authoritative recommender. It scores the two things a household can
do with roof runoff - store it for use, or put it back into the ground - and then
picks a structure, a combination, or neither, giving a reason for each step. The
machine-learning layer in ``app/core/ml.py`` is trained to reproduce these labels;
when the two disagree the rules win and the disagreement is reported.

Scoring rather than a decision tree of ``if`` statements keeps the reasoning
legible: every point added to a score carries a sentence explaining it, and those
sentences become the "why" section of the report.

Decision inputs are gathered in :class:`DecisionContext` - a flat bundle of
numbers and categories with no dependency on the web layer or the datasets - so
the same function can label a real request or a synthetic training sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import param
from app.core.sizing import (
    BorewellRechargeSizing,
    RechargePitSizing,
    size_borewell_recharge,
    size_recharge_pit,
)

STORAGE_TANK = "storage_tank"
RECHARGE_PIT = "recharge_pit"
BOREWELL_RECHARGE = "borewell_recharge"
STORAGE_AND_PIT = "storage_tank_and_recharge_pit"
STORAGE_AND_BOREWELL = "storage_tank_and_borewell_recharge"
NOT_RECOMMENDED = "not_recommended"

ALL_LABELS = [
    STORAGE_TANK,
    RECHARGE_PIT,
    BOREWELL_RECHARGE,
    STORAGE_AND_PIT,
    STORAGE_AND_BOREWELL,
    NOT_RECOMMENDED,
]

LABEL_TITLES = {
    STORAGE_TANK: "Storage tank (sump or HDPE tank)",
    RECHARGE_PIT: "Recharge pit (percolation pit)",
    BOREWELL_RECHARGE: "Borewell recharge structure",
    STORAGE_AND_PIT: "Storage tank with a recharge pit for the overflow",
    STORAGE_AND_BOREWELL: "Storage tank with borewell recharge for the overflow",
    NOT_RECOMMENDED: "No structure - the catchment is too small to be worth plumbing",
}

GW_CATEGORIES = ["safe", "semi_critical", "critical", "over_exploited"]
WATER_SOURCES = ["cauvery", "borewell", "tanker", "cauvery_and_borewell"]
HOUSE_TYPES = ["independent", "apartment", "villa", "row_house"]


@dataclass
class DecisionContext:
    """Flat site + household inputs the rules need."""

    roof_area_m2: float
    roof_runoff_coeff: float
    open_area_m2: float
    infiltration_mm_per_hr: float
    gw_depth_m: float
    gw_category: str
    annual_rainfall_mm: float
    annual_roof_harvest_litres: float
    annual_non_potable_demand_litres: float
    daily_non_potable_demand_litres: float
    persons: int
    house_type: str
    water_source: str
    has_borewell: bool
    slope_pct: float = 2.0
    population_density_per_km2: float = 8000.0
    plot_area_m2: float | None = None

    @property
    def demand_coverage(self) -> float:
        """Share of yearly non-potable demand the roof could cover."""
        if self.annual_non_potable_demand_litres <= 0:
            return 0.0
        return self.annual_roof_harvest_litres / self.annual_non_potable_demand_litres


@dataclass
class ScoreLine:
    points: float
    reason: str


@dataclass
class StructureDecision:
    label: str
    title: str
    components: list[str]
    storage_score: float
    recharge_score: float
    pit_score: float
    borewell_score: float
    confidence: float
    reasons: list[str] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    pit_sizing: RechargePitSizing | None = None
    borewell_sizing: BorewellRechargeSizing | None = None

    @property
    def includes_storage(self) -> bool:
        return STORAGE_TANK in self.components

    @property
    def includes_pit(self) -> bool:
        return RECHARGE_PIT in self.components

    @property
    def includes_borewell(self) -> bool:
        return BOREWELL_RECHARGE in self.components

    @property
    def includes_recharge(self) -> bool:
        return self.includes_pit or self.includes_borewell

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "title": self.title,
            "components": list(self.components),
            "confidence": round(self.confidence, 2),
            "scores": {
                "storage": round(self.storage_score, 2),
                "recharge": round(self.recharge_score, 2),
                "recharge_pit": round(self.pit_score, 2),
                "borewell_recharge": round(self.borewell_score, 2),
            },
            "reasons": list(self.reasons),
            "rejected_options": list(self.rejected),
        }


def _storage_lines(ctx: DecisionContext) -> list[ScoreLine]:
    lines: list[ScoreLine] = []

    if ctx.water_source == "tanker":
        lines.append(ScoreLine(3.0, "The household buys tanker water, the costliest source to displace."))
    elif ctx.water_source == "borewell":
        lines.append(ScoreLine(2.0, "The household depends on a borewell, which weakens through the dry months."))
    elif ctx.water_source == "cauvery_and_borewell":
        lines.append(ScoreLine(1.5, "Piped supply is backed up by a borewell, so stored rainwater can take load off both."))
    else:
        lines.append(ScoreLine(1.0, "Piped Cauvery supply is available, so stored rainwater mainly trims the bill."))

    coverage = ctx.demand_coverage
    if coverage >= 1.0:
        lines.append(ScoreLine(2.0, f"The roof can yield more than a year of non-potable demand ({coverage:.0%} of it)."))
    elif coverage >= 0.5:
        lines.append(ScoreLine(1.5, f"The roof can cover about {coverage:.0%} of yearly non-potable demand."))
    elif coverage >= 0.25:
        lines.append(ScoreLine(1.0, f"The roof can cover about {coverage:.0%} of yearly non-potable demand."))
    else:
        lines.append(ScoreLine(0.25, f"The roof covers only about {coverage:.0%} of non-potable demand, so storage has limited value."))

    if ctx.gw_category in ("over_exploited", "critical"):
        lines.append(ScoreLine(1.0, f"Groundwater here is assessed as {ctx.gw_category.replace('_', '-')}, so reducing pumping matters."))

    if ctx.population_density_per_km2 >= float(param("suitability_thresholds", "high_density_per_km2")):
        lines.append(ScoreLine(1.0, "The plot is in a densely built zone where there is little open ground for a pit."))

    if ctx.open_area_m2 < float(param("recharge_design", "min_open_area_per_pit_m2")):
        lines.append(ScoreLine(1.0, "There is almost no open ground, which leaves storage as the practical option."))

    return lines


def _pit_lines(ctx: DecisionContext, pit: RechargePitSizing) -> list[ScoreLine]:
    if not pit.feasible:
        return []

    lines = [ScoreLine(1.0, "The site has enough open ground for a percolation pit.")]

    gw_points = {"over_exploited": 3.0, "critical": 2.5, "semi_critical": 2.0, "safe": 1.0}
    lines.append(
        ScoreLine(
            gw_points.get(ctx.gw_category, 1.5),
            f"The aquifer is {ctx.gw_category.replace('_', '-')}, so recharge has clear local value.",
        )
    )

    if ctx.infiltration_mm_per_hr >= 20:
        lines.append(ScoreLine(1.5, f"The soil drains freely ({ctx.infiltration_mm_per_hr:g} mm/hr), ideal for percolation."))
    elif ctx.infiltration_mm_per_hr >= 10:
        lines.append(ScoreLine(1.0, f"The soil has moderate infiltration ({ctx.infiltration_mm_per_hr:g} mm/hr)."))
    else:
        lines.append(ScoreLine(0.5, f"The soil drains slowly ({ctx.infiltration_mm_per_hr:g} mm/hr), so the pit must be generous."))

    if ctx.demand_coverage > 0.6:
        lines.append(ScoreLine(1.5, "Harvest exceeds what a domestic tank can hold, so the surplus needs somewhere to go."))

    if ctx.gw_depth_m >= 35:
        lines.append(
            ScoreLine(
                -1.5,
                f"At {ctx.gw_depth_m:g} m the water table is far below a hand-dug pit, which will "
                "wet the upper soil rather than the aquifer being pumped.",
            )
        )

    if ctx.slope_pct > float(param("suitability_thresholds", "steep_slope_pct")):
        lines.append(ScoreLine(-1.0, f"The ground slopes at about {ctx.slope_pct:g}%, which makes runoff harder to lead into a pit."))

    return lines


def _borewell_lines(ctx: DecisionContext, bore: BorewellRechargeSizing) -> list[ScoreLine]:
    if not bore.feasible:
        return []

    lines = [ScoreLine(1.5, "There is an existing borewell that can be recharged directly.")]

    if ctx.gw_depth_m >= 40:
        lines.append(ScoreLine(3.5, f"The water table is {ctx.gw_depth_m:g} m down - only a borewell connection reaches that far."))
    elif ctx.gw_depth_m >= 30:
        lines.append(ScoreLine(3.0, f"The water table is {ctx.gw_depth_m:g} m down, so recharging the bore reaches the aquifer that is actually being drawn."))
    elif ctx.gw_depth_m >= 20:
        lines.append(ScoreLine(2.0, f"At {ctx.gw_depth_m:g} m depth, borewell recharge reaches the producing zone."))
    else:
        lines.append(ScoreLine(1.0, f"The water table at {ctx.gw_depth_m:g} m is within reach of borewell recharge."))

    if ctx.gw_category == "over_exploited":
        lines.append(ScoreLine(1.0, "The taluk is classed as over-exploited, where borewell recharge is the fastest way to slow the decline."))

    if ctx.infiltration_mm_per_hr < float(param("suitability_thresholds", "min_infiltration_mm_per_hr")):
        lines.append(ScoreLine(1.5, "Surface soil drains too slowly for a pit, but the bore bypasses the surface layer entirely."))

    if ctx.open_area_m2 < float(param("recharge_design", "min_open_area_per_pit_m2")):
        lines.append(ScoreLine(1.0, "With no room for a pit, the borewell is the only recharge route available."))

    return lines


def decide_structure(ctx: DecisionContext) -> StructureDecision:
    """Score the options and choose a structure for this site."""
    thresholds = param("suitability_thresholds")
    min_roof = float(thresholds["min_roof_area_m2"])

    pit = size_recharge_pit(
        catchment_area_m2=ctx.roof_area_m2,
        runoff_coeff=ctx.roof_runoff_coeff,
        infiltration_mm_per_hr=ctx.infiltration_mm_per_hr,
        gw_depth_m=ctx.gw_depth_m,
        open_area_m2=ctx.open_area_m2,
    )
    bore = size_borewell_recharge(
        catchment_area_m2=ctx.roof_area_m2,
        runoff_coeff=ctx.roof_runoff_coeff,
        gw_depth_m=ctx.gw_depth_m,
        has_borewell=ctx.has_borewell,
    )

    rejected: list[dict] = []
    if not pit.feasible:
        rejected.append({"option": RECHARGE_PIT, "title": LABEL_TITLES[RECHARGE_PIT], "why": pit.reason})
    if not bore.feasible:
        rejected.append({"option": BOREWELL_RECHARGE, "title": LABEL_TITLES[BOREWELL_RECHARGE], "why": bore.reason})

    if ctx.roof_area_m2 < min_roof:
        return StructureDecision(
            label=NOT_RECOMMENDED,
            title=LABEL_TITLES[NOT_RECOMMENDED],
            components=[],
            storage_score=0.0,
            recharge_score=0.0,
            pit_score=0.0,
            borewell_score=0.0,
            confidence=0.95,
            reasons=[
                f"The roof is only {ctx.roof_area_m2:g} m2. Below about {min_roof:g} m2 the yearly "
                "harvest does not justify gutters, a filter and a tank. A single covered barrel "
                "under the downpipe for garden use is the sensible step."
            ],
            rejected=rejected,
            pit_sizing=pit,
            borewell_sizing=bore,
        )

    storage_lines = _storage_lines(ctx)
    pit_lines = _pit_lines(ctx, pit)
    bore_lines = _borewell_lines(ctx, bore)

    storage_score = sum(line.points for line in storage_lines)
    pit_score = sum(line.points for line in pit_lines)
    borewell_score = sum(line.points for line in bore_lines)
    recharge_score = max(pit_score, borewell_score)

    components: list[str] = []
    reasons: list[str] = []

    # Storage is included unless the harvest is too small to be worth the plumbing
    # and recharge is clearly the better use of the same money.
    storage_worth_it = ctx.demand_coverage >= 0.15 or ctx.water_source == "tanker"
    if storage_worth_it:
        components.append(STORAGE_TANK)
        reasons.extend(line.reason for line in storage_lines if line.points > 0)
    else:
        rejected.append(
            {
                "option": STORAGE_TANK,
                "title": LABEL_TITLES[STORAGE_TANK],
                "why": (
                    f"the roof would cover only {ctx.demand_coverage:.0%} of non-potable demand, "
                    "too little to repay a tank, pump and separate plumbing line"
                ),
            }
        )

    # A recharge component is included when it is feasible and either worth points
    # on its own or needed to take the overflow the tank cannot hold.
    recharge_needed = recharge_score >= 2.0 or ctx.demand_coverage > 0.6
    if recharge_needed and (pit.feasible or bore.feasible):
        prefer_borewell = borewell_score > pit_score or not pit.feasible
        if prefer_borewell and bore.feasible:
            components.append(BOREWELL_RECHARGE)
            reasons.extend(line.reason for line in bore_lines if line.points > 0)
            if pit.feasible:
                rejected.append(
                    {
                        "option": RECHARGE_PIT,
                        "title": LABEL_TITLES[RECHARGE_PIT],
                        "why": "feasible here, but recharging the existing borewell reaches the drawn aquifer more directly",
                    }
                )
        elif pit.feasible:
            components.append(RECHARGE_PIT)
            reasons.extend(line.reason for line in pit_lines if line.points > 0)
            if bore.feasible:
                rejected.append(
                    {
                        "option": BOREWELL_RECHARGE,
                        "title": LABEL_TITLES[BOREWELL_RECHARGE],
                        "why": "feasible here, but a percolation pit is cheaper and carries less risk of silting the bore",
                    }
                )

    if not components:
        # Nothing scored: fall back to storage, which is always physically possible.
        components.append(STORAGE_TANK)
        reasons.append(
            "Neither recharge option is feasible on this site, so a storage tank is the only "
            "way to use the roof runoff."
        )
        rejected = [r for r in rejected if r["option"] != STORAGE_TANK]

    label = _label_for(components)

    scores = sorted([storage_score, pit_score, borewell_score], reverse=True)
    gap = scores[0] - scores[1]
    confidence = min(0.97, 0.6 + 0.08 * gap) if len(components) == 1 else min(0.97, 0.7 + 0.05 * gap)

    return StructureDecision(
        label=label,
        title=LABEL_TITLES[label],
        components=components,
        storage_score=storage_score,
        recharge_score=recharge_score,
        pit_score=pit_score,
        borewell_score=borewell_score,
        confidence=confidence,
        reasons=reasons,
        rejected=rejected,
        pit_sizing=pit,
        borewell_sizing=bore,
    )


def _label_for(components: list[str]) -> str:
    has_storage = STORAGE_TANK in components
    has_pit = RECHARGE_PIT in components
    has_bore = BOREWELL_RECHARGE in components

    if has_storage and has_pit:
        return STORAGE_AND_PIT
    if has_storage and has_bore:
        return STORAGE_AND_BOREWELL
    if has_storage:
        return STORAGE_TANK
    if has_pit:
        return RECHARGE_PIT
    if has_bore:
        return BOREWELL_RECHARGE
    return NOT_RECOMMENDED


def components_for_label(label: str) -> list[str]:
    """Inverse of :func:`_label_for` - used when the ML layer supplies the label."""
    return {
        STORAGE_TANK: [STORAGE_TANK],
        RECHARGE_PIT: [RECHARGE_PIT],
        BOREWELL_RECHARGE: [BOREWELL_RECHARGE],
        STORAGE_AND_PIT: [STORAGE_TANK, RECHARGE_PIT],
        STORAGE_AND_BOREWELL: [STORAGE_TANK, BOREWELL_RECHARGE],
        NOT_RECOMMENDED: [],
    }[label]
