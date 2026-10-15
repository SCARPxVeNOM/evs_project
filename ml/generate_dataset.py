"""Generate the labelled training set.

Each sample is a Bengaluru-plausible site: a hobli drawn from ``data/hoblis.csv``
with its rainfall, soil and groundwater perturbed a little, plus a randomly drawn
household. The label is whatever the rule engine recommends for it, so the
classifier learns the rules rather than inventing them.

House type drives the household draw, because a villa and a flat are not the same
site: villas have large roofs and gardens, flats have large shared roofs, many
residents and almost no open ground.

A share of samples is deliberately awkward - a tiny roof, a plot with no open
ground, impermeable soil, a shallow water table. Drawn purely at random these
cases are rare, and a training set that barely contains them teaches the model
nothing about the boundaries where the decision actually changes.

A small fraction of labels is flipped to a neighbouring structure before saving.
Real survey data would carry disagreement between surveyors, and a model trained
on perfectly clean labels reports an accuracy that no field dataset would ever
reproduce.

    python -m ml.generate_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.config import PROJECT_ROOT, param, runoff_coefficient
from app.core.calculator import effective_rainfall_mm, harvest_litres
from app.core.demand import resolve_lpcd
from app.core.ml import FEATURE_NAMES, features_from_context
from app.core.recommender import (
    BOREWELL_RECHARGE,
    NOT_RECOMMENDED,
    RECHARGE_PIT,
    STORAGE_AND_BOREWELL,
    STORAGE_AND_PIT,
    STORAGE_TANK,
    DecisionContext,
    decide_structure,
)
from app.datasets import load_hoblis

DEFAULT_OUTPUT = PROJECT_ROOT / "ml" / "data" / "training_data.csv"

ROOF_TYPES = ["rcc_flat", "rcc_sloped", "tiled", "gi_metal_sheet", "asbestos_sheet"]
ROOF_TYPE_WEIGHTS = [0.55, 0.15, 0.18, 0.07, 0.05]
OPEN_AREA_TYPES = ["paved_concrete", "paved_block", "gravel", "bare_soil", "lawn_garden"]
WATER_SOURCES = ["cauvery", "borewell", "tanker", "cauvery_and_borewell"]
WATER_SOURCE_WEIGHTS = [0.4, 0.25, 0.1, 0.25]
HOUSE_TYPES = ["independent", "apartment", "villa", "row_house"]
HOUSE_TYPE_WEIGHTS = [0.5, 0.25, 0.1, 0.15]

# Labels a mistaken surveyor might plausibly give instead of the right one.
NEIGHBOURING_LABELS = {
    STORAGE_TANK: [STORAGE_AND_PIT, STORAGE_AND_BOREWELL],
    RECHARGE_PIT: [STORAGE_AND_PIT, BOREWELL_RECHARGE],
    BOREWELL_RECHARGE: [STORAGE_AND_BOREWELL, RECHARGE_PIT],
    STORAGE_AND_PIT: [STORAGE_TANK, STORAGE_AND_BOREWELL],
    STORAGE_AND_BOREWELL: [STORAGE_TANK, STORAGE_AND_PIT],
    NOT_RECOMMENDED: [STORAGE_TANK],
}


def _draw_household(rng: np.random.Generator) -> dict:
    house_type = str(rng.choice(HOUSE_TYPES, p=HOUSE_TYPE_WEIGHTS))

    if house_type == "apartment":
        roof_area = float(rng.uniform(150, 1200))
        persons = int(rng.integers(8, 120))
        open_area = float(rng.uniform(0, 0.35 * roof_area))
    elif house_type == "villa":
        roof_area = float(rng.uniform(120, 400))
        persons = int(rng.integers(2, 8))
        open_area = float(rng.uniform(40, 300))
    elif house_type == "row_house":
        roof_area = float(rng.uniform(50, 150))
        persons = int(rng.integers(2, 8))
        open_area = float(rng.uniform(0, 40))
    else:  # independent
        roof_area = float(rng.uniform(8, 250))
        persons = int(rng.integers(1, 10))
        open_area = float(rng.uniform(0, 120))

    return {
        "house_type": house_type,
        "roof_area_m2": round(roof_area, 1),
        "persons": persons,
        "open_area_m2": round(open_area, 1),
        "roof_type": str(rng.choice(ROOF_TYPES, p=ROOF_TYPE_WEIGHTS)),
        "open_area_type": str(rng.choice(OPEN_AREA_TYPES)),
        "water_source": str(rng.choice(WATER_SOURCES, p=WATER_SOURCE_WEIGHTS)),
        "slope_pct": round(float(rng.gamma(2.0, 1.8)), 1),
    }


def _draw_site(rng: np.random.Generator, hoblis: pd.DataFrame) -> dict:
    row = hoblis.iloc[int(rng.integers(0, len(hoblis)))]
    return {
        "hobli": str(row["hobli"]),
        "annual_rainfall_mm": round(float(row["annual_rainfall_mm"]) * float(rng.uniform(0.85, 1.15)), 1),
        "annual_rainy_days": round(float(row["annual_rainy_days"]) * float(rng.uniform(0.9, 1.1)), 1),
        "infiltration_mm_per_hr": round(
            max(1.0, float(row["infiltration_mm_per_hr"]) * float(rng.uniform(0.6, 1.4))), 1
        ),
        "gw_depth_m": round(max(2.0, float(row["gw_depth_m"]) * float(rng.uniform(0.6, 1.5))), 1),
        "gw_category": str(row["gw_category"]),
        "population_density_per_km2": round(float(rng.uniform(2000, 22000)), 0),
    }


EDGE_CASE_FRACTION = 0.18


def _apply_edge_case(site: dict, household: dict, rng: np.random.Generator) -> str:
    """Push a sample onto one of the decision boundaries. Returns the case name."""
    case = str(
        rng.choice(
            ["tiny_roof", "no_open_ground", "impermeable_soil", "shallow_water_table", "deep_bore"]
        )
    )
    if case == "tiny_roof":
        household["roof_area_m2"] = round(float(rng.uniform(3, 14)), 1)
    elif case == "no_open_ground":
        household["open_area_m2"] = round(float(rng.uniform(0, 3.5)), 1)
    elif case == "impermeable_soil":
        site["infiltration_mm_per_hr"] = round(float(rng.uniform(1, 6)), 1)
    elif case == "shallow_water_table":
        site["gw_depth_m"] = round(float(rng.uniform(2, 9)), 1)
    else:  # deep_bore
        site["gw_depth_m"] = round(float(rng.uniform(30, 70)), 1)
        household["water_source"] = "borewell"
    return case


def _context_for(site: dict, household: dict, rng: np.random.Generator) -> DecisionContext:
    roof_coeff = runoff_coefficient(household["roof_type"], "roof")
    first_flush = float(param("system_losses", "first_flush_mm_per_event"))
    efficiency = float(param("system_losses", "system_efficiency_storage"))

    effective = effective_rainfall_mm(
        site["annual_rainfall_mm"], site["annual_rainy_days"], first_flush
    )
    annual_harvest = harvest_litres(effective, household["roof_area_m2"], roof_coeff, efficiency)

    stated_lpcd = float(rng.uniform(70, 220)) if rng.random() < 0.4 else None
    lpcd, _ = resolve_lpcd(stated_lpcd, household["house_type"], household["persons"])
    share = float(param("demand", "non_potable_share"))
    daily_non_potable = household["persons"] * lpcd * share

    has_borewell = bool(
        rng.random() < {"borewell": 0.98, "cauvery_and_borewell": 0.95, "tanker": 0.25, "cauvery": 0.3}[
            household["water_source"]
        ]
    )

    return DecisionContext(
        roof_area_m2=household["roof_area_m2"],
        roof_runoff_coeff=roof_coeff,
        open_area_m2=household["open_area_m2"],
        infiltration_mm_per_hr=site["infiltration_mm_per_hr"],
        gw_depth_m=site["gw_depth_m"],
        gw_category=site["gw_category"],
        annual_rainfall_mm=site["annual_rainfall_mm"],
        annual_roof_harvest_litres=annual_harvest,
        annual_non_potable_demand_litres=daily_non_potable * 365,
        daily_non_potable_demand_litres=daily_non_potable,
        persons=household["persons"],
        house_type=household["house_type"],
        water_source=household["water_source"],
        has_borewell=has_borewell,
        slope_pct=household["slope_pct"],
        population_density_per_km2=site["population_density_per_km2"],
    )


def generate(n_samples: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Build the labelled dataset as a DataFrame."""
    n_samples = int(n_samples if n_samples is not None else param("ml", "n_samples"))
    seed = int(seed if seed is not None else param("ml", "random_state"))
    noise = float(param("ml", "label_noise_fraction"))

    rng = np.random.default_rng(seed)
    hoblis = load_hoblis()

    rows = []
    for _ in range(n_samples):
        site = _draw_site(rng, hoblis)
        household = _draw_household(rng)
        edge_case = ""
        if rng.random() < EDGE_CASE_FRACTION:
            edge_case = _apply_edge_case(site, household, rng)
        ctx = _context_for(site, household, rng)
        decision = decide_structure(ctx)

        label = decision.label
        noisy = False
        if rng.random() < noise:
            alternatives = NEIGHBOURING_LABELS.get(label, [])
            if alternatives:
                label = str(rng.choice(alternatives))
                noisy = True

        row = features_from_context(ctx)
        row.update(
            {
                "hobli": site["hobli"],
                "house_type": ctx.house_type,
                "water_source": ctx.water_source,
                "roof_type": household["roof_type"],
                "gw_category": ctx.gw_category,
                "edge_case": edge_case,
                "rule_label": decision.label,
                "label": label,
                "label_is_noisy": noisy,
            }
        )
        rows.append(row)

    columns = FEATURE_NAMES + [
        "hobli",
        "house_type",
        "water_source",
        "roof_type",
        "gw_category",
        "edge_case",
        "rule_label",
        "label",
        "label_is_noisy",
    ]
    return pd.DataFrame(rows)[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the RWH structure training set")
    parser.add_argument("--samples", type=int, default=None, help="number of samples")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="output CSV path")
    args = parser.parse_args()

    df = generate(args.samples, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"wrote {args.output} ({len(df)} rows)")
    print("\nlabel distribution:")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
