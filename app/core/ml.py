"""The machine-learning layer: a classifier that reproduces the rule engine.

There is no field dataset of Bengaluru households labelled with the structure
they *should* have built, so there is nothing to learn a recommendation from
directly. What this layer does instead is honest and useful: the rule engine is
treated as the ground truth, a large sample of Bengaluru-plausible sites is
labelled with it, and a decision tree and random forest are trained to reproduce
those labels. That gives the project three things a bare rule engine does not
have - a compact model that generalises between the rule boundaries, feature
importances that show which site factors actually drive the choice, and a
measurable accuracy figure.

It does not give the project new knowledge. The rules remain authoritative: when
the model disagrees with them the API reports the disagreement and follows the
rules. Replace ``ml/data/training_data.csv`` with surveyed sites and the same
pipeline learns something genuinely new.

Categorical inputs are ordinal-encoded rather than one-hot encoded. Tree models
split on thresholds, so an ordinal code costs nothing in accuracy, and the
groundwater category is genuinely ordered (safe to over-exploited).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import resolve_ml_path
from app.core.recommender import (
    GW_CATEGORIES,
    HOUSE_TYPES,
    WATER_SOURCES,
    DecisionContext,
)

FEATURE_NAMES = [
    "annual_rainfall_mm",
    "roof_area_m2",
    "roof_runoff_coeff",
    "open_area_m2",
    "infiltration_mm_per_hr",
    "gw_depth_m",
    "gw_category_code",
    "persons",
    "daily_non_potable_demand_litres",
    "annual_roof_harvest_litres",
    "demand_coverage",
    "has_borewell",
    "population_density_per_km2",
    "slope_pct",
    "house_type_code",
    "water_source_code",
]

FEATURE_LABELS = {
    "annual_rainfall_mm": "Annual rainfall",
    "roof_area_m2": "Roof area",
    "roof_runoff_coeff": "Roof runoff coefficient",
    "open_area_m2": "Open area available",
    "infiltration_mm_per_hr": "Soil infiltration rate",
    "gw_depth_m": "Depth to groundwater",
    "gw_category_code": "Groundwater assessment category",
    "persons": "Household size",
    "daily_non_potable_demand_litres": "Daily non-potable demand",
    "annual_roof_harvest_litres": "Annual roof harvest",
    "demand_coverage": "Harvest as a share of demand",
    "has_borewell": "Existing borewell",
    "population_density_per_km2": "Ward population density",
    "slope_pct": "Ground slope",
    "house_type_code": "House type",
    "water_source_code": "Current water source",
}


def _code(value: str, options: list[str]) -> int:
    try:
        return options.index(value)
    except ValueError:
        return len(options)  # unknown category gets its own code


def features_from_context(ctx: DecisionContext) -> dict[str, float]:
    """Flatten a decision context into the model's feature vector."""
    return {
        "annual_rainfall_mm": float(ctx.annual_rainfall_mm),
        "roof_area_m2": float(ctx.roof_area_m2),
        "roof_runoff_coeff": float(ctx.roof_runoff_coeff),
        "open_area_m2": float(ctx.open_area_m2),
        "infiltration_mm_per_hr": float(ctx.infiltration_mm_per_hr),
        "gw_depth_m": float(ctx.gw_depth_m),
        "gw_category_code": float(_code(ctx.gw_category, GW_CATEGORIES)),
        "persons": float(ctx.persons),
        "daily_non_potable_demand_litres": float(ctx.daily_non_potable_demand_litres),
        "annual_roof_harvest_litres": float(ctx.annual_roof_harvest_litres),
        "demand_coverage": float(ctx.demand_coverage),
        "has_borewell": 1.0 if ctx.has_borewell else 0.0,
        "population_density_per_km2": float(ctx.population_density_per_km2),
        "slope_pct": float(ctx.slope_pct),
        "house_type_code": float(_code(ctx.house_type, HOUSE_TYPES)),
        "water_source_code": float(_code(ctx.water_source, WATER_SOURCES)),
    }


def feature_row(ctx: DecisionContext) -> list[float]:
    features = features_from_context(ctx)
    return [features[name] for name in FEATURE_NAMES]


@dataclass
class Prediction:
    label: str
    probability: float
    probabilities: dict[str, float]
    model_name: str

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "probability": round(self.probability, 3),
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
            "model": self.model_name,
        }


class StructureModel:
    """Thin wrapper over the trained estimator stored by ``ml/train.py``."""

    def __init__(self, bundle: dict[str, Any]):
        self.estimator = bundle["estimator"]
        self.feature_names: list[str] = bundle["feature_names"]
        self.labels: list[str] = list(bundle["labels"])
        self.model_name: str = bundle.get("model_name", type(self.estimator).__name__)
        self.metrics: dict[str, Any] = bundle.get("metrics", {})
        # A calibrated estimator wraps the tuned one and exposes no
        # feature_importances_, so the trainer stores them alongside it.
        self.stored_importances: list[dict[str, Any]] = bundle.get("feature_importances", [])

        if self.feature_names != FEATURE_NAMES:
            raise ValueError(
                "the saved model's features do not match the current code; retrain with "
                "python -m ml.train"
            )

    def predict(self, ctx: DecisionContext) -> Prediction:
        import pandas as pd

        # A frame rather than a bare list: the estimator was fitted with column
        # names, and passing them back keeps scikit-learn from warning about it.
        row = pd.DataFrame([feature_row(ctx)], columns=self.feature_names)
        label = str(self.estimator.predict(row)[0])
        probabilities: dict[str, float] = {}
        if hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(row)[0]
            classes = [str(c) for c in self.estimator.classes_]
            probabilities = dict(zip(classes, (float(p) for p in probs)))
        probability = probabilities.get(label, 1.0)
        return Prediction(
            label=label,
            probability=probability,
            probabilities=probabilities,
            model_name=self.model_name,
        )

    def feature_importances(self) -> list[dict[str, Any]]:
        if not hasattr(self.estimator, "feature_importances_"):
            return list(self.stored_importances)
        pairs = sorted(
            zip(self.feature_names, self.estimator.feature_importances_),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return [
            {
                "feature": name,
                "label": FEATURE_LABELS.get(name, name),
                "importance": round(float(value), 4),
            }
            for name, value in pairs
        ]

    def info(self) -> dict[str, Any]:
        return {
            "available": True,
            "model": self.model_name,
            "labels": self.labels,
            "metrics": self.metrics,
            "feature_importances": self.feature_importances(),
        }


@lru_cache(maxsize=1)
def load_model() -> StructureModel | None:
    """Load the trained model, or ``None`` if it has not been trained yet."""
    path: Path = resolve_ml_path("artifact_path")
    if not path.exists():
        return None
    import joblib  # imported lazily so the rules work without a model present

    try:
        return StructureModel(joblib.load(path))
    except Exception:  # a stale or corrupt artifact must not take the API down
        return None


def clear_model_cache() -> None:
    load_model.cache_clear()


def model_info() -> dict[str, Any]:
    model = load_model()
    if model is None:
        metrics_path = resolve_ml_path("metrics_path")
        return {
            "available": False,
            "reason": (
                "no trained model found; run 'python -m ml.train' to build "
                f"{resolve_ml_path('artifact_path').name}"
            ),
            "metrics_path": str(metrics_path),
        }
    return model.info()


def read_metrics() -> dict[str, Any]:
    path = resolve_ml_path("metrics_path")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
