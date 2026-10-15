"""Paths and tunable parameters.

Every design constant the model uses lives in ``data/parameters.json`` so that it
can be reviewed and changed without touching code. Nothing in ``app/core`` should
hard-code a coefficient, tariff or threshold.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("RWH_DATA_DIR", PROJECT_ROOT / "data"))
WEB_DIR = PROJECT_ROOT / "web"
ML_ARTIFACT_DIR = PROJECT_ROOT / "ml" / "artifacts"

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


@lru_cache(maxsize=1)
def parameters() -> dict[str, Any]:
    """Load ``data/parameters.json`` once per process."""
    with (DATA_DIR / "parameters.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def param(*path: str, default: Any = None) -> Any:
    """Fetch a nested parameter, e.g. ``param("demand", "default_lpcd")``."""
    node: Any = parameters()
    for key in path:
        if not isinstance(node, dict) or key not in node:
            if default is not None:
                return default
            raise KeyError(f"parameter not found: {'.'.join(path)}")
        node = node[key]
    return node


def runoff_coefficient(surface: str, kind: str = "roof") -> float:
    """Runoff coefficient for a roof or open-area surface type."""
    table = param("runoff_coefficients", kind)
    if surface not in table:
        valid = ", ".join(sorted(table))
        raise ValueError(f"unknown {kind} surface {surface!r}; expected one of: {valid}")
    return float(table[surface])


def roof_types() -> list[str]:
    return sorted(param("runoff_coefficients", "roof"))


def open_area_types() -> list[str]:
    return sorted(param("runoff_coefficients", "open_area"))


def resolve_ml_path(key: str) -> Path:
    """Absolute path for an ``ml`` parameter that is stored project-relative."""
    return PROJECT_ROOT / str(param("ml", key))


def clear_caches() -> None:
    """Drop cached parameters and datasets (used by tests and the ingest scripts)."""
    from app import datasets

    parameters.cache_clear()
    datasets.clear_caches()
