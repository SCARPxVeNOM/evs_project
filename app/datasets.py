"""Loading and spatial lookup for the bundled Bengaluru datasets.

Four tables back the whole model:

``hoblis.csv``            one row per hobli: centroid, annual rainfall, rainy days,
                          soil class with its infiltration rate, depth to
                          groundwater and the CGWB assessment category
``rainfall_monthly.csv``  twelve rows per hobli: rainfall and rainy days by month
``bbmp_zones.csv``        BBMP administrative zones with population density
``costs.csv``             unit rates for the cost estimate

Provenance is tracked per field group rather than per row, because the two differ:
rainfall comes from KSNDMC gauge normals (``rainfall_data_grade``), while soil and
groundwater are still reference estimates (``data_grade``). Both travel out in the
API response so a reader can see which numbers are measured.
Location lookup is nearest-centroid, not point-in-polygon: the bundled tables
carry hobli and zone centroids rather than boundaries. For a household pin inside
Bengaluru this picks the right hobli almost always, but it can be wrong close to a
boundary, so every response reports which hobli was matched and how far away its
centroid is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd

from app.config import DATA_DIR, MONTH_NAMES, param

EARTH_RADIUS_KM = 6371.0088


class LocationOutOfRangeError(ValueError):
    """Raised when a requested point falls outside the covered area."""


class UnknownHobliError(ValueError):
    """Raised when a hobli name is not in the dataset."""


@lru_cache(maxsize=1)
def load_hoblis() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "hoblis.csv")
    df["hobli_key"] = df["hobli"].str.strip().str.casefold()
    return df


@lru_cache(maxsize=1)
def load_rainfall_monthly() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "rainfall_monthly.csv")
    df["hobli_key"] = df["hobli"].str.strip().str.casefold()
    return df.sort_values(["hobli_key", "month"]).reset_index(drop=True)


@lru_cache(maxsize=1)
def load_soil_wrb() -> dict[str, dict[str, Any]]:
    """ISRIC SoilGrids classification per hobli, keyed by casefolded hobli name.

    Optional: the app runs without it, just without the soil cross-check. See
    ``scripts/ingest_soilgrids.py`` for why this does not set the infiltration rate.
    """
    path = DATA_DIR / "soil_wrb.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {
        str(row["hobli"]).strip().casefold(): row.to_dict()
        for _, row in df.iterrows()
    }


@lru_cache(maxsize=1)
def load_zones() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "bbmp_zones.csv")


@lru_cache(maxsize=1)
def load_costs() -> dict[str, dict[str, Any]]:
    df = pd.read_csv(DATA_DIR / "costs.csv")
    return {row["item"]: row.to_dict() for _, row in df.iterrows()}


def clear_caches() -> None:
    load_hoblis.cache_clear()
    load_rainfall_monthly.cache_clear()
    load_zones.cache_clear()
    load_costs.cache_clear()
    load_soil_wrb.cache_clear()


def unit_cost(item: str) -> float:
    costs = load_costs()
    if item not in costs:
        raise KeyError(f"no unit cost for {item!r} in data/costs.csv")
    return float(costs[item]["unit_cost_inr"])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in kilometres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def check_in_bounds(lat: float, lon: float) -> None:
    box = param("bounding_box")
    if not (box["min_lat"] <= lat <= box["max_lat"] and box["min_lon"] <= lon <= box["max_lon"]):
        raise LocationOutOfRangeError(
            f"({lat:.4f}, {lon:.4f}) is outside the covered area "
            f"(lat {box['min_lat']}-{box['max_lat']}, lon {box['min_lon']}-{box['max_lon']}). "
            "This tool currently covers Bengaluru Urban and the adjoining Bengaluru Rural taluks only."
        )


@dataclass(frozen=True)
class MonthlyRainfall:
    month: int
    month_name: str
    rainfall_mm: float
    rainy_days: float


@dataclass(frozen=True)
class SiteContext:
    """Everything the model knows about a location before household inputs."""

    hobli: str
    taluk: str
    district: str
    hobli_lat: float
    hobli_lon: float
    distance_to_centroid_km: float
    annual_rainfall_mm: float
    annual_rainy_days: float
    soil_type: str
    infiltration_mm_per_hr: float
    gw_depth_m: float
    gw_category: str
    monthly_rainfall: list[MonthlyRainfall]
    bbmp_zone: str | None = None
    population_density_per_km2: float | None = None
    density_band: str | None = None
    data_grade: str = "reference_estimate"
    source_ref: str = ""
    rainfall_data_grade: str = "reference_estimate"
    rainfall_source_ref: str = ""
    soil_wrb_class: str | None = None
    soil_wrb_confidence_pct: float | None = None
    soil_vertisol_probability_pct: float | None = None
    soil_infiltration_disputed: bool = False
    soil_source_ref: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = {
            "hobli": self.hobli,
            "taluk": self.taluk,
            "district": self.district,
            "hobli_centroid": {"lat": self.hobli_lat, "lon": self.hobli_lon},
            "distance_to_centroid_km": round(self.distance_to_centroid_km, 2),
            "annual_rainfall_mm": self.annual_rainfall_mm,
            "annual_rainy_days": self.annual_rainy_days,
            "soil_type": self.soil_type,
            "infiltration_mm_per_hr": self.infiltration_mm_per_hr,
            "gw_depth_m": self.gw_depth_m,
            "gw_category": self.gw_category,
            "bbmp_zone": self.bbmp_zone,
            "population_density_per_km2": self.population_density_per_km2,
            "density_band": self.density_band,
            "monthly_rainfall": [
                {
                    "month": m.month,
                    "month_name": m.month_name,
                    "rainfall_mm": m.rainfall_mm,
                    "rainy_days": m.rainy_days,
                }
                for m in self.monthly_rainfall
            ],
            "data_grade": self.data_grade,
            "source_ref": self.source_ref,
            "rainfall_data_grade": self.rainfall_data_grade,
            "rainfall_source_ref": self.rainfall_source_ref,
            "soil_wrb_class": self.soil_wrb_class,
            "soil_wrb_confidence_pct": self.soil_wrb_confidence_pct,
            "soil_vertisol_probability_pct": self.soil_vertisol_probability_pct,
            "soil_infiltration_disputed": self.soil_infiltration_disputed,
            "soil_source_ref": self.soil_source_ref,
            "notes": list(self.notes),
        }
        return out


def _monthly_for(hobli_key: str, annual_mm: float, annual_days: float) -> list[MonthlyRainfall]:
    rain = load_rainfall_monthly()
    rows = rain[rain["hobli_key"] == hobli_key]
    if not rows.empty:
        return [
            MonthlyRainfall(
                month=int(r["month"]),
                month_name=str(r["month_name"]),
                rainfall_mm=float(r["rainfall_mm"]),
                rainy_days=float(r["rainy_days"]),
            )
            for _, r in rows.iterrows()
        ]

    # No monthly record for this hobli: fall back to the IMD Bengaluru shape.
    rain_shares = {int(k): v for k, v in param("monthly_rainfall_shares").items() if k.isdigit()}
    day_shares = {int(k): v for k, v in param("monthly_rainy_day_shares").items() if k.isdigit()}
    return [
        MonthlyRainfall(
            month=m,
            month_name=MONTH_NAMES[m - 1],
            rainfall_mm=round(annual_mm * rain_shares[m], 1),
            rainy_days=round(annual_days * day_shares[m], 1),
        )
        for m in range(1, 13)
    ]


def _zone_for(lat: float, lon: float) -> tuple[str, float, str] | None:
    """Nearest BBMP zone centroid, if the point is plausibly inside BBMP limits."""
    zones = load_zones()
    best = None
    for _, row in zones.iterrows():
        d = haversine_km(lat, lon, float(row["lat"]), float(row["lon"]))
        if best is None or d < best[0]:
            best = (d, row)
    if best is None:
        return None
    distance, row = best
    # Zone centroids are ~5-12 km apart; beyond 20 km the point is outside BBMP.
    if distance > 20.0:
        return None
    return str(row["zone"]), float(row["population_density_per_km2"]), str(row["density_band"])


def _context_from_row(row: pd.Series, distance_km: float, lat: float, lon: float) -> SiteContext:
    notes: list[str] = []
    if distance_km > 6.0:
        notes.append(
            f"The pin is {distance_km:.1f} km from the {row['hobli']} centroid, so the rainfall "
            "and soil figures used here may belong to a neighbouring hobli."
        )
    rainfall_grade = str(row.get("rainfall_data_grade", "reference_estimate"))
    if rainfall_grade == "ksndmc_hobli_normal":
        notes.append(
            "Rainfall is this hobli's KSNDMC rain-gauge normal, and its split across the "
            "pre-monsoon, south-west and north-east seasons is measured too. The shape "
            "within each season follows the IMD Bengaluru City monthly normals, so a single "
            "month can be out even where the season total is right."
        )
    elif rainfall_grade == "ksndmc_taluk_normal":
        notes.append(
            f"{row['hobli']} has no rain gauge of its own, so rainfall here is the KSNDMC "
            f"normal for the whole of {row['taluk']} taluk - measured, but averaged over an "
            "area larger than this hobli."
        )
    else:
        notes.append(
            "Rainfall for this hobli is a reference estimate, not a gauge reading. Load the "
            "KSNDMC normals with scripts/ingest_ksndmc.py before using the output for design."
        )

    if str(row["data_grade"]) != "measured":
        notes.append(
            "Soil infiltration and depth to groundwater are reference estimates for the hobli, "
            "not site measurements. A percolation test and a local water-level reading are "
            "what a real design would use."
        )

    soil = load_soil_wrb().get(str(row["hobli_key"]), {})
    disputed = str(soil.get("infiltration_disputed", "no")).strip().lower() == "yes"
    if disputed:
        notes.append(
            f"The ISRIC SoilGrids soil model puts {soil['vertisol_probability_pct']:.0f}% "
            f"probability on shrink-swell clay at this centroid, which percolates far slower "
            f"than the {soil['assumed_infiltration_mm_per_hr']:.0f} mm/hr assumed here. A "
            "percolation test matters more than usual before committing to a recharge pit."
        )

    zone_info = _zone_for(lat, lon)
    return SiteContext(
        hobli=str(row["hobli"]),
        taluk=str(row["taluk"]),
        district=str(row["district"]),
        hobli_lat=float(row["lat"]),
        hobli_lon=float(row["lon"]),
        distance_to_centroid_km=distance_km,
        annual_rainfall_mm=float(row["annual_rainfall_mm"]),
        annual_rainy_days=float(row["annual_rainy_days"]),
        soil_type=str(row["soil_type"]),
        infiltration_mm_per_hr=float(row["infiltration_mm_per_hr"]),
        gw_depth_m=float(row["gw_depth_m"]),
        gw_category=str(row["gw_category"]),
        monthly_rainfall=_monthly_for(
            str(row["hobli_key"]),
            float(row["annual_rainfall_mm"]),
            float(row["annual_rainy_days"]),
        ),
        bbmp_zone=zone_info[0] if zone_info else None,
        population_density_per_km2=zone_info[1] if zone_info else None,
        density_band=zone_info[2] if zone_info else None,
        data_grade=str(row["data_grade"]),
        source_ref=str(row["source_ref"]),
        rainfall_data_grade=rainfall_grade,
        rainfall_source_ref=str(row.get("rainfall_source_ref", "")),
        soil_wrb_class=str(soil["wrb_class"]) if soil else None,
        soil_wrb_confidence_pct=float(soil["wrb_confidence_pct"]) if soil else None,
        soil_vertisol_probability_pct=(
            float(soil["vertisol_probability_pct"]) if soil else None
        ),
        soil_infiltration_disputed=disputed,
        soil_source_ref=str(soil.get("source_ref", "")) if soil else "",
        notes=notes,
    )


def site_context_for_point(lat: float, lon: float) -> SiteContext:
    """Resolve a map pin to its hobli and the site data attached to it."""
    check_in_bounds(lat, lon)
    hoblis = load_hoblis()
    distances = hoblis.apply(
        lambda r: haversine_km(lat, lon, float(r["lat"]), float(r["lon"])), axis=1
    )
    idx = distances.idxmin()
    return _context_from_row(hoblis.loc[idx], float(distances.loc[idx]), lat, lon)


def site_context_for_hobli(name: str) -> SiteContext:
    """Resolve a hobli by name, using its centroid as the location."""
    hoblis = load_hoblis()
    key = name.strip().casefold()
    match = hoblis[hoblis["hobli_key"] == key]
    if match.empty:
        match = hoblis[hoblis["hobli_key"].str.startswith(key)]
    if match.empty:
        raise UnknownHobliError(
            f"unknown hobli {name!r}; see GET /api/hoblis for the {len(hoblis)} covered hoblis"
        )
    row = match.iloc[0]
    return _context_from_row(row, 0.0, float(row["lat"]), float(row["lon"]))


def list_hoblis() -> list[dict[str, Any]]:
    """Compact hobli list for the map and the dropdown fallback."""
    hoblis = load_hoblis()
    return [
        {
            "hobli": str(r["hobli"]),
            "taluk": str(r["taluk"]),
            "district": str(r["district"]),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "annual_rainfall_mm": float(r["annual_rainfall_mm"]),
            "soil_type": str(r["soil_type"]),
            "gw_depth_m": float(r["gw_depth_m"]),
            "gw_category": str(r["gw_category"]),
        }
        for _, r in hoblis.iterrows()
    ]


def list_zones() -> list[dict[str, Any]]:
    return [
        {
            "zone": str(r["zone"]),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "wards": int(r["wards"]),
            "population_density_per_km2": float(r["population_density_per_km2"]),
            "density_band": str(r["density_band"]),
        }
        for _, r in load_zones().iterrows()
    ]
