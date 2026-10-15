"""HTTP routes.

    GET  /api/health       liveness and what data and model are loaded
    GET  /api/parameters   form defaults and allowed values for the UI
    GET  /api/hoblis       covered hoblis with centroids, for the map
    GET  /api/zones        BBMP zones with population density
    GET  /api/rainfall     monthly rainfall for a pin or a hobli
    GET  /api/location     site lookup for a pin: hobli, soil, groundwater, zone
    GET  /api/model        trained classifier: metrics and feature importances
    POST /api/recommend    the full recommendation report
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from app import __version__
from app.config import open_area_types, param, parameters, roof_types
from app.core import ml as ml_layer
from app.core.engine import ENGINE_MODES, recommend
from app.core.recommender import ALL_LABELS, HOUSE_TYPES, LABEL_TITLES, WATER_SOURCES
from app.datasets import (
    LocationOutOfRangeError,
    SiteContext,
    UnknownHobliError,
    list_hoblis,
    list_zones,
    load_hoblis,
    site_context_for_hobli,
    site_context_for_point,
)
from app.schemas import RecommendRequest

router = APIRouter(prefix="/api")


def _resolve_site(lat: float | None, lon: float | None, hobli: str | None) -> SiteContext:
    try:
        if lat is not None and lon is not None:
            return site_context_for_point(lat, lon)
        if hobli:
            return site_context_for_hobli(hobli)
    except (LocationOutOfRangeError, UnknownHobliError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(status_code=400, detail="give either lat and lon, or a hobli name")


@router.get("/health", summary="Service and data status")
def health() -> dict[str, Any]:
    model = ml_layer.load_model()
    return {
        "status": "ok",
        "app_version": __version__,
        "hoblis_loaded": int(len(load_hoblis())),
        "zones_loaded": len(list_zones()),
        "model_trained": model is not None,
    }


@router.get("/parameters", summary="Form defaults, allowed values and key constants")
def get_parameters() -> dict[str, Any]:
    params = parameters()
    return {
        "roof_types": roof_types(),
        "open_area_types": open_area_types(),
        "house_types": HOUSE_TYPES,
        "water_sources": WATER_SOURCES,
        "engine_modes": list(ENGINE_MODES),
        "structure_labels": {label: LABEL_TITLES[label] for label in ALL_LABELS},
        "runoff_coefficients": params["runoff_coefficients"],
        "defaults": {
            "roof_type": "rcc_flat",
            "open_area_type": "paved_concrete",
            "household_size": 4,
            "lpcd": param("demand", "default_lpcd"),
            "non_potable_share": param("demand", "non_potable_share"),
            "storage_design_days": param("storage_design", "design_days"),
            "slope_pct": 2.0,
        },
        "bounding_box": params["bounding_box"],
        "byelaw": params["byelaw"],
        "design_storm_mm_per_hr": param("recharge_design", "design_storm_mm_per_hr"),
        "price_base_year": params["price_base_year"],
    }


@router.get("/hoblis", summary="Hoblis covered by the dataset")
def get_hoblis() -> dict[str, Any]:
    hoblis = list_hoblis()
    return {"count": len(hoblis), "hoblis": hoblis}


@router.get("/zones", summary="BBMP administrative zones")
def get_zones() -> dict[str, Any]:
    zones = list_zones()
    return {"count": len(zones), "zones": zones}


@router.get("/location", summary="Site data for a map pin")
def get_location(
    lat: float | None = Query(None, ge=-90, le=90),
    lon: float | None = Query(None, ge=-180, le=180),
    hobli: str | None = Query(None),
) -> dict[str, Any]:
    site = _resolve_site(lat, lon, hobli)
    return site.as_dict()


@router.get("/rainfall", summary="Monthly rainfall for a location")
def get_rainfall(
    lat: float | None = Query(None, ge=-90, le=90),
    lon: float | None = Query(None, ge=-180, le=180),
    hobli: str | None = Query(None),
) -> dict[str, Any]:
    site = _resolve_site(lat, lon, hobli)
    return {
        "hobli": site.hobli,
        "taluk": site.taluk,
        "district": site.district,
        "annual_rainfall_mm": site.annual_rainfall_mm,
        "annual_rainy_days": site.annual_rainy_days,
        "data_grade": site.data_grade,
        "source_ref": site.source_ref,
        "monthly": [
            {
                "month": m.month,
                "month_name": m.month_name,
                "rainfall_mm": m.rainfall_mm,
                "rainy_days": m.rainy_days,
            }
            for m in site.monthly_rainfall
        ],
    }


@router.get("/model", summary="Trained classifier metrics and feature importances")
def get_model() -> dict[str, Any]:
    info = ml_layer.model_info()
    info["training_metrics_file"] = ml_layer.read_metrics()
    return info


@router.post("/recommend", summary="Full rainwater harvesting recommendation")
def post_recommend(request: RecommendRequest) -> dict[str, Any]:
    site = _resolve_site(request.lat, request.lon, request.hobli)
    try:
        return recommend(site, request.to_inputs())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
