"""Request validation for the HTTP API.

Only requests are strictly typed. Responses are plain dictionaries assembled by
:func:`app.core.engine.recommend`; their shape is documented in the README and in
``docs/API.md`` rather than mirrored in a second set of models that would have to
be kept in step with the engine.

Allowed values for the categorical fields come from ``data/parameters.json`` and
``app.core.recommender``, so adding a roof type to the parameter file is enough -
no code change is needed here.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import open_area_types, roof_types
from app.core.engine import ENGINE_MODES
from app.core.recommender import HOUSE_TYPES, WATER_SOURCES


class RecommendRequest(BaseModel):
    """A household's site description."""

    lat: float | None = Field(None, ge=-90, le=90, description="Latitude of the map pin")
    lon: float | None = Field(None, ge=-180, le=180, description="Longitude of the map pin")
    hobli: str | None = Field(None, description="Hobli name, used when no pin is given")

    roof_area_m2: float = Field(..., gt=0, le=50000, description="Roof catchment area")
    roof_type: str = Field("rcc_flat", description="Roof surface type")
    open_area_m2: float = Field(0.0, ge=0, le=50000, description="Open ground available on the plot")
    open_area_type: str = Field("paved_concrete", description="Surface of the open area")

    household_size: int = Field(4, ge=1, le=500, description="Number of residents")
    daily_use_lpcd: float | None = Field(
        None, gt=0, le=1000, description="Stated water use in litres per person per day"
    )
    monthly_consumption_litres: float | None = Field(
        None, gt=0, le=10_000_000, description="Metered monthly consumption, if known"
    )
    house_type: str = Field("independent", description="independent, apartment, villa or row_house")
    water_source: str = Field("cauvery", description="Current main water source")
    has_borewell: bool = Field(False, description="Whether a borewell exists on the plot")

    plot_area_m2: float | None = Field(
        None, gt=0, le=500000, description="Plot area, needed for the byelaw check"
    )
    slope_pct: float = Field(2.0, ge=0, le=60, description="Average ground slope in percent")
    new_building: bool = Field(False, description="Whether this is a new building under construction")

    existing_storage_litres: float = Field(
        0.0,
        ge=0,
        le=10_000_000,
        description="Capacity of a sump or tank already on site that the roof can feed",
    )
    storage_design_days: int | None = Field(
        None, ge=1, le=60, description="Dry-spell days the tank should bridge"
    )
    tanker_cost_per_load_inr: float | None = Field(
        None, gt=0, le=100000, description="Local tanker price per load, if known"
    )
    engine: str = Field("auto", description="auto, rules or ml")

    @field_validator("roof_type")
    @classmethod
    def _check_roof_type(cls, value: str) -> str:
        if value not in roof_types():
            raise ValueError(f"roof_type must be one of: {', '.join(roof_types())}")
        return value

    @field_validator("open_area_type")
    @classmethod
    def _check_open_area_type(cls, value: str) -> str:
        if value not in open_area_types():
            raise ValueError(f"open_area_type must be one of: {', '.join(open_area_types())}")
        return value

    @field_validator("house_type")
    @classmethod
    def _check_house_type(cls, value: str) -> str:
        if value not in HOUSE_TYPES:
            raise ValueError(f"house_type must be one of: {', '.join(HOUSE_TYPES)}")
        return value

    @field_validator("water_source")
    @classmethod
    def _check_water_source(cls, value: str) -> str:
        if value not in WATER_SOURCES:
            raise ValueError(f"water_source must be one of: {', '.join(WATER_SOURCES)}")
        return value

    @field_validator("engine")
    @classmethod
    def _check_engine(cls, value: str) -> str:
        if value not in ENGINE_MODES:
            raise ValueError(f"engine must be one of: {', '.join(ENGINE_MODES)}")
        return value

    @model_validator(mode="after")
    def _check_location(self) -> "RecommendRequest":
        has_pin = self.lat is not None and self.lon is not None
        if not has_pin and not self.hobli:
            raise ValueError("give either a map pin (lat and lon) or a hobli name")
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be given together")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "lat": 12.9352,
                "lon": 77.6245,
                "roof_area_m2": 120,
                "roof_type": "rcc_flat",
                "open_area_m2": 40,
                "open_area_type": "paved_concrete",
                "household_size": 4,
                "house_type": "independent",
                "water_source": "cauvery_and_borewell",
                "has_borewell": True,
                "plot_area_m2": 223,
                "engine": "auto",
            }
        }
    }

    def to_inputs(self) -> Any:
        from app.core.engine import HouseholdInputs

        return HouseholdInputs(
            roof_area_m2=self.roof_area_m2,
            roof_type=self.roof_type,
            open_area_m2=self.open_area_m2,
            open_area_type=self.open_area_type,
            household_size=self.household_size,
            daily_use_lpcd=self.daily_use_lpcd,
            house_type=self.house_type,
            water_source=self.water_source,
            has_borewell=self.has_borewell,
            plot_area_m2=self.plot_area_m2,
            slope_pct=self.slope_pct,
            monthly_consumption_litres=self.monthly_consumption_litres,
            existing_storage_litres=self.existing_storage_litres,
            storage_design_days=self.storage_design_days,
            tanker_cost_per_load_inr=self.tanker_cost_per_load_inr,
            new_building=self.new_building,
            engine=self.engine,
        )
