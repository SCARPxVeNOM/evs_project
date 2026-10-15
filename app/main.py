"""Application entry point.

    uvicorn app.main:app --reload

serves the API under ``/api`` and the Leaflet web UI at ``/``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import router
from app.config import WEB_DIR
from app.datasets import LocationOutOfRangeError, UnknownHobliError

DESCRIPTION = """
Recommends a rainwater harvesting structure for a location in Bengaluru from
rainfall, soil, groundwater, roof and household data.

Estimates the harvestable volume, recommends a structure type, sizes and costs it,
compares supply with demand month by month and reports the likely water saving.
Outputs are planning estimates, not engineering designs.
"""

app = FastAPI(
    title="Rainwater Harvesting Recommendation System - Bengaluru",
    description=DESCRIPTION,
    version=__version__,
)

# Open CORS: this is a local teaching/demo service with no authentication and no
# user data. Lock it down before putting it on a public host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router)


@app.exception_handler(LocationOutOfRangeError)
async def _out_of_range(_request, exc: LocationOutOfRangeError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(UnknownHobliError)
async def _unknown_hobli(_request, exc: UnknownHobliError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
