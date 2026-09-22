"""
Village search, detail and warm-up.

Selecting a village kicks off the expensive work in the background so that by
the time the user has drawn a polygon the elevation window, the OSM features
and the routed hydrology are already on local disk.  The warm-up returns at
once with a job id; the client polls it and can draw in the meantime, because a
polygon drawn early simply takes the cold path.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, HTTPException, Query

import config
from core import pool
from core.cache import get_cache
from core.resilience import DataQuality
from models.schemas import (
    JobStatus, VillageDetail, VillageSearchResponse, VillageSuggestion, WarmResponse,
)
from services import gazetteer_service as gaz
from services.bundle_service import (
    get_bundle, has_bundle, window_for_selection,
)
from services.rainfall_service import get_daily_rainfall, rainfall_summary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Villages"])

# Half-width of the window prepared for a village, in metres
VILLAGE_WINDOW_M = 2500.0
JOB_TTL_S = 900


def _village_bounds(lat: float, lon: float) -> tuple:
    """
    The shared window prepared for a village.

    Built the same way the analysis builds its own window, so warming a village
    fills exactly the cache entry that the first drawn polygon will ask for.
    """
    return window_for_selection((lon, lat, lon, lat))


def _job_key(job_id: str) -> str:
    return f"job:{config.ALGO_VERSION}:{job_id}"


def _set_job(job_id: str, **fields) -> dict:
    cache = get_cache()
    state = cache.get(_job_key(job_id), default={}) or {}
    state.update(fields)
    cache.set(_job_key(job_id), state, expire=JOB_TTL_S)
    return state


@router.get(
    "/villages",
    response_model=VillageSearchResponse,
    summary="Search for a village by name",
    description=(
        "Prefix search over the local GeoNames gazetteer (about 558,000 populated "
        "places in India), ranked by text match, population and distance from the "
        "current map centre.  Falls back to Photon when the local index misses."
    ),
)
async def search_villages(
    q: str = Query(..., min_length=2, max_length=80, description="Typed text"),
    lat: float | None = Query(None, ge=-90, le=90, description="Map centre, to bias results"),
    lon: float | None = Query(None, ge=-180, le=180),
    limit: int = Query(8, ge=1, le=20),
):
    try:
        return gaz.search(q, lat, lon, limit)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Village search failed")
        raise HTTPException(status_code=503, detail=f"Search is unavailable: {exc}") from exc


@router.get(
    "/villages/{village_id}",
    response_model=VillageDetail,
    summary="Village detail, boundary and rainfall",
)
async def village_detail(
    village_id: str,
    lat: float | None = Query(None, description="Required for a Photon result"),
    lon: float | None = Query(None),
    name: str | None = Query(None),
):
    village = gaz.get_village(village_id)
    if village is None:
        if lat is None or lon is None:
            raise HTTPException(
                status_code=404,
                detail="Unknown village id. Pass lat and lon for a result that came from Photon.",
            )
        village = {
            "id": village_id, "name": name or "Selected place",
            "subdistrict": None, "district": None, "state": None,
            "lat": lat, "lon": lon, "population": 0, "kind": "village",
            "source": "client",
        }

    boundary = gaz.boundary_for(village)

    rainfall: dict = {}
    try:
        series = get_daily_rainfall(village["lat"], village["lon"], quality=DataQuality())
        rainfall = rainfall_summary(series)
    except Exception as exc:  # noqa: BLE001
        logger.info("Rainfall summary unavailable for %s: %s", village_id, exc)

    bounds = _village_bounds(village["lat"], village["lon"])
    return VillageDetail(
        village=VillageSuggestion(**village),
        boundary=boundary,
        suggested_area=boundary,
        rainfall=rainfall,
        bundle_ready=has_bundle(bounds, config.AnalysisConfig().dem_resolution_m),
    )


@router.post(
    "/villages/{village_id}/warm",
    response_model=WarmResponse,
    summary="Prepare the elevation and hydrology for a village",
    description=(
        "Starts fetching the elevation window, the OSM features and the routed "
        "hydrology, and caches them so the first polygon the user draws is "
        "answered from local disk.  Returns at once: poll /api/jobs/{job_id} for "
        "progress.  Drawing before it finishes is fine, it just takes longer."
    ),
)
async def warm_village(
    village_id: str,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    resolution_m: float = Query(30.0, ge=5, le=90),
):
    bounds = _village_bounds(lat, lon)
    if has_bundle(bounds, resolution_m):
        return WarmResponse(status="ready", village_id=village_id, detail="Already prepared")

    job_id = uuid.uuid4().hex[:12]
    _set_job(job_id, status="running", stage="fetching elevation and map data",
             village_id=village_id, started=time.time())

    async def _run() -> None:
        started = time.perf_counter()
        try:
            await pool.run(_warm_task, bounds, resolution_m, timeout=180.0)
            _set_job(job_id, status="done", stage="ready",
                     elapsed_s=round(time.perf_counter() - started, 2))
            logger.info("Warmed %s in %.1f s", village_id, time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001
            _set_job(job_id, status="failed", stage="failed", detail=str(exc))
            logger.warning("Warm-up failed for %s: %s", village_id, exc)

    asyncio.create_task(_run())
    return WarmResponse(status="started", village_id=village_id, job_id=job_id,
                        detail="Preparing this village in the background")


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    summary="Progress of a background warm-up",
)
async def job_status(job_id: str):
    state = get_cache().get(_job_key(job_id), default=None)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown or expired job")
    return JobStatus(
        job_id=job_id,
        status=state.get("status", "running"),
        stage=state.get("stage"),
        detail=state.get("detail"),
        elapsed_s=state.get("elapsed_s"),
    )


def _warm_task(bounds: tuple, resolution_m: float) -> bool:
    """
    Runs in a pool worker: build and cache everything the first analysis needs.

    That is the terrain bundle *and* the climate series, because the rainfall
    and evaporation lookups are the other network round trip on the cold path.
    """
    quality = DataQuality()
    get_bundle(bounds, resolution_m, quality=quality)

    west, south, east, north = bounds
    lat, lon = (south + north) / 2, (west + east) / 2
    try:
        from services.evaporation_service import monthly_et0

        get_daily_rainfall(lat, lon, quality=quality)
        monthly_et0(lat, lon, quality)
    except Exception as exc:  # noqa: BLE001 - terrain alone is still worth caching
        logger.info("Climate warm-up skipped: %s", exc)
    return True
