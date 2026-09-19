"""Health and metrics, used by the watchdog, nginx and the stress tests."""

from __future__ import annotations

from fastapi import APIRouter

import config
from core import pool
from core.cache import get_cache, read_metrics
from core.resilience import breaker_states
from models.schemas import HealthResponse, MetricsResponse

router = APIRouter(tags=["System"])


@router.get("/", response_model=HealthResponse, summary="Service banner")
@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health():
    from services.gazetteer_service import has_local_index
    from services.hydrology_engine import HAS_PYFLWDIR
    from services.rainfall_service import has_local_archive

    return HealthResponse(
        status="healthy",
        version="3.0.0",
        algo_version=config.ALGO_VERSION,
        engine="pyflwdir" if HAS_PYFLWDIR else "numpy_legacy",
        local_gazetteer=has_local_index(),
        local_rainfall=has_local_archive(),
    )


@router.get("/metrics", response_model=MetricsResponse, summary="Counters for the stress tests")
async def metrics():
    cache = get_cache()
    try:
        volume = cache.volume()
        count = len(cache)
    except Exception:  # pragma: no cover
        volume, count = 0, 0

    counters = read_metrics()
    hits = counters.get("cache_hits", 0)
    misses = counters.get("cache_misses", 0)
    total = hits + misses

    return MetricsResponse(
        counters=counters,
        breakers=breaker_states(),
        pool=pool.stats(),
        cache={
            "entries": count,
            "bytes": volume,
            "hit_ratio": round(hits / total, 3) if total else None,
        },
    )
