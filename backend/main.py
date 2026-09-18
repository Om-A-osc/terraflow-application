"""
TerraFlow — FastAPI application.

Search any Indian village, select an area on the map, and get the suggested
pond location, its catchment and the water it can collect, plus a budget filter
that says how much more it could store after paid earthwork.

Run one of these per compute node behind the nginx edge:

    uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2 --limit-concurrency 64
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
from core import pool
from routes.analysis import router as contour_router
from routes.analyze import router as analyze_router
from routes.system import router as system_router
from routes.villages import router as villages_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the pool and warm the compiled kernels before serving traffic."""
    logger.info("Starting TerraFlow API (algorithm version %s)", config.ALGO_VERSION)
    try:
        from services.gazetteer_service import has_local_index
        from services.hydrology_engine import HAS_PYFLWDIR, warm_up
        from services.rainfall_service import has_local_archive

        warm_up()
        logger.info(
            "Engine %s | local gazetteer %s | local rainfall archive %s",
            "pyflwdir" if HAS_PYFLWDIR else "numpy fallback",
            "yes" if has_local_index() else "no (Photon fallback)",
            "yes" if has_local_archive() else "no (NASA POWER fallback)",
        )
        pool.get_pool()
    except Exception as exc:  # noqa: BLE001 - never block start-up
        logger.warning("Start-up warm-up incomplete: %s", exc)

    yield

    pool.shutdown_pool()
    logger.info("TerraFlow API stopped")


app = FastAPI(
    title="TerraFlow API",
    version="3.0.0",
    description=(
        "Rainwater pond planning for Indian villages: village search, area "
        "selection, catchment delineation, storage and yield estimation, and a "
        "budget-constrained earthwork optimiser.\n\n"
        "Elevation comes from Copernicus GLO-30 (AWS Open Data) with the Tilezen "
        "terrain tiles as a fallback, rainfall from the IMD gridded archive or "
        "NASA POWER, land cover and constraints from OpenStreetMap, and the "
        "village index from GeoNames."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Server-Timing so the browser devtools show where the time went."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["Server-Timing"] = f"app;dur={elapsed_ms:.1f}"
    if elapsed_ms > 5000:
        logger.info("%s %s took %.1f s", request.method, request.url.path, elapsed_ms / 1000)
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """A bad polygon or an out-of-range parameter is the caller's to fix."""
    return JSONResponse(status_code=400, content={"status": "error", "detail": str(exc)})


app.include_router(system_router)
app.include_router(villages_router)
app.include_router(analyze_router)
app.include_router(contour_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=bool(int(__import__("os").getenv("POND_RELOAD", "0"))),
        limit_concurrency=64,
    )
