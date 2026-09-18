"""
Analysis API Routes.
POST /api/analyzeContour — Main endpoint for contour map analysis.
POST /api/findCatchment  — Alias for the same endpoint.
"""

import logging
import time

from fastapi import APIRouter, File, Query, UploadFile, HTTPException

from config import AnalysisConfig
from models.schemas import (
    AnalysisMetadata,
    AnalysisResponse,
    CatchmentInfo,
    DEMSize,
    DEMStats,
    ElevationRange,
    ErrorResponse,
    Extent,
    PondCandidate,
)
from services.kml_parser import parse_kml_bytes, contour_lines_to_geojson
from services.dem_generator import generate_dem
from services.hydrology import run_hydrology_pipeline
from services.candidate_finder import find_candidates
from services.catchment_service import delineate_catchment
from services.volume_estimator import estimate_volume
from services.water_body_service import (
    fetch_water_bodies,
    filter_candidates_by_water,
    water_zone_to_geojson,
)
from utils.raster_utils import pixel_to_utm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.post(
    "/analyzeContour",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
    summary="Analyze a contour map for pond planning",
    description=(
        "Upload a KML or KMZ contour map file. The system will parse the contour lines, "
        "generate a Digital Elevation Model (DEM), perform hydrological analysis, "
        "and identify optimal pond candidate sites with catchment information. "
        "Candidates falling within existing water bodies (rivers, lakes) are automatically excluded."
    ),
)
async def analyze_contour(
    contour_map: UploadFile = File(..., description="KML or KMZ contour map file"),
    resolution: float = Query(
        5.0, ge=1.0, le=50.0,
        description="DEM grid resolution in meters (smaller = more precise but slower)"
    ),
    num_candidates: int = Query(
        5, ge=1, le=20,
        description="Number of pond candidate sites to return"
    ),
):
    """
    Analyze a contour map and return pond planning information.

    This endpoint:
    1. Parses the KML/KMZ file to extract contour lines with elevation
    2. Interpolates a continuous DEM from the contour data
    3. Runs hydrological analysis (depression detection, flow direction, accumulation, TWI)
    4. Identifies candidate pond sites using composite terrain scoring
    5. Queries OpenStreetMap for existing water bodies (rivers, lakes, ponds, streams)
    6. Filters out candidates that fall within existing water bodies
    7. Delineates catchment areas for each valid candidate
    8. Estimates storage volumes for each candidate

    All results are derived from the uploaded contour map — no hardcoded coordinates or values.
    """
    start_time = time.time()

    # ── Validate file ────────────────────────────────────────────────────

    filename = contour_map.filename or "upload.kml"
    if not filename.lower().endswith((".kml", ".kmz")):
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Please upload a KML or KMZ file."
        )

    file_bytes = await contour_map.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    logger.info(f"Received file: {filename} ({len(file_bytes)} bytes)")

    # ── Step 1: Parse KML ────────────────────────────────────────────────

    try:
        kml_data = parse_kml_bytes(file_bytes, filename)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    t1 = time.time()
    logger.info(f"KML parsing: {t1 - start_time:.2f}s")

    # ── Step 2: Generate DEM ─────────────────────────────────────────────

    # We request extra candidates internally so that after water-body
    # filtering and depression deduplication we still have enough.
    OVERSAMPLE_FACTOR = 8
    config = AnalysisConfig(
        dem_resolution_m=resolution,
        num_candidates=num_candidates * OVERSAMPLE_FACTOR,
    )

    try:
        dem_result = generate_dem(
            vertices=kml_data["vertices"],
            resolution_m=config.dem_resolution_m,
            max_vertices=config.max_contour_vertices,
            kernel=config.rbf_kernel,
        )
    except Exception as e:
        logger.error(f"DEM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"DEM generation failed: {e}")

    dem = dem_result["dem"]
    grid_params = dem_result["grid_params"]
    to_wgs84 = dem_result["to_wgs84"]

    t2 = time.time()
    logger.info(f"DEM generation: {t2 - t1:.2f}s")

    # ── Step 3: Hydrological analysis ────────────────────────────────────

    try:
        hydro = run_hydrology_pipeline(dem, config.dem_resolution_m)
    except Exception as e:
        logger.error(f"Hydrology pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Hydrology analysis failed: {e}")

    t3 = time.time()
    logger.info(f"Hydrology pipeline: {t3 - t2:.2f}s")

    # ── Step 4: Find candidates (oversampled) ────────────────────────────

    try:
        candidates = find_candidates(hydro, dem, grid_params, config)
    except Exception as e:
        logger.error(f"Candidate finding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Candidate selection failed: {e}")

    t4 = time.time()
    logger.info(f"Candidate finding: {t4 - t3:.2f}s, {len(candidates)} raw candidates")

    # ── Step 5: Fetch water bodies from OSM & filter ─────────────────────

    min_lon, max_lon, min_lat, max_lat = kml_data["extent"]

    try:
        water_zone = fetch_water_bodies(min_lon, max_lon, min_lat, max_lat)
    except Exception as e:
        logger.warning(f"Water body detection failed (will proceed without filtering): {e}")
        water_zone = None

    t5 = time.time()
    logger.info(f"Water body detection: {t5 - t4:.2f}s")

    # Filter candidates against water bodies
    candidates = filter_candidates_by_water(
        candidates, water_zone, grid_params, to_wgs84, config.dem_resolution_m
    )

    # Take only the user-requested number after filtering
    candidates = candidates[:num_candidates]

    t5b = time.time()
    logger.info(
        f"Water body filtering: {t5b - t5:.2f}s, "
        f"returning {len(candidates)} candidates to user"
    )

    # ── Step 6: Catchment + Volume for each remaining candidate ──────────

    pond_candidates_raw = []
    for i, cand in enumerate(candidates):
        # Catchment delineation
        try:
            catchment = delineate_catchment(
                flow_dir=hydro["flow_dir"],
                pour_row=cand["row"],
                pour_col=cand["col"],
                grid_params=grid_params,
                to_wgs84=to_wgs84,
                resolution_m=config.dem_resolution_m,
            )
        except Exception as e:
            logger.warning(f"Catchment delineation failed for candidate {i}: {e}")
            catchment = {
                "area_km2": 0.0,
                "polygon": {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[]]}},
            }

        # Volume estimation (with pond footprint polygon)
        try:
            volume = estimate_volume(
                dem, hydro["fill_depth"], cand, grid_params,
                to_wgs84=to_wgs84, resolution_m=config.dem_resolution_m,
            )
        except Exception as e:
            logger.warning(f"Volume estimation failed for candidate {i}: {e}")
            volume = {
                "volume_m3": 0.0,
                "surface_area_m2": 0.0,
                "pond_footprint": None,
                "basin_label": -1,
            }

        # Convert grid position to lon/lat
        utm_x, utm_y = pixel_to_utm(
            cand["row"], cand["col"],
            grid_params["origin_x"], grid_params["origin_y"],
            config.dem_resolution_m,
        )
        lon, lat = to_wgs84.transform(utm_x, utm_y)

        pond_candidates_raw.append({
            "score": cand["score"],
            "location": {"lat": round(float(lat), 6), "lon": round(float(lon), 6)},
            "elevation_m": round(cand["elevation"], 2),
            "depression_depth_m": round(cand["depression_depth"], 2),
            "volume": volume,
            "catchment": catchment,
            "twi": round(cand["twi"], 2),
            "slope_deg": round(cand["slope_deg"], 2),
            "basin_label": volume.get("basin_label", -1),
        })

    # ── Deduplicate: keep only the best candidate per depression basin ────

    seen_basins = set()
    deduped = []
    for pc in pond_candidates_raw:
        bl = pc["basin_label"]
        if bl >= 0 and bl in seen_basins:
            logger.info(
                f"  Skipping duplicate candidate at ({pc['location']['lat']}, "
                f"{pc['location']['lon']}) — same depression basin #{bl}"
            )
            continue
        if bl >= 0:
            seen_basins.add(bl)
        deduped.append(pc)

    # Trim to user-requested count and assign final ranks
    deduped = deduped[:num_candidates]

    pond_candidates = []
    for i, pc in enumerate(deduped):
        pond_candidates.append(PondCandidate(
            rank=i + 1,
            score=round(pc["score"], 4),
            location=pc["location"],
            elevation_m=pc["elevation_m"],
            depression_depth_m=pc["depression_depth_m"],
            estimated_volume_m3=round(pc["volume"]["volume_m3"], 2),
            estimated_surface_area_m2=round(pc["volume"]["surface_area_m2"], 2),
            catchment=CatchmentInfo(
                area_km2=pc["catchment"]["area_km2"],
                polygon=pc["catchment"]["polygon"],
            ),
            pond_footprint=pc["volume"].get("pond_footprint"),
            twi=pc["twi"],
            slope_deg=pc["slope_deg"],
        ))

    t6 = time.time()
    logger.info(f"Catchment + volume + dedup: {t6 - t5b:.2f}s, {len(pond_candidates)} final candidates")

    # ── Build response ───────────────────────────────────────────────────

    min_elev, max_elev = kml_data["elevation_range"]

    # Convert contour lines to GeoJSON
    contours_geojson = contour_lines_to_geojson(kml_data["contour_lines"])

    # Get water body GeoJSON for frontend visualization
    water_geojson = water_zone_to_geojson(water_zone)

    response = AnalysisResponse(
        status="success",
        metadata=AnalysisMetadata(
            filename=filename,
            contour_count=kml_data["contour_count"],
            elevation_range=ElevationRange(min=min_elev, max=max_elev),
            extent=Extent(
                min_lon=min_lon, max_lon=max_lon,
                min_lat=min_lat, max_lat=max_lat,
            ),
            dem_resolution_m=resolution,
            dem_size=DEMSize(rows=dem.shape[0], cols=dem.shape[1]),
        ),
        contours_geojson=contours_geojson,
        candidates=pond_candidates,
        dem_stats=DEMStats(
            mean_elevation=round(float(dem.mean()), 2),
            std_elevation=round(float(dem.std()), 2),
            mean_slope_deg=round(float(hydro["slope_deg"].mean()), 2),
        ),
        water_bodies_geojson=water_geojson,
    )

    total_time = time.time() - start_time
    logger.info(f"Total analysis time: {total_time:.2f}s")

    return response


# ── Alias endpoint ────────────────────────────────────────────────────────────

@router.post(
    "/findCatchment",
    response_model=AnalysisResponse,
    include_in_schema=False,
)
async def find_catchment(
    contour_map: UploadFile = File(...),
    resolution: float = Query(5.0, ge=1.0, le=50.0),
    num_candidates: int = Query(5, ge=1, le=20),
):
    """Alias for /analyzeContour."""
    return await analyze_contour(contour_map, resolution, num_candidates)
