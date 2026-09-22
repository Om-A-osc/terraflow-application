"""
Analysis of a selected area, and the budget-constrained earthwork filter.

Both endpoints run their CPU work in the process pool under a hard timeout, so
one oversized request cannot stall the worker that other users are waiting on.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

import config
from core import pool
from core.cache import bump
from models.schemas import AnalyzeRequest, EarthworkRequest, ErrorResponse
from services.analysis_service import AnalysisError, analyze_polygon, load_context
from services.earthwork_service import optimise
from services.storage_service import flood_fill_at_level
from utils.vector_utils import mask_to_feature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Analysis"])


@router.post(
    "/analyze",
    responses={400: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
    summary="Analyse a selected area",
    description=(
        "Takes the polygon drawn on the map and returns the catchment of the "
        "selection, ranked pond sites with their footprint and their own "
        "catchment, the storage each can hold, the water each can expect to "
        "collect in a year, and the stage-storage curve the budget filter uses."
    ),
)
async def analyze(request: AnalyzeRequest):
    payload = {
        "geometry": request.geometry,
        "num_sites": request.num_sites,
        "resolution_m": request.resolution_m,
        "dem_source": request.dem_source,
        "hydrology": request.hydrology.model_dump(),
        "hsg": request.hydrology.hsg,
    }
    try:
        return await pool.run(_analyze_task, payload)
    except pool.ComputeTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except AnalysisError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        bump("analyses_failed")
        logger.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


def _analyze_task(payload: dict) -> dict:
    """Runs in a pool worker; arguments are plain data so they pickle cleanly."""
    from models.schemas import HydrologyOverrides

    overrides = HydrologyOverrides(**payload["hydrology"])
    return analyze_polygon(
        payload["geometry"],
        num_sites=payload["num_sites"],
        resolution_m=payload["resolution_m"],
        dem_source=payload["dem_source"],
        hydrology=overrides.to_params(),
        hsg=payload["hsg"],
    )


@router.post(
    "/earthwork",
    responses={404: {"model": ErrorResponse}, 504: {"model": ErrorResponse}},
    summary="How much more water a budget buys",
    description=(
        "Given a budget in rupees and the unit rates for excavating, hauling "
        "and placing soil, searches over excavation depth and bund height for "
        "the design that adds the most storage without exceeding the budget. "
        "Optionally solves the haul allocation as a transportation problem."
    ),
)
async def earthwork(request: EarthworkRequest):
    context = load_context(request.analysis_id)
    if context is None:
        raise HTTPException(
            status_code=404,
            detail="That analysis has expired. Run the analysis again before applying a budget.",
        )
    if request.site_id not in context["sites"]:
        raise HTTPException(status_code=404, detail=f"Unknown site {request.site_id}")

    payload = {
        "analysis_id": request.analysis_id,
        "site_id": request.site_id,
        "budget": request.budget,
        "costs": request.costs.model_dump(),
        "geometry": request.geometry.model_dump(),
        "include_haul_plan": request.include_haul_plan,
    }
    try:
        result = await pool.run(_earthwork_task, payload, timeout=20.0)
    except pool.ComputeTimeout as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Earthwork optimisation failed")
        raise HTTPException(status_code=500, detail=f"Optimisation failed: {exc}") from exc

    bump("earthwork_runs")
    return result


def _earthwork_task(payload: dict) -> dict:
    from affine import Affine

    from models.schemas import CostOverrides, GeometryOverrides

    context = load_context(payload["analysis_id"])
    if context is None:
        raise RuntimeError("The analysis context expired while the request was queued")

    site = context["sites"][payload["site_id"]]
    dem = context["dem"]
    transform = Affine(*context["transform"])
    epsg = context["epsg"]
    resolution_m = context["resolution_m"]
    cell_area = resolution_m ** 2

    # An excavated pond has no basin to flood: it is a designed hole, so the
    # budget makes it deeper or wider rather than raising a level
    if site.get("kind") == "dugout":
        return _to_jsonable(_earthwork_dugout(payload, context, site, resolution_m))

    row, col = site["cell"]["row"], site["cell"]["col"]
    curve = site["stage_storage"]
    spill = curve.get("spill_level_m") or float(dem[row, col])

    # Confine the pond to a window the size of a village structure.  Without
    # this, raising the spill level floods the whole valley on the map and the
    # footprint drawn bears no relation to what a bund could actually hold.
    window = _containment_window(dem.shape, row, col, resolution_m)
    region = context.get("region")
    if region is not None and region.shape == dem.shape and region.any():
        # Same boundary the analysis used: extra storage bought with a budget
        # still has to fit inside the area the user selected.
        window = window & region
        if not window.any():
            window = _containment_window(dem.shape, row, col, resolution_m)

    footprint = flood_fill_at_level(dem, row, col, float(spill), bounds_mask=window)
    if not footprint.any():
        footprint = np.zeros(dem.shape, dtype=bool)
        footprint[row, col] = True

    result = optimise(
        dem=dem,
        footprint_mask=footprint,
        curve=curve,
        cell_area_m2=cell_area,
        resolution_m=resolution_m,
        budget=payload["budget"],
        costs=CostOverrides(**payload["costs"]).to_config(),
        geometry=GeometryOverrides(**payload["geometry"]).to_config(),
        include_haul_plan=payload["include_haul_plan"],
        seed=(row, col),
        containment=window,
    )

    # Footprint the pond would have after the works, for the map overlay
    new_level = float(curve.get("usable_level_m") or spill) + result["design"]["spill_raise_m"]
    new_footprint = flood_fill_at_level(dem, row, col, new_level, bounds_mask=window)
    if _reaches_edge_of(new_footprint, window):
        result.setdefault("warnings", []).append(
            "At this raised level the water would spread beyond what a single bund can "
            "enclose; the footprint shown is clipped and the design needs a survey."
        )
    result["new_footprint"] = mask_to_feature(
        new_footprint, transform, epsg,
        properties={
            "kind": "pond_footprint_after_works",
            "capacity_m3": result["storage"]["new_capacity_m3"],
        },
        simplify_m=resolution_m / 2,
    )
    result["existing_footprint"] = mask_to_feature(
        footprint, transform, epsg,
        properties={"kind": "pond_footprint_before_works"},
        simplify_m=resolution_m / 2,
    )

    # The bund runs around the rim of the enlarged pond
    if result["design"]["spill_raise_m"] > 0:
        from scipy import ndimage

        rim = ndimage.binary_dilation(new_footprint) & ~new_footprint
        result["bund_geometry"] = mask_to_feature(
            rim, transform, epsg,
            properties={
                "kind": "bund",
                "height_m": result["design"]["bund_max_height_m"],
                "length_m": result["design"]["bund_length_m"],
            },
            simplify_m=resolution_m / 2,
        )

    # Haul arrows come back in grid metres; convert them to map coordinates
    if result.get("haul_plan", {}).get("arrows"):
        result["haul_plan"]["arrows"] = _arrows_to_wgs84(
            result["haul_plan"]["arrows"], transform, epsg, resolution_m
        )

    result["site_id"] = payload["site_id"]
    result["analysis_id"] = payload["analysis_id"]
    return _to_jsonable(result)


def _earthwork_dugout(payload: dict, context: dict, site: dict, resolution_m: float) -> dict:
    """Size up an excavated pond for the budget and redraw it at its new size."""
    from shapely import wkt as shapely_wkt

    import config
    from models.schemas import CostOverrides, GeometryOverrides
    from services.analysis_service import _clip_to_selection, _dugout_footprint
    from services.earthwork_service import optimise_dugout
    from utils.vector_utils import polygon_area_m2

    cfg = config.AnalysisConfig()
    geometry = GeometryOverrides(**payload["geometry"]).to_config()
    selection = (
        shapely_wkt.loads(context["selection_wkt"]) if context.get("selection_wkt") else None
    )
    lon, lat = site["location"]["lon"], site["location"]["lat"]

    def footprint(side_m: float, capacity_m3: float = 0.0) -> dict | None:
        return _clip_to_selection(_dugout_footprint(lon, lat, cfg, capacity_m3, side_m=side_m), selection)

    def area_for_side(side_m: float) -> float:
        feature = footprint(side_m)
        return polygon_area_m2(feature["geometry"]) if feature else 0.0

    base_side = float(cfg.dugout_side_m)
    base_area = float(site.get("surface_area_m2") or area_for_side(base_side))
    base_depth = float(site.get("max_depth_m") or cfg.dugout_depth_m)

    # Widen only while the pond still fits the selection: once the clipped
    # shape loses a tenth of its area to the boundary it is running past it
    max_side = base_side
    side = base_side
    while side + 2.0 <= geometry.dugout_max_side_m + 1e-9:
        side += 2.0
        unclipped = polygon_area_m2(_dugout_footprint(lon, lat, cfg, 0.0, side_m=side)["geometry"])
        if unclipped <= 0 or area_for_side(side) < 0.9 * unclipped:
            break
        max_side = side

    result = optimise_dugout(
        base_area, base_depth, base_side, cfg.dugout_side_slope_factor, payload["budget"],
        costs=CostOverrides(**payload["costs"]).to_config(),
        geometry=geometry,
        resolution_m=resolution_m,
        area_for_side=area_for_side,
        max_side_m=max_side,
    )

    new_side = result["design"].get("new_side_m", base_side)
    after = footprint(new_side, result["storage"]["new_capacity_m3"])
    if after is not None:
        after["properties"] = {**after["properties"], "kind": "pond_footprint_after_works"}
    before = footprint(base_side, result["storage"]["base_capacity_m3"])
    if before is not None:
        before["properties"] = {**before["properties"], "kind": "pond_footprint_before_works"}
    result["new_footprint"] = after
    result["existing_footprint"] = before
    result["site_id"] = payload["site_id"]
    result["analysis_id"] = payload["analysis_id"]
    return result


def _arrows_to_wgs84(arrows: list, transform, epsg: int, resolution_m: float) -> list:
    from pyproj import Transformer

    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    out = []
    for arrow in arrows:
        try:
            fx, fy = arrow["from"]
            tx, ty = arrow["to"]
            # Aggregation returned positions in grid metres (col*res, row*res)
            fux, fuy = transform * (fx / resolution_m, fy / resolution_m)
            tux, tuy = transform * (tx / resolution_m, ty / resolution_m)
            flon, flat = to_wgs.transform(fux, fuy)
            tlon, tlat = to_wgs.transform(tux, tuy)
            out.append({
                "from": [round(flon, 6), round(flat, 6)],
                "to": [round(tlon, 6), round(tlat, 6)],
                "volume_m3": arrow["volume_m3"],
                "distance_m": arrow["distance_m"],
            })
        except Exception:  # noqa: BLE001
            continue
    return out


def _to_jsonable(value):
    """NumPy scalars leak in through the optimiser; JSON cannot encode them."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _containment_window(shape_hw: tuple, row: int, col: int, resolution_m: float,
                        extent_m: float = 400.0) -> np.ndarray:
    """Square mask of ``extent_m`` half-width around a site."""
    radius = max(4, int(round(extent_m / resolution_m)))
    window = np.zeros(shape_hw, dtype=bool)
    r0, r1 = max(0, row - radius), min(shape_hw[0], row + radius + 1)
    c0, c1 = max(0, col - radius), min(shape_hw[1], col + radius + 1)
    window[r0:r1, c0:c1] = True
    return window


def _reaches_edge_of(wet: np.ndarray, window: np.ndarray) -> bool:
    from scipy import ndimage

    boundary = window & ~ndimage.binary_erosion(window)
    return bool((wet & boundary).any())
