"""
Assembling and caching a hydrology bundle for an area.

A bundle is the expensive part of the work: fetch the elevation window, fetch
the OSM features, correct the surface model, then fill, route and accumulate.
It is cached per rounded bounding box and algorithm version, so the first
polygon drawn in a village pays for it and every later one is nearly free.

Selecting a village kicks this off in the background, which is why the
interactive path usually sees a warm bundle.
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict

import numpy as np

import config
from core.cache import cached_call, get_cache, make_key
from core.resilience import DataQuality
from services import hydrology_engine as he
from services.dem_service import fetch_dem
from services.osm_service import fetch_osm, rasterize_constraints

logger = logging.getLogger(__name__)

# Rasterised constraint masks, keyed by bundle.  They are derived entirely from
# the cached OSM features and the bundle's grid, so they are the same on every
# request for a window; burning several hundred polygons again each time was
# the largest remaining cost on the warm path.  Held in the worker process
# rather than on disk because the masks are large and cheap to rebuild.
_CONSTRAINT_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CONSTRAINT_CACHE_MAX = 4


# Windows are snapped to this grid in degrees (about 4.4 km).
#
# The quantum has to be comfortably larger than the jitter between two polygons
# a user might draw in the same village.  At 0.02 degrees a few hundred metres
# of difference pushed the padded window across a cell boundary and forced a
# fresh elevation and OpenStreetMap fetch, which showed up as a ten-second tail
# on the warm path under load.
BOUNDS_QUANTUM_DEG = 0.04


def quantise_bounds(bounds: tuple, quantum: float = BOUNDS_QUANTUM_DEG) -> tuple:
    """Snap a bbox outward to a fixed grid so nearby requests share a window."""
    west, south, east, north = bounds
    return (
        math.floor(west / quantum) * quantum,
        math.floor(south / quantum) * quantum,
        math.ceil(east / quantum) * quantum,
        math.ceil(north / quantum) * quantum,
    )


def window_for_selection(
    bounds: tuple, rings: int = 1, quantum: float = BOUNDS_QUANTUM_DEG
) -> tuple:
    """
    The analysis window for a selection: the grid cells it touches, plus a ring
    of whole cells around them for the catchment to run into.

    Padding by a distance and *then* snapping is not enough, because the padded
    extent depends on the size of the polygon, so two areas drawn in the same
    village still landed in different windows and each paid for its own
    elevation and OpenStreetMap fetch.  Expanding by whole cells makes the
    window a function of position alone: every polygon inside the same cell
    shares one cached bundle.
    """
    west, south, east, north = quantise_bounds(bounds, quantum)
    pad = rings * quantum
    return (
        round(west - pad, 6),
        round(south - pad, 6),
        round(east + pad, 6),
        round(north + pad, 6),
    )


def bundle_key(bounds: tuple, resolution_m: float, dem_source: str = "copernicus") -> str:
    west, south, east, north = [round(float(v), 3) for v in bounds]
    return make_key("bundle", west, south, east, north, resolution_m, dem_source)


def has_bundle(bounds: tuple, resolution_m: float, dem_source: str = "copernicus") -> bool:
    return bundle_key(bounds, resolution_m, dem_source) in get_cache()


def get_bundle(
    bounds: tuple,
    resolution_m: float = 30.0,
    dem_source: str = "copernicus",
    quality: DataQuality | None = None,
) -> dict:
    """
    Return ``{bundle, constraints, osm, bounds, resolution_m}`` for a bbox.

    The constraint rasters are rebuilt from the cached vector features rather
    than stored, because the masks are large and cheap to burn again.
    """
    quality = quality or DataQuality()
    key = bundle_key(bounds, resolution_m, dem_source)

    payload = cached_call(
        key,
        lambda: _build(bounds, resolution_m, dem_source, quality),
        expire=config.CACHE_TTL_BUNDLE,
    )

    bundle = he.HydroBundle.from_payload(payload["bundle"])
    quality.dem_source = bundle.dem_source
    quality.terrain_corrected = payload.get("terrain_corrected", False)

    feats = fetch_osm(bounds, quality)

    cached = _CONSTRAINT_CACHE.get(key)
    if cached is not None:
        _CONSTRAINT_CACHE.move_to_end(key)
        constraints = cached
    else:
        constraints = rasterize_constraints(feats, bundle.shape, bundle.transform, bundle.epsg)
        _CONSTRAINT_CACHE[key] = constraints
        while len(_CONSTRAINT_CACHE) > _CONSTRAINT_CACHE_MAX:
            _CONSTRAINT_CACHE.popitem(last=False)

    return {
        "bundle": bundle,
        "constraints": constraints,
        "osm": feats,
        "bounds": bounds,
        "resolution_m": resolution_m,
    }


def _build(bounds: tuple, resolution_m: float, dem_source: str, quality: DataQuality) -> dict:
    grid = fetch_dem(bounds, resolution_m, quality, prefer=dem_source)

    feats = fetch_osm(bounds, quality)
    constraints = rasterize_constraints(feats, grid.shape, grid.transform, grid.epsg)

    dem, corrected = _correct_surface_model(
        grid.dem, constraints, grid.source, resolution_m
    )
    if corrected:
        quality.terrain_corrected = True
        quality.note(
            "Buildings and woodland were levelled to their surroundings; the global DEM is a "
            "surface model and would otherwise show roofs and canopy as terrain"
        )

    bundle = he.build_bundle(
        dem, grid.transform, grid.epsg, resolution_m, grid.source,
        water_mask=constraints.get("water"),
    )
    return {
        "bundle": bundle.to_payload(),
        "terrain_corrected": corrected,
        "osm_available": feats.available,
    }


def _correct_surface_model(
    dem: np.ndarray, constraints: dict, source: str, resolution_m: float
) -> tuple:
    """
    Remove buildings and canopy from a digital *surface* model.

    Copernicus GLO-30 includes whatever stands on the ground: about 1.6 m over
    built-up areas and up to 5 m over forest.  Those bumps invent rims and pits
    in exactly the flat village terrain we care about.  Each affected cell is
    replaced by the median of the ring around it, which is the bare ground when
    the patch is small compared with the ring.
    """
    from scipy import ndimage

    if source not in ("copernicus_glo30",):
        # SRTM tiles are already close to bare earth (2000 vintage, integer metres)
        if source == "terrain_tiles":
            return ndimage.gaussian_filter(dem, sigma=1.0).astype(np.float32), False
        return dem, False

    raised = np.zeros(dem.shape, dtype=bool)
    for key in ("building", "forest"):
        mask = constraints.get(key)
        if mask is not None and mask.any():
            raised |= mask
    if not raised.any():
        return dem, False

    # Only correct where the patch is small enough for the ring to be ground
    ring = max(3, int(round(90.0 / resolution_m)) | 1)      # ~90 m, odd size
    background = ndimage.median_filter(dem, size=ring)
    corrected = np.where(raised & (dem > background), background, dem).astype(np.float32)

    changed = float(np.abs(corrected - dem).mean())
    logger.info(
        "Surface-model correction on %.1f%% of cells, mean change %.2f m",
        100 * raised.mean(), changed,
    )
    return corrected, True


def grow_bounds(bounds: tuple, pad_m: float) -> tuple:
    """Expand a WGS84 bbox by a distance in metres (approximate, good enough)."""
    west, south, east, north = bounds
    mid_lat = (south + north) / 2
    dlat = pad_m / 111_320.0
    dlon = pad_m / (111_320.0 * max(0.2, np.cos(np.radians(mid_lat))))
    return (west - dlon, south - dlat, east + dlon, north + dlat)


def cells_for(bounds: tuple, resolution_m: float) -> int:
    """Rough cell count for a bbox, used to enforce the working-grid cap."""
    west, south, east, north = bounds
    mid_lat = (south + north) / 2
    width_m = (east - west) * 111_320.0 * max(0.2, np.cos(np.radians(mid_lat)))
    height_m = (north - south) * 111_320.0
    return int((width_m / resolution_m) * (height_m / resolution_m))


def choose_resolution(bounds: tuple, requested: float, cfg: config.AnalysisConfig) -> float:
    """
    Coarsen automatically rather than blowing the memory budget.

    S_eff = max(requested, sqrt(area / max_cells)) keeps any selection under the
    working-grid cap, and the response reports what was actually used.
    """
    cells = cells_for(bounds, requested)
    if cells <= cfg.max_cells:
        return requested
    factor = np.sqrt(cells / cfg.max_cells)
    coarsened = float(np.ceil(requested * factor / 5.0) * 5.0)
    logger.info(
        "Coarsening from %.0f m to %.0f m: %s cells exceeds the cap of %s",
        requested, coarsened, f"{cells:,}", f"{cfg.max_cells:,}",
    )
    return coarsened
