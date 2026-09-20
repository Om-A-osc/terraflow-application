"""
DEM acquisition.

Primary source is Copernicus GLO-30 as Cloud-Optimized GeoTIFFs on AWS Open
Data: anonymous windowed reads of the 1-degree tiles covering the request, so a
5 km window costs a handful of HTTP range requests rather than a 41 MB download.
Fallback is the Tilezen terrarium PNG pyramid (SRTM 30 m), decoded with OpenCV,
which needs no GDAL at all.

Both paths return the same thing: a float32 elevation grid in a metric UTM CRS,
its affine transform, and the name of the source that produced it.
"""

from __future__ import annotations

import io
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio import windows as rio_windows
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

import config
from core.cache import bump, cached_call, make_key
from core.resilience import DataQuality, SourceUnavailable, call_source, get
from utils.geo_utils import get_utm_epsg

logger = logging.getLogger(__name__)


@dataclass
class DemGrid:
    """An elevation grid in a projected (metric) CRS."""

    dem: np.ndarray                 # float32, (rows, cols), NaN where no data
    transform: object               # affine.Affine of the UTM grid
    epsg: int                       # UTM EPSG code
    resolution_m: float
    source: str                     # 'copernicus_glo30' | 'terrain_tiles' | 'contour_kml'
    bounds_utm: tuple               # (minx, miny, maxx, maxy)

    @property
    def shape(self) -> tuple:
        return self.dem.shape

    @property
    def cell_area_m2(self) -> float:
        return self.resolution_m ** 2

    def to_payload(self) -> dict:
        """Picklable form for the disk cache."""
        return {
            "dem": self.dem,
            "transform": tuple(self.transform)[:6],
            "epsg": self.epsg,
            "resolution_m": self.resolution_m,
            "source": self.source,
            "bounds_utm": self.bounds_utm,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "DemGrid":
        from affine import Affine

        return cls(
            dem=payload["dem"],
            transform=Affine(*payload["transform"]),
            epsg=payload["epsg"],
            resolution_m=payload["resolution_m"],
            source=payload["source"],
            bounds_utm=tuple(payload["bounds_utm"]),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────


def fetch_dem(
    bounds_wgs84: tuple,
    resolution_m: float = 30.0,
    quality: DataQuality | None = None,
    prefer: str = "copernicus",
) -> DemGrid:
    """
    Return an elevation grid covering ``bounds_wgs84`` = (west, south, east, north).

    The result is cached on disk keyed by the rounded bounds, the resolution and
    the algorithm version, so a second analysis of the same village is free.
    """
    quality = quality or DataQuality()
    west, south, east, north = [round(float(v), 4) for v in bounds_wgs84]
    key = make_key("dem", west, south, east, north, resolution_m, prefer)

    cached = cached_call(
        key,
        lambda: _fetch_dem_uncached((west, south, east, north), resolution_m, prefer),
        expire=config.CACHE_TTL_DEM,
    )
    grid = DemGrid.from_payload(cached)
    quality.dem_source = grid.source
    if grid.source == "terrain_tiles":
        quality.note("Elevation from AWS Terrain Tiles (SRTM 30 m); Copernicus was unreachable")
    return grid


def _fetch_dem_uncached(bounds: tuple, resolution_m: float, prefer: str) -> dict:
    order = ["copernicus", "terrain"] if prefer == "copernicus" else ["terrain", "copernicus"]
    errors = []
    for source in order:
        try:
            if source == "copernicus":
                grid = call_source("copernicus_dem", lambda: _read_copernicus(bounds, resolution_m))
            else:
                grid = call_source("terrain_tiles", lambda: _read_terrain_tiles(bounds, resolution_m))
            if grid is not None:
                bump(f"dem_fetch_{'copernicus' if source == 'copernicus' else 'terrain_tiles'}")
                return grid.to_payload()
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source}: {exc}")
            logger.warning("DEM source %s failed: %s", source, exc)
    bump("dem_fetch_failed")
    raise SourceUnavailable(
        "No elevation source is reachable. Tried " + "; ".join(errors or order)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Destination grid
# ─────────────────────────────────────────────────────────────────────────────


def _utm_target(bounds: tuple, resolution_m: float):
    """Build the destination UTM grid for a WGS84 bbox."""
    from pyproj import Transformer

    west, south, east, north = bounds
    epsg = get_utm_epsg((west + east) / 2, (south + north) / 2)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)

    # Transform the whole ring, not just two corners: UTM grid lines are curved
    # with respect to lat/lon, so corner-only conversion clips the edges.
    lons = np.linspace(west, east, 25)
    lats = np.linspace(south, north, 25)
    ring_lon = np.concatenate([lons, np.full(25, east), lons[::-1], np.full(25, west)])
    ring_lat = np.concatenate([np.full(25, south), lats, np.full(25, north), lats[::-1]])
    xs, ys = to_utm.transform(ring_lon, ring_lat)

    minx = math.floor(min(xs) / resolution_m) * resolution_m
    maxx = math.ceil(max(xs) / resolution_m) * resolution_m
    miny = math.floor(min(ys) / resolution_m) * resolution_m
    maxy = math.ceil(max(ys) / resolution_m) * resolution_m

    ncols = max(2, int(round((maxx - minx) / resolution_m)))
    nrows = max(2, int(round((maxy - miny) / resolution_m)))
    transform = from_origin(minx, maxy, resolution_m, resolution_m)
    return epsg, transform, nrows, ncols, (minx, miny, maxx, maxy)


# ─────────────────────────────────────────────────────────────────────────────
# Copernicus GLO-30
# ─────────────────────────────────────────────────────────────────────────────


def copernicus_tile_url(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    key = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    return f"/vsicurl/{config.COPERNICUS_BUCKET}/{key}/{key}.tif"


def _read_copernicus(bounds: tuple, resolution_m: float) -> DemGrid:
    west, south, east, north = bounds
    epsg, transform, nrows, ncols, bounds_utm = _utm_target(bounds, resolution_m)

    dest = np.full((nrows, ncols), np.nan, dtype=np.float32)
    filled_any = False

    for lat in range(math.floor(south), math.floor(north) + 1):
        for lon in range(math.floor(west), math.floor(east) + 1):
            url = copernicus_tile_url(lat, lon)
            try:
                with rasterio.open(url) as src:
                    window = rio_windows.from_bounds(
                        max(west, lon), max(south, lat),
                        min(east, lon + 1), min(north, lat + 1),
                        src.transform,
                    )
                    window = window.round_offsets().round_lengths()
                    # Pad by two pixels so reprojection has neighbours at the seam
                    window = rio_windows.Window(
                        max(0, window.col_off - 2),
                        max(0, window.row_off - 2),
                        min(src.width, window.width + 4),
                        min(src.height, window.height + 4),
                    )
                    if window.width <= 0 or window.height <= 0:
                        continue
                    patch = src.read(1, window=window).astype(np.float32)
                    patch_transform = src.window_transform(window)
                    src_crs = src.crs
                    src_nodata = src.nodata
            except Exception as exc:  # noqa: BLE001 - ocean tiles simply do not exist
                logger.info("Copernicus tile %s unavailable: %s", url.split("/")[-1], exc)
                continue

            if src_nodata is not None:
                patch = np.where(patch == src_nodata, np.nan, patch)

            tmp = np.full((nrows, ncols), np.nan, dtype=np.float32)
            reproject(
                source=patch,
                destination=tmp,
                src_transform=patch_transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=f"EPSG:{epsg}",
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan,
            )
            mask = ~np.isnan(tmp)
            dest[mask] = tmp[mask]
            filled_any = filled_any or bool(mask.any())

    if not filled_any:
        raise SourceUnavailable("No Copernicus tile covered the requested area")

    dem = _fill_small_gaps(dest)
    return DemGrid(dem, transform, epsg, resolution_m, "copernicus_glo30", bounds_utm)


# ─────────────────────────────────────────────────────────────────────────────
# Terrain tiles (Tilezen terrarium PNG, SRTM in India)
# ─────────────────────────────────────────────────────────────────────────────


def _deg2tile(lon: float, lat: float, z: int) -> tuple:
    lat_rad = math.radians(lat)
    n = 2.0 ** z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile2merc(x: int, y: int, z: int) -> tuple:
    """Web-Mercator bounds (EPSG:3857) of a tile."""
    world = 20037508.342789244
    size = 2 * world / (2 ** z)
    minx = -world + x * size
    maxy = world - y * size
    return minx, maxy - size, minx + size, maxy


def _read_terrain_tiles(bounds: tuple, resolution_m: float) -> DemGrid:
    import cv2

    west, south, east, north = bounds
    z = config.TERRAIN_TILE_ZOOM
    x0, y0 = _deg2tile(west, north, z)
    x1, y1 = _deg2tile(east, south, z)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
    if n_tiles > 64:
        raise SourceUnavailable(f"Area needs {n_tiles} terrain tiles, too many")

    def fetch(xy: tuple):
        x, y = xy
        url = config.TERRAIN_TILES_URL.format(z=z, x=x, y=y)
        resp = get(url, timeout=(3.05, 15.0))
        buf = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)   # OpenCV returns BGR
        if img is None:
            raise SourceUnavailable(f"Could not decode tile {z}/{x}/{y}")
        b = img[:, :, 0].astype(np.float32)
        g = img[:, :, 1].astype(np.float32)
        r = img[:, :, 2].astype(np.float32)
        return (x, y), (r * 256.0 + g + b / 256.0) - 32768.0

    coords = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        tiles = dict(pool.map(fetch, coords))

    tile_px = next(iter(tiles.values())).shape[0]
    mosaic = np.full(((y1 - y0 + 1) * tile_px, (x1 - x0 + 1) * tile_px), np.nan, dtype=np.float32)
    for (x, y), arr in tiles.items():
        r0 = (y - y0) * tile_px
        c0 = (x - x0) * tile_px
        mosaic[r0:r0 + tile_px, c0:c0 + tile_px] = arr

    merc_minx, _, _, merc_maxy = _tile2merc(x0, y0, z)
    _, merc_miny, merc_maxx, _ = _tile2merc(x1, y1, z)
    px = (merc_maxx - merc_minx) / mosaic.shape[1]
    py = (merc_maxy - merc_miny) / mosaic.shape[0]
    src_transform = from_origin(merc_minx, merc_maxy, px, py)

    epsg, transform, nrows, ncols, bounds_utm = _utm_target(bounds, resolution_m)
    dest = np.full((nrows, ncols), np.nan, dtype=np.float32)
    reproject(
        source=mosaic,
        destination=dest,
        src_transform=src_transform,
        src_crs="EPSG:3857",
        dst_transform=transform,
        dst_crs=f"EPSG:{epsg}",
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    dem = _fill_small_gaps(dest)
    return DemGrid(dem, transform, epsg, resolution_m, "terrain_tiles", bounds_utm)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fill_small_gaps(dem: np.ndarray) -> np.ndarray:
    """
    Fill NaN holes by nearest-neighbour so the hydrology never sees a gap.

    Holes at this stage come from tile seams and missing ocean tiles, both small
    relative to the window.
    """
    mask = np.isnan(dem)
    if not mask.any():
        return dem.astype(np.float32)
    if mask.all():
        raise SourceUnavailable("Elevation grid is empty")

    from scipy import ndimage

    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = dem[tuple(idx)]
    n = int(mask.sum())
    if n:
        logger.info("Filled %d empty DEM cells (%.2f%%) by nearest neighbour", n, 100 * n / dem.size)
    return filled.astype(np.float32)
