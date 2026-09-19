"""
Raster to GeoJSON conversion for the map overlays.

The phase-2 code vectorised a mask with OpenCV and kept only the largest outer
ring, which silently dropped islands and holes.  Catchments regularly have both,
so this uses ``rasterio.features.shapes`` and returns a proper (Multi)Polygon
reprojected to WGS84.
"""

from __future__ import annotations

import logging

import numpy as np
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

logger = logging.getLogger(__name__)


def mask_to_geometry(
    mask: np.ndarray,
    transform,
    epsg: int,
    simplify_m: float = 5.0,
    min_area_m2: float = 0.0,
):
    """Vectorise a boolean raster mask into one shapely geometry in WGS84."""
    from pyproj import Transformer
    from rasterio.features import shapes as rio_shapes

    if mask is None or not np.any(mask):
        return None

    data = mask.astype(np.uint8)
    polys = []
    for geom, value in rio_shapes(data, mask=data.astype(bool), transform=transform):
        if value != 1:
            continue
        poly = shape(geom)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < min_area_m2:
            continue
        polys.append(poly)

    if not polys:
        return None

    merged = unary_union(polys)
    if simplify_m > 0:
        merged = merged.simplify(simplify_m, preserve_topology=True)
    if merged.is_empty:
        return None

    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    return shapely_transform(lambda x, y, z=None: to_wgs.transform(x, y), merged)


def mask_to_feature(
    mask: np.ndarray,
    transform,
    epsg: int,
    properties: dict | None = None,
    simplify_m: float = 5.0,
    min_area_m2: float = 0.0,
) -> dict | None:
    """Vectorise a mask into a GeoJSON Feature, or None when it is empty."""
    geom = mask_to_geometry(mask, transform, epsg, simplify_m, min_area_m2)
    if geom is None:
        return None
    return {
        "type": "Feature",
        "properties": properties or {},
        "geometry": mapping(geom),
    }


def lines_to_featurecollection(
    mask: np.ndarray,
    values: np.ndarray,
    transform,
    epsg: int,
    max_features: int = 600,
) -> dict:
    """
    Turn the stream raster into a FeatureCollection, one feature per Strahler
    order, so the map can style higher-order channels more heavily.
    """
    features = []
    if mask is None or not np.any(mask):
        return {"type": "FeatureCollection", "features": []}

    orders = np.unique(values[mask])
    orders = [int(o) for o in orders if o > 0] or [1]
    for order in sorted(orders, reverse=True)[:8]:
        sel = mask & (values == order) if (values > 0).any() else mask
        geom = mask_to_geometry(sel, transform, epsg, simplify_m=10.0)
        if geom is None:
            continue
        features.append({
            "type": "Feature",
            "properties": {"strahler_order": order},
            "geometry": mapping(geom),
        })
        if len(features) >= max_features:
            break
    return {"type": "FeatureCollection", "features": features}


def rasterize_polygon(
    geojson_geom: dict, shape_hw: tuple, transform, epsg: int, all_touched: bool = False
) -> np.ndarray:
    """
    Burn a WGS84 GeoJSON polygon onto the UTM analysis grid.

    ``all_touched`` defaults to False so a cell counts only when its centre
    falls inside the polygon.  With it on, any cell the boundary grazes is
    included, and since a site is reported at its cell centre a suggestion
    could appear up to half a cell outside the area the user drew.

    A polygon smaller than a cell would rasterise to nothing under the strict
    rule, so that case falls back to the permissive one.
    """
    from pyproj import Transformer
    from rasterio.features import rasterize

    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    geom = shape(geojson_geom)
    geom_utm = shapely_transform(lambda x, y, z=None: to_utm.transform(x, y), geom)

    mask = rasterize(
        [(mapping(geom_utm), 1)],
        out_shape=shape_hw, transform=transform, fill=0,
        dtype="uint8", all_touched=all_touched,
    ).astype(bool)

    if not mask.any() and not all_touched:
        mask = rasterize(
            [(mapping(geom_utm), 1)],
            out_shape=shape_hw, transform=transform, fill=0,
            dtype="uint8", all_touched=True,
        ).astype(bool)
    return mask


def pixel_to_lonlat(row: int, col: int, transform, epsg: int) -> tuple:
    """Centre of a cell in WGS84."""
    from pyproj import Transformer

    x, y = transform * (col + 0.5, row + 0.5)
    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = to_wgs.transform(x, y)
    return float(lon), float(lat)


def polygon_area_m2(geojson_geom: dict) -> float:
    """Area of a WGS84 polygon in square metres, via an equal-area projection."""
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    geom = shape(geojson_geom)
    if geom.is_empty:
        return 0.0
    area, _ = geod.geometry_area_perimeter(geom)
    return abs(area)


def geometry_to_linestrings(mask: np.ndarray, transform, epsg: int, simplify_m: float = 10.0) -> list:
    """Boundary of a mask as WGS84 line coordinates (used for the bund overlay)."""
    geom = mask_to_geometry(mask, transform, epsg, simplify_m)
    if geom is None:
        return []
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    out = []
    for poly in polys:
        try:
            out.append([[float(x), float(y)] for x, y in poly.exterior.coords])
        except AttributeError:
            continue
    return out


def stream_features_to_geojson(
    features: list, epsg: int, max_features: int = 500, clip_to=None
) -> dict:
    """
    Reproject pyflwdir stream segments to WGS84 and trim them to an area.

    Order matters here.  The cap is applied *after* clipping and after sorting
    by Strahler order, so the channels that survive it are the ones that carry
    the most water inside the area asked about.  Capping first threw away every
    segment in the selection whenever the routing window held more segments
    than the cap, and the map then showed no drainage at all.
    """
    from pyproj import Transformer
    from shapely.geometry import LineString, MultiLineString, mapping

    if not features:
        return {"type": "FeatureCollection", "features": []}

    to_wgs = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    kept = []

    for feature in features:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not geometry or geometry.get("type") != "LineString":
            continue
        coords = geometry.get("coordinates") or []
        if len(coords) < 2:
            continue
        try:
            line = LineString(coords).simplify(10.0, preserve_topology=False)
            xs, ys = zip(*list(line.coords))
            lons, lats = to_wgs.transform(xs, ys)
            geom = LineString(zip(lons, lats))

            if clip_to is not None:
                geom = geom.intersection(clip_to)
                if geom.is_empty:
                    continue
                if not isinstance(geom, (LineString, MultiLineString)):
                    continue

            props = feature.get("properties", {}) or {}
            order = int(props.get("strord", props.get("strahler_order", 1)) or 1)
            kept.append((order, geom.length, geom))
        except Exception:  # noqa: BLE001
            continue

    # Biggest channels first, so the cap drops the least important trickles
    kept.sort(key=lambda item: (-item[0], -item[1]))

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"strahler_order": order},
                "geometry": mapping(geom),
            }
            for order, _length, geom in kept[:max_features]
        ],
    }
