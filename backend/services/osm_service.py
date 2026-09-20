"""
OpenStreetMap features: water bodies, land use, buildings, roads and railways.

Fetched from Overpass (backend only, never from the browser) with mirrors and a
thirty-day disk cache, because the public instance asks regular applications to
stay near a hundred queries a day.  If every Overpass endpoint fails, the older
``water_body_service`` path against the main OSM API is tried, and if that fails
too the analysis continues with no constraints and says so in data quality.

Geometries come back in WGS84 and are rasterised onto the UTM analysis grid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from shapely.geometry import LineString, Polygon, mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

import config
from core.cache import bump, cached_call, make_key
from core.resilience import DataQuality, call_source, http

logger = logging.getLogger(__name__)

# Which OSM tags map to which of our land-use classes (config.CN_TABLE_AMC2)
LANDUSE_MAP = {
    "forest": "forest", "wood": "forest",
    "farmland": "cropland", "farmyard": "cropland", "orchard": "orchard",
    "vineyard": "orchard", "plantation": "orchard",
    "meadow": "grass", "grass": "grass", "grassland": "grass",
    "scrub": "scrub", "heath": "scrub", "shrubbery": "scrub",
    "residential": "builtup", "industrial": "builtup", "commercial": "builtup",
    "retail": "builtup", "construction": "builtup", "quarry": "wasteland",
    "brownfield": "wasteland", "greenfield": "fallow", "village_green": "grass",
    "recreation_ground": "grass", "cemetery": "grass",
    "bare_rock": "wasteland", "sand": "wasteland", "wetland": "water",
}

OVERPASS_QUERY = """[out:json][timeout:25];
(
  way["natural"="water"]({bbox});
  way["water"]({bbox});
  way["landuse"="reservoir"]({bbox});
  way["waterway"~"^(river|stream|canal|drain|ditch)$"]({bbox});
  way["landuse"]({bbox});
  way["natural"~"^(wood|scrub|heath|grassland|bare_rock|sand|wetland)$"]({bbox});
  way["building"]({bbox});
  way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential)$"]({bbox});
  way["railway"]({bbox});
  relation["natural"="water"]({bbox});
  relation["landuse"="reservoir"]({bbox});
);
out geom;
"""


@dataclass
class OsmFeatures:
    """Feature geometries in WGS84, grouped by what they constrain."""

    water: list = field(default_factory=list)        # polygons + buffered waterways
    buildings: list = field(default_factory=list)
    roads: list = field(default_factory=list)
    railways: list = field(default_factory=list)
    landuse: list = field(default_factory=list)      # (geometry, class) pairs
    available: bool = False
    source: str = "none"

    def water_union(self):
        return unary_union(self.water) if self.water else None


# ─────────────────────────────────────────────────────────────────────────────
# Fetching
# ─────────────────────────────────────────────────────────────────────────────


def fetch_osm(bounds_wgs84: tuple, quality: DataQuality | None = None) -> OsmFeatures:
    """Fetch and cache OSM features for a bbox (west, south, east, north)."""
    quality = quality or DataQuality()
    west, south, east, north = [round(float(v), 2) for v in bounds_wgs84]
    # Round outward to a 0.01 degree grid so nearby requests share one entry
    key = make_key("osm", west, south, east, north)

    payload = cached_call(
        key,
        lambda: _fetch_osm_uncached((west, south, east, north)),
        expire=config.CACHE_TTL_OSM,
    )
    feats = _payload_to_features(payload)
    quality.osm_available = feats.available
    if not feats.available:
        quality.note("OpenStreetMap features unavailable; siting ran without constraint masks")
    return feats


# OpenStreetMap features are optional: without them the analysis runs and says
# so.  They are therefore held to a hard time budget.  Three mirrors at a
# twenty-second read timeout each could burn a minute before the real work
# started, which is how a large area ran out of time entirely.
OSM_ENDPOINT_TIMEOUT_S = 12.0
OSM_TOTAL_BUDGET_S = 25.0


def _fetch_osm_uncached(bounds: tuple) -> dict:
    import time

    west, south, east, north = bounds
    bbox = f"{south},{west},{north},{east}"
    query = OVERPASS_QUERY.format(bbox=bbox)
    deadline = time.monotonic() + OSM_TOTAL_BUDGET_S

    for endpoint in config.OVERPASS_ENDPOINTS:
        remaining = deadline - time.monotonic()
        if remaining < 4.0:
            logger.warning("Out of time for OpenStreetMap; continuing without it")
            break

        read_timeout = min(OSM_ENDPOINT_TIMEOUT_S, remaining)

        def _call(url=endpoint, timeout=read_timeout):
            resp = http().post(url, data={"data": query}, timeout=(3.05, timeout))
            resp.raise_for_status()
            return resp.json()

        data = call_source(f"overpass:{endpoint.split('/')[2]}", _call)
        if data and "elements" in data:
            bump("osm_fetch")
            logger.info(
                "Overpass %s returned %d elements", endpoint.split("/")[2], len(data["elements"])
            )
            return _parse_overpass(data)

    if time.monotonic() < deadline:
        logger.warning("All Overpass endpoints failed; trying the OSM map API")
        legacy = _fetch_legacy_osm(bounds)
        if legacy is not None:
            return legacy

    bump("osm_failed")
    return {"available": False, "source": "none", "features": []}


def _fetch_legacy_osm(bounds: tuple) -> dict | None:
    """Last resort: the water-body path that the phase-2 prototype used."""
    try:
        from services.water_body_service import fetch_water_bodies

        west, south, east, north = bounds
        zone = fetch_water_bodies(west, east, south, north)
        if zone is None:
            return None
        geom = zone.context
        polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        return {
            "available": True,
            "source": "osm_map_api",
            "features": [{"kind": "water", "geom": mapping(p)} for p in polys],
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Legacy OSM water fetch failed: %s", exc)
        return None


def _parse_overpass(data: dict) -> dict:
    """Turn an Overpass ``out geom`` response into classified GeoJSON pieces."""
    features = []
    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        coords = _element_coords(el)
        if len(coords) < 2:
            continue

        closed = len(coords) >= 4 and coords[0] == coords[-1]
        kind, cls = _classify(tags)
        if kind is None:
            continue

        try:
            if closed and kind in ("water", "building", "landuse"):
                geom = Polygon(coords)
                if not geom.is_valid:
                    geom = geom.buffer(0)
            else:
                geom = LineString(coords)
            if geom.is_empty:
                continue
        except Exception:  # noqa: BLE001
            continue

        features.append({"kind": kind, "class": cls, "geom": mapping(geom)})

    return {"available": True, "source": "overpass", "features": features}


def _element_coords(el: dict) -> list:
    if "geometry" in el and el["geometry"]:
        return [(p["lon"], p["lat"]) for p in el["geometry"] if "lon" in p and "lat" in p]
    if el.get("type") == "relation":
        pts = []
        for member in el.get("members", []):
            for p in member.get("geometry", []) or []:
                if "lon" in p and "lat" in p:
                    pts.append((p["lon"], p["lat"]))
        return pts
    return []


def _classify(tags: dict) -> tuple:
    """Map OSM tags onto (constraint kind, land-use class)."""
    if tags.get("natural") == "water" or tags.get("water") or tags.get("landuse") == "reservoir":
        return "water", "water"
    if tags.get("waterway") in ("river", "stream", "canal", "drain", "ditch"):
        return "waterway", tags.get("waterway")
    if tags.get("building"):
        return "building", "builtup"
    if tags.get("highway"):
        return "road", None
    if tags.get("railway"):
        return "railway", None
    for key in ("landuse", "natural"):
        value = tags.get(key)
        if value and value in LANDUSE_MAP:
            return "landuse", LANDUSE_MAP[value]
    return None, None


def _payload_to_features(payload: dict) -> OsmFeatures:
    feats = OsmFeatures(available=payload.get("available", False), source=payload.get("source", "none"))
    for item in payload.get("features", []):
        try:
            geom = shape(item["geom"])
        except Exception:  # noqa: BLE001
            continue
        kind = item.get("kind")
        if kind == "water":
            feats.water.append(geom)
        elif kind == "waterway":
            # Buffer linear water in degrees: ~0.0005 deg is about 55 m
            width = 0.0005 if item.get("class") == "river" else 0.0002
            feats.water.append(geom.buffer(width))
        elif kind == "building":
            feats.buildings.append(geom)
        elif kind == "road":
            feats.roads.append(geom)
        elif kind == "railway":
            feats.railways.append(geom)
        elif kind == "landuse":
            feats.landuse.append((geom, item.get("class") or "unknown"))
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Rasterisation onto the analysis grid
# ─────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=8)
def _transformer(epsg: int):
    """Building a Transformer costs milliseconds; there may be hundreds of geometries."""
    from pyproj import Transformer

    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def _to_utm(geom, epsg: int):
    tr = _transformer(epsg)
    return shapely_transform(lambda x, y, z=None: tr.transform(x, y), geom)


def rasterize_constraints(
    feats: OsmFeatures,
    shape_hw: tuple,
    transform,
    epsg: int,
    cfg: config.AnalysisConfig | None = None,
) -> dict:
    """
    Build the Boolean constraint masks and the land-use class grid.

    Returns masks that are True where a pond may NOT be built, plus a land-use
    suitability grid in 0..1 and a water mask used to blank existing tanks.
    """
    from rasterio.features import rasterize

    cfg = cfg or config.AnalysisConfig()
    rows, cols = shape_hw
    empty = np.zeros((rows, cols), dtype=bool)
    out = {
        "water": empty.copy(),
        "water_buffered": empty.copy(),
        "building": empty.copy(),
        "road": empty.copy(),
        "railway": empty.copy(),
        "forest": empty.copy(),
        "blocked": empty.copy(),
        "landuse_score": np.full((rows, cols), config.LANDUSE_SUITABILITY["unknown"], dtype=np.float32),
        "landuse_class": np.full((rows, cols), "unknown", dtype=object),
    }
    if not feats.available:
        return out

    def burn(geoms, buffer_m=0.0):
        if not geoms:
            return empty.copy()
        shapes = []
        for g in geoms:
            try:
                gu = _to_utm(g, epsg)
                if buffer_m:
                    gu = gu.buffer(buffer_m)
                if not gu.is_empty:
                    shapes.append((mapping(gu), 1))
            except Exception:  # noqa: BLE001
                continue
        if not shapes:
            return empty.copy()
        return rasterize(
            shapes, out_shape=(rows, cols), transform=transform, fill=0, dtype="uint8"
        ).astype(bool)

    out["water"] = burn(feats.water)
    out["water_buffered"] = burn(feats.water, cfg.buffer_water_m)
    out["building"] = burn(feats.buildings, cfg.buffer_building_m)
    out["road"] = burn(feats.roads, cfg.buffer_road_m)
    out["railway"] = burn(feats.railways, cfg.buffer_railway_m)

    # Land-use classes: burn each class, last write wins for overlaps
    forest = empty.copy()
    for geom, cls in feats.landuse:
        mask = burn([geom])
        if not mask.any():
            continue
        out["landuse_score"][mask] = config.LANDUSE_SUITABILITY.get(cls, 0.6)
        out["landuse_class"][mask] = cls
        if cls == "forest":
            forest |= mask
    out["forest"] = forest

    out["blocked"] = (
        out["water_buffered"] | out["building"] | out["road"] | out["railway"] | out["forest"]
    )
    logger.info(
        "Constraints: %.1f%% of the grid blocked (water %.1f%%, built %.1f%%, forest %.1f%%)",
        100 * out["blocked"].mean(),
        100 * out["water_buffered"].mean(),
        100 * out["building"].mean(),
        100 * out["forest"].mean(),
    )
    return out


def constraints_geojson(feats: OsmFeatures, limit: int = 400) -> dict:
    """A light FeatureCollection of the constraints, for the map overlay."""
    features = []

    def add(geoms, kind, cap):
        for g in list(geoms)[:cap]:
            try:
                features.append({
                    "type": "Feature",
                    "properties": {"kind": kind},
                    "geometry": mapping(g.simplify(0.00005, preserve_topology=True)),
                })
            except Exception:  # noqa: BLE001
                continue

    add(feats.water, "water", limit)
    add(feats.buildings, "building", limit // 2)
    add([g for g, _ in feats.landuse if _ == "forest"], "forest", limit // 4)
    return {"type": "FeatureCollection", "features": features}
