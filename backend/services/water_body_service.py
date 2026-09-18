"""
Water Body Detection Service.
Queries OpenStreetMap to identify rivers, streams, lakes, and other
water bodies within a given bounding box.

Uses the main OSM API (/api/0.6/map) which is more reliably reachable
than the Overpass API from restricted network environments.

Used to filter out pond candidate sites that fall within existing water features.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np
import requests
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from shapely import prepared

logger = logging.getLogger(__name__)

# Buffer distances around linear water features (in degrees)
# ~55m at equator ≈ 0.0005 degrees
RIVER_BUFFER_DEG = 0.0005
STREAM_BUFFER_DEG = 0.0003

# OSM API endpoints (ordered by preference)
OSM_API_URLS = [
    "https://www.openstreetmap.org/api/0.6/map",
    "https://api.openstreetmap.org/api/0.6/map",
]

# The OSM map API limits bbox to 0.25 degree² area.
# For larger areas we tile the request.
OSM_MAX_BBOX_AREA = 0.20  # stay safely below 0.25
OSM_TIMEOUT = 20


def fetch_water_bodies(
    min_lon: float, max_lon: float,
    min_lat: float, max_lat: float,
    buffer_deg: float = 0.005,
) -> Optional[prepared.PreparedGeometry]:
    """
    Fetch water bodies from OpenStreetMap and return a prepared geometry
    representing all water exclusion zones.

    Uses the main OSM map API (/api/0.6/map) which returns raw OSM XML
    for all features in a bounding box. We parse waterway and water
    features from the response.

    Args:
        min_lon, max_lon, min_lat, max_lat: Bounding box of the contour map
        buffer_deg: Extra padding around the bbox

    Returns:
        PreparedGeometry of the union of all water exclusion zones,
        or None if the query fails or no water bodies are found.
    """
    # Expand bbox slightly
    q_min_lon = min_lon - buffer_deg
    q_max_lon = max_lon + buffer_deg
    q_min_lat = min_lat - buffer_deg
    q_max_lat = max_lat + buffer_deg

    logger.info(
        f"Fetching water bodies from OSM API for bbox "
        f"[{q_min_lon:.4f},{q_min_lat:.4f}]-[{q_max_lon:.4f},{q_max_lat:.4f}]..."
    )

    # Tile the bbox if it's too large for the OSM API limit
    bboxes = _tile_bbox(q_min_lon, q_min_lat, q_max_lon, q_max_lat, OSM_MAX_BBOX_AREA)
    logger.info(f"Split into {len(bboxes)} tile(s)")

    all_nodes = {}
    all_ways = []

    for bbox in bboxes:
        nodes, ways = _fetch_osm_tile(bbox)
        if nodes is not None:
            all_nodes.update(nodes)
            all_ways.extend(ways)

    if not all_ways:
        logger.info("No water features found from OSM API")
        return None

    # Convert ways to geometries
    geometries = _ways_to_geometries(all_ways, all_nodes)

    if not geometries:
        logger.info("No valid water geometries constructed")
        return None

    # Union all water geometries
    try:
        combined = unary_union(geometries)
        prep_geom = prepared.prep(combined)
        logger.info(f"Water exclusion zone created from {len(geometries)} features")
        return prep_geom
    except Exception as e:
        logger.warning(f"Failed to create water exclusion zone: {e}")
        return None


def _tile_bbox(min_lon, min_lat, max_lon, max_lat, max_area):
    """Split a large bbox into tiles that fit within OSM API limits."""
    width = max_lon - min_lon
    height = max_lat - min_lat
    area = width * height

    if area <= max_area:
        return [(min_lon, min_lat, max_lon, max_lat)]

    # Determine how many tiles we need
    n_cols = max(1, int(np.ceil(width / np.sqrt(max_area))))
    n_rows = max(1, int(np.ceil(height / np.sqrt(max_area))))

    tiles = []
    dx = width / n_cols
    dy = height / n_rows

    for i in range(n_cols):
        for j in range(n_rows):
            tiles.append((
                min_lon + i * dx,
                min_lat + j * dy,
                min_lon + (i + 1) * dx,
                min_lat + (j + 1) * dy,
            ))

    return tiles


def _fetch_osm_tile(bbox):
    """
    Fetch OSM data for a single tile bbox using the main OSM API.
    Returns (nodes_dict, ways_list) or (None, []) on failure.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    headers = {"User-Agent": "PondPlanningSystem/1.0"}

    for base_url in OSM_API_URLS:
        url = f"{base_url}?bbox={bbox_str}"
        try:
            logger.info(f"  Fetching tile from {base_url}...")
            response = requests.get(url, headers=headers, timeout=OSM_TIMEOUT)
            response.raise_for_status()
            return _parse_osm_xml(response.content)
        except requests.RequestException as e:
            logger.warning(f"  OSM API request failed ({base_url}): {e}")
        except ET.ParseError as e:
            logger.warning(f"  Failed to parse OSM XML: {e}")

    logger.warning(f"  All OSM API endpoints failed for tile {bbox_str}")
    return None, []


def _parse_osm_xml(xml_content: bytes):
    """
    Parse OSM XML response and extract nodes and water-related ways.

    Returns:
        (nodes_dict, water_ways_list)
        nodes_dict: {node_id: (lon, lat)}
        water_ways_list: [{'id': ..., 'nodes': [...], 'tags': {...}}, ...]
    """
    root = ET.fromstring(xml_content)

    # Build node lookup
    nodes = {}
    for node in root.findall(".//node"):
        nid = node.get("id")
        lat = float(node.get("lat"))
        lon = float(node.get("lon"))
        nodes[nid] = (lon, lat)

    # Find water-related ways
    water_ways = []
    for way in root.findall(".//way"):
        tags = {}
        for tag in way.findall("tag"):
            tags[tag.get("k")] = tag.get("v")

        # Check if this way is water-related
        is_waterway = tags.get("waterway") in (
            "river", "stream", "canal", "drain", "ditch"
        )
        is_water_body = (
            tags.get("natural") == "water"
            or tags.get("water") in ("river", "lake", "reservoir", "pond", "basin")
            or tags.get("landuse") == "reservoir"
        )

        if is_waterway or is_water_body:
            node_refs = [nd.get("ref") for nd in way.findall("nd")]
            water_ways.append({
                "id": way.get("id"),
                "nodes": node_refs,
                "tags": tags,
            })

    logger.info(
        f"  Parsed {len(nodes)} nodes, found {len(water_ways)} water features"
    )
    return nodes, water_ways


def _ways_to_geometries(ways, nodes):
    """Convert OSM ways to Shapely geometries (polygons or buffered lines)."""
    geometries = []

    for way in ways:
        coords = []
        for nid in way["nodes"]:
            if nid in nodes:
                coords.append(nodes[nid])

        if len(coords) < 2:
            continue

        tags = way["tags"]

        # Check if it's a closed polygon (area feature)
        is_closed = len(coords) >= 4 and coords[0] == coords[-1]
        is_area = (
            tags.get("natural") == "water"
            or tags.get("water") is not None
            or tags.get("landuse") == "reservoir"
        )

        if is_closed and is_area:
            try:
                poly = Polygon(coords)
                if poly.is_valid and poly.area > 0:
                    geometries.append(poly)
                    continue
            except Exception:
                pass

        # Linear waterway → buffer into a polygon
        waterway_type = tags.get("waterway", "")
        if waterway_type == "river":
            buffer = RIVER_BUFFER_DEG
        else:
            buffer = STREAM_BUFFER_DEG

        try:
            line = LineString(coords)
            if line.is_valid and line.length > 0:
                geometries.append(line.buffer(buffer))
        except Exception:
            pass

    return geometries


# ─── Candidate Filtering ─────────────────────────────────────────────────────


def is_in_water(
    lon: float, lat: float,
    water_zone: Optional[prepared.PreparedGeometry],
) -> bool:
    """Check if a point falls within a water body exclusion zone."""
    if water_zone is None:
        return False
    point = Point(lon, lat)
    return water_zone.contains(point)


def filter_candidates_by_water(
    candidates: list,
    water_zone: Optional[prepared.PreparedGeometry],
    grid_params: dict,
    to_wgs84,
    resolution_m: float,
) -> list:
    """
    Filter out candidates that fall within water bodies.

    Args:
        candidates: List of candidate dicts with 'row' and 'col' keys
        water_zone: PreparedGeometry from fetch_water_bodies()
        grid_params: Grid parameters
        to_wgs84: pyproj Transformer
        resolution_m: Grid cell size

    Returns:
        Filtered list of candidates (those NOT in water)
    """
    if water_zone is None:
        return candidates

    filtered = []
    removed = 0

    for cand in candidates:
        # Convert grid position to lon/lat
        utm_x = grid_params["origin_x"] + cand["col"] * resolution_m
        utm_y = grid_params["origin_y"] - cand["row"] * resolution_m
        lon, lat = to_wgs84.transform(utm_x, utm_y)

        if is_in_water(lon, lat, water_zone):
            removed += 1
            logger.info(
                f"  Removed candidate at ({lat:.5f}, {lon:.5f}) — inside water body"
            )
        else:
            filtered.append(cand)

    if removed > 0:
        logger.info(
            f"Filtered out {removed} candidates in water bodies, "
            f"{len(filtered)} remaining"
        )
    else:
        logger.info("No candidates were in water bodies")

    return filtered


# ─── GeoJSON Export ──────────────────────────────────────────────────────────


def water_zone_to_geojson(
    water_zone: Optional[prepared.PreparedGeometry],
) -> Optional[dict]:
    """Convert water exclusion zone to GeoJSON for frontend visualization."""
    if water_zone is None:
        return None

    geom = water_zone.context  # unwrap the PreparedGeometry

    features = []
    if geom.geom_type == "MultiPolygon":
        for poly in geom.geoms:
            features.append(_polygon_to_feature(poly))
    elif geom.geom_type == "Polygon":
        features.append(_polygon_to_feature(geom))
    elif geom.geom_type == "GeometryCollection":
        for g in geom.geoms:
            if g.geom_type == "MultiPolygon":
                for poly in g.geoms:
                    features.append(_polygon_to_feature(poly))
            elif g.geom_type == "Polygon":
                features.append(_polygon_to_feature(g))

    if not features:
        return None

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _polygon_to_feature(polygon) -> dict:
    """Convert a Shapely Polygon to a GeoJSON Feature."""
    coords = [list(polygon.exterior.coords)]
    for interior in polygon.interiors:
        coords.append(list(interior.coords))

    return {
        "type": "Feature",
        "properties": {"type": "water_body"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[round(x, 6), round(y, 6)] for x, y in ring]
                for ring in coords
            ],
        },
    }
