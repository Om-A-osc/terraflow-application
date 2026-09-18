"""
Geospatial utility functions.
CRS conversions, coordinate transforms, distance calculations.
"""

import numpy as np
from pyproj import Transformer


def get_utm_zone(lon: float) -> int:
    """Determine UTM zone number from longitude."""
    return int((lon + 180) / 6) + 1


def get_utm_epsg(lon: float, lat: float) -> int:
    """Get the EPSG code for the appropriate UTM zone."""
    zone = get_utm_zone(lon)
    if lat >= 0:
        return 32600 + zone  # Northern hemisphere
    else:
        return 32700 + zone  # Southern hemisphere


def create_transformers(center_lon: float, center_lat: float):
    """
    Create forward (WGS84→UTM) and inverse (UTM→WGS84) transformers
    for the UTM zone containing the given center point.

    Returns:
        tuple: (to_utm, to_wgs84, utm_epsg)
    """
    utm_epsg = get_utm_epsg(center_lon, center_lat)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    to_wgs84 = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    return to_utm, to_wgs84, utm_epsg


def lonlat_to_utm(lons: np.ndarray, lats: np.ndarray, to_utm: Transformer):
    """
    Convert arrays of lon/lat to UTM easting/northing.

    Args:
        lons: Longitude values
        lats: Latitude values
        to_utm: pyproj Transformer (WGS84 → UTM)

    Returns:
        tuple: (eastings, northings) in meters
    """
    eastings, northings = to_utm.transform(lons, lats)
    return np.array(eastings), np.array(northings)


def utm_to_lonlat(eastings: np.ndarray, northings: np.ndarray, to_wgs84: Transformer):
    """
    Convert UTM easting/northing back to lon/lat.

    Args:
        eastings: UTM easting values (meters)
        northings: UTM northing values (meters)
        to_wgs84: pyproj Transformer (UTM → WGS84)

    Returns:
        tuple: (lons, lats)
    """
    lons, lats = to_wgs84.transform(eastings, northings)
    return np.array(lons), np.array(lats)


def compute_grid_params(
    eastings: np.ndarray,
    northings: np.ndarray,
    resolution_m: float,
    padding_cells: int = 5,
):
    """
    Compute grid origin, size, and coordinate arrays for DEM generation.

    Args:
        eastings: UTM X coordinates of all contour vertices
        northings: UTM Y coordinates of all contour vertices
        resolution_m: Grid cell size in meters
        padding_cells: Number of padding cells around the data extent

    Returns:
        dict with keys: origin_x, origin_y, nrows, ncols, x_coords, y_coords
    """
    padding = padding_cells * resolution_m

    x_min = eastings.min() - padding
    x_max = eastings.max() + padding
    y_min = northings.min() - padding
    y_max = northings.max() + padding

    ncols = int(np.ceil((x_max - x_min) / resolution_m))
    nrows = int(np.ceil((y_max - y_min) / resolution_m))

    x_coords = np.linspace(x_min, x_min + ncols * resolution_m, ncols)
    y_coords = np.linspace(y_max, y_max - nrows * resolution_m, nrows)  # Top-down

    return {
        "origin_x": x_min,
        "origin_y": y_max,  # Top-left corner (raster convention)
        "nrows": nrows,
        "ncols": ncols,
        "resolution_m": resolution_m,
        "x_coords": x_coords,
        "y_coords": y_coords,
    }


def haversine_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate the great-circle distance between two points in meters.
    """
    R = 6371000  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
