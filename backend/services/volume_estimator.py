"""
Volume Estimator Service.
Estimates pond storage volume using cut-fill analysis on the DEM.
For a candidate depression site, computes how much water the natural basin can hold.
"""

import logging

import numpy as np
from scipy import ndimage

from utils.raster_utils import mask_to_polygon_coords

logger = logging.getLogger(__name__)


def estimate_volume(
    dem: np.ndarray,
    fill_depth: np.ndarray,
    candidate: dict,
    grid_params: dict,
    to_wgs84=None,
    resolution_m: float = None,
) -> dict:
    """
    Estimate the storage volume and surface area of a pond at a candidate site.

    Method: Find the connected depression around the candidate point,
    determine the spill elevation (rim), and compute volume as the sum of
    (rim_elevation - cell_elevation) × cell_area for all cells in the basin.

    Args:
        dem: Original DEM array
        fill_depth: fill_depth = filled_DEM - original_DEM
        candidate: Candidate dict with 'row' and 'col' keys
        grid_params: Grid parameters with 'resolution_m'

    Returns:
        dict with keys:
            - volume_m3: Estimated storage volume in cubic meters
            - surface_area_m2: Estimated water surface area in square meters
            - max_depth_m: Maximum depth of the pond
            - rim_elevation_m: Spill elevation
            - mean_depth_m: Average depth across the pond area
            - pond_footprint: GeoJSON Feature polygon of the inundation area
    """
    row, col = candidate["row"], candidate["col"]
    resolution_m = grid_params["resolution_m"]
    cell_area = resolution_m ** 2
    nrows, ncols = dem.shape

    # ── Find the connected depression region ─────────────────────────────

    # Create a binary mask of cells that are part of any depression
    depression_mask = (fill_depth > 0.01).astype(np.int32)

    # Label connected regions
    labeled, num_features = ndimage.label(depression_mask)

    # Determine which depression region the candidate belongs to
    # Search in expanding neighborhood if the exact cell isn't in a depression
    search_radius = max(5, int(50 / resolution_m))
    target_label = 0

    for radius in range(0, search_radius + 1):
        r_lo = max(0, row - radius)
        r_hi = min(nrows, row + radius + 1)
        c_lo = max(0, col - radius)
        c_hi = min(ncols, col + radius + 1)
        local_labels = labeled[r_lo:r_hi, c_lo:c_hi]
        nonzero_labels = local_labels[local_labels > 0]
        if len(nonzero_labels) > 0:
            # Pick the most common label in the neighborhood
            target_label = int(np.bincount(nonzero_labels).argmax())
            break

    if target_label == 0:
        # No depression found — estimate from the local topography
        logger.warning(f"No depression region found near ({row}, {col}), using local estimate")
        result = _local_volume_estimate(dem, row, col, resolution_m, cell_area)
        result["basin_label"] = -1  # unique fallback label
        return result

    # Get cells belonging to this depression
    basin_mask = (labeled == target_label)
    basin_cells = np.where(basin_mask)

    if len(basin_cells[0]) == 0:
        return _zero_volume()

    # ── Compute rim elevation (spill point) ──────────────────────────────

    # The rim elevation is the maximum of fill_depth + dem at the boundary of the basin
    # Equivalently, it's the minimum elevation of cells just outside the basin that
    # are adjacent to basin cells (the spill point)
    # Simpler: rim elevation = filled DEM value at basin cells (should all be the same
    # after filling, since the basin is filled to a flat surface)
    filled_at_basin = dem[basin_mask] + fill_depth[basin_mask]
    rim_elevation = float(filled_at_basin.max())

    # ── Volume calculation ───────────────────────────────────────────────

    # Volume = Σ (rim_elevation - cell_elevation) × cell_area for cells below rim
    depths_at_cells = rim_elevation - dem[basin_mask]
    depths_at_cells = np.maximum(depths_at_cells, 0)  # Clamp negative

    volume_m3 = float(np.sum(depths_at_cells) * cell_area)
    surface_area_m2 = float(np.sum(depths_at_cells > 0) * cell_area)
    max_depth_m = float(depths_at_cells.max()) if len(depths_at_cells) > 0 else 0.0
    mean_depth_m = float(depths_at_cells.mean()) if len(depths_at_cells) > 0 else 0.0

    logger.info(
        f"Volume estimate: {volume_m3:.0f} m³, "
        f"surface={surface_area_m2:.0f} m², "
        f"max_depth={max_depth_m:.1f}m"
    )

    # Generate pond footprint polygon (the actual inundation area)
    pond_footprint = _basin_to_footprint(
        basin_mask, grid_params, to_wgs84, resolution_m
    )

    return {
        "volume_m3": round(volume_m3, 2),
        "surface_area_m2": round(surface_area_m2, 2),
        "max_depth_m": round(max_depth_m, 2),
        "rim_elevation_m": round(rim_elevation, 2),
        "mean_depth_m": round(mean_depth_m, 2),
        "pond_footprint": pond_footprint,
        "basin_label": target_label,
    }


def _local_volume_estimate(
    dem: np.ndarray, row: int, col: int, resolution_m: float, cell_area: float
) -> dict:
    """
    Fallback volume estimate using local topography when no distinct depression is found.
    Creates a hypothetical pond by finding the local minimum and estimating
    a small basin around it.
    """
    nrows, ncols = dem.shape
    radius = max(10, int(100 / resolution_m))

    r_lo = max(0, row - radius)
    r_hi = min(nrows, row + radius + 1)
    c_lo = max(0, col - radius)
    c_hi = min(ncols, col + radius + 1)

    local_dem = dem[r_lo:r_hi, c_lo:c_hi]
    center_elev = dem[row, col]

    # Estimate rim as the mean elevation of the boundary of the local patch
    boundary = np.concatenate([
        local_dem[0, :], local_dem[-1, :],
        local_dem[:, 0], local_dem[:, -1],
    ])
    rim_elevation = float(np.median(boundary))

    if rim_elevation <= center_elev:
        return _zero_volume()

    # Count cells below rim
    below_rim = local_dem < rim_elevation
    depths = rim_elevation - local_dem[below_rim]

    volume_m3 = float(np.sum(depths) * cell_area)
    surface_area_m2 = float(np.sum(below_rim) * cell_area)
    max_depth_m = float(depths.max()) if len(depths) > 0 else 0.0
    mean_depth_m = float(depths.mean()) if len(depths) > 0 else 0.0

    # Generate a rough circular footprint for the fallback case
    pond_footprint = _circular_footprint(
        row, col, grid_params, to_wgs84, resolution_m or grid_params["resolution_m"],
        radius_cells=max(3, int(np.sqrt(np.sum(below_rim)) / 2))
    )

    return {
        "volume_m3": round(volume_m3, 2),
        "surface_area_m2": round(surface_area_m2, 2),
        "max_depth_m": round(max_depth_m, 2),
        "rim_elevation_m": round(rim_elevation, 2),
        "mean_depth_m": round(mean_depth_m, 2),
        "pond_footprint": pond_footprint,
    }


def _zero_volume() -> dict:
    """Return a zero-volume result."""
    return {
        "volume_m3": 0.0,
        "surface_area_m2": 0.0,
        "max_depth_m": 0.0,
        "rim_elevation_m": 0.0,
        "mean_depth_m": 0.0,
        "pond_footprint": None,
    }


def _basin_to_footprint(basin_mask, grid_params, to_wgs84, resolution_m):
    """
    Convert a basin mask to a GeoJSON Feature polygon.
    Uses morphological dilation on tiny basins to ensure visible area.
    """
    if to_wgs84 is None:
        return None

    cell_count = int(basin_mask.sum())
    if cell_count < 2:
        return None

    # For very small basins, dilate the mask slightly so the polygon is visible
    working_mask = basin_mask.copy()
    if cell_count < 20:
        working_mask = ndimage.binary_dilation(working_mask, iterations=2).astype(np.uint8)
    else:
        working_mask = working_mask.astype(np.uint8)

    origin_x = grid_params["origin_x"]
    origin_y = grid_params["origin_y"]

    polygon_coords = mask_to_polygon_coords(
        working_mask, origin_x, origin_y, resolution_m, to_wgs84
    )

    if not polygon_coords or not polygon_coords[0]:
        return None

    return {
        "type": "Feature",
        "properties": {"type": "pond_footprint", "cells": cell_count},
        "geometry": {
            "type": "Polygon",
            "coordinates": polygon_coords,
        },
    }


def _circular_footprint(row, col, grid_params, to_wgs84, resolution_m, radius_cells=5):
    """
    Generate an approximate circular pond footprint for the fallback case.
    """
    if to_wgs84 is None:
        return None

    origin_x = grid_params["origin_x"]
    origin_y = grid_params["origin_y"]
    radius_cells = max(3, radius_cells)

    # Generate circle points
    n_points = 24
    coords = []
    for i in range(n_points + 1):
        angle = 2 * np.pi * i / n_points
        c = col + radius_cells * np.cos(angle)
        r = row + radius_cells * np.sin(angle)
        utm_x = origin_x + c * resolution_m
        utm_y = origin_y - r * resolution_m
        lon, lat = to_wgs84.transform(utm_x, utm_y)
        coords.append([float(lon), float(lat)])

    return {
        "type": "Feature",
        "properties": {"type": "pond_footprint", "cells": 0},
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
    }
