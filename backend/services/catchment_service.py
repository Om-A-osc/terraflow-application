"""
Catchment Service.
Delineates the catchment (watershed) area for a given pour point
by tracing upstream through the D8 flow direction grid.
"""

import logging
from collections import deque

import numpy as np

from services.hydrology import D8_CODES
from utils.raster_utils import mask_to_polygon_coords

logger = logging.getLogger(__name__)

# Reverse D8 lookup: for each direction code, which direction points TO it?
# If cell (nr, nc) has flow_dir pointing to (r, c), then (nr, nc) is upstream of (r, c).
REVERSE_D8 = {}
for code, (dr, dc) in D8_CODES.items():
    # The reverse: a cell at (r-dr, r-dc) flowing in direction `code` flows INTO (r, c)
    REVERSE_D8[code] = (-dr, -dc)


def delineate_catchment(
    flow_dir: np.ndarray,
    pour_row: int,
    pour_col: int,
    grid_params: dict,
    to_wgs84,
    resolution_m: float,
) -> dict:
    """
    Delineate the catchment area upstream of a pour point.

    Algorithm: BFS upstream — find all cells whose flow direction ultimately
    leads to the pour point.

    Args:
        flow_dir: D8 flow direction grid
        pour_row, pour_col: Grid position of the pour point
        grid_params: Grid parameters (origin, resolution)
        to_wgs84: pyproj Transformer (UTM → WGS84)
        resolution_m: Cell size in meters

    Returns:
        dict with keys:
            - area_km2: Catchment area in square kilometers
            - area_m2: Catchment area in square meters
            - cell_count: Number of cells in the catchment
            - mask: Binary 2D array (1 = in catchment)
            - polygon: GeoJSON Feature dict
    """
    nrows, ncols = flow_dir.shape

    # BFS upstream from pour point
    mask = np.zeros((nrows, ncols), dtype=np.uint8)
    queue = deque([(pour_row, pour_col)])
    mask[pour_row, pour_col] = 1

    while queue:
        r, c = queue.popleft()

        # Check all 8 neighbors — if a neighbor flows INTO (r, c), it's upstream
        for code, (dr, dc) in D8_CODES.items():
            # The neighbor at (r - dr, r - dc) would flow INTO (r, c) if it has direction `code`
            nr, nc = r - dr, c - dc
            if 0 <= nr < nrows and 0 <= nc < ncols:
                if not mask[nr, nc] and flow_dir[nr, nc] == code:
                    mask[nr, nc] = 1
                    queue.append((nr, nc))

    # Compute area
    cell_count = int(mask.sum())
    cell_area_m2 = resolution_m ** 2
    area_m2 = cell_count * cell_area_m2
    area_km2 = area_m2 / 1_000_000

    # Vectorize mask to polygon
    origin_x = grid_params["origin_x"]
    origin_y = grid_params["origin_y"]

    polygon_coords = mask_to_polygon_coords(
        mask, origin_x, origin_y, resolution_m, to_wgs84
    )

    polygon_feature = {
        "type": "Feature",
        "properties": {
            "area_km2": round(area_km2, 4),
            "cell_count": cell_count,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": polygon_coords if polygon_coords else [[]],
        },
    }

    logger.info(f"Catchment: {cell_count} cells, {area_km2:.4f} km²")

    return {
        "area_km2": round(area_km2, 4),
        "area_m2": round(area_m2, 2),
        "cell_count": cell_count,
        "mask": mask,
        "polygon": polygon_feature,
    }
