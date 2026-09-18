"""
Hydrology Service.
Implements the core hydrological analysis pipeline:
  1. Depression detection (Priority-Flood fill)
  2. D8 Flow direction
  3. Flow accumulation
  4. Slope computation
  5. Topographic Wetness Index (TWI)

Optimized for performance using NumPy vectorized operations where possible.
"""

import heapq
import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# D8 flow direction encoding
# Each direction code maps to a (row_offset, col_offset)
D8_CODES = {
    1: (0, 1),     # East
    2: (1, 1),     # Southeast
    4: (1, 0),     # South
    8: (1, -1),    # Southwest
    16: (0, -1),   # West
    32: (-1, -1),  # Northwest
    64: (-1, 0),   # North
    128: (-1, 1),  # Northeast
}

# Neighbor offsets (8-connected) and their corresponding D8 codes
NEIGHBORS = [
    (-1, -1, 32),   # NW
    (-1,  0, 64),   # N
    (-1,  1, 128),  # NE
    ( 0, -1, 16),   # W
    ( 0,  1, 1),    # E
    ( 1, -1, 8),    # SW
    ( 1,  0, 4),    # S
    ( 1,  1, 2),    # SE
]

# Distance factors for D8 (diagonal = sqrt(2), cardinal = 1)
NEIGHBOR_DISTANCES = [
    1.4142,  # NW
    1.0,     # N
    1.4142,  # NE
    1.0,     # W
    1.0,     # E
    1.4142,  # SW
    1.0,     # S
    1.4142,  # SE
]


def run_hydrology_pipeline(dem: np.ndarray, resolution_m: float) -> dict:
    """
    Run the complete hydrological analysis pipeline on a DEM.

    Args:
        dem: 2D elevation array (float32)
        resolution_m: Cell size in meters

    Returns:
        dict with keys:
            - filled_dem: DEM with depressions filled
            - fill_depth: fill_depth = filled - original (depression depth indicator)
            - flow_dir: D8 flow direction grid
            - flow_acc: Flow accumulation grid (number of upstream cells)
            - slope: Slope in radians
            - slope_deg: Slope in degrees
            - twi: Topographic Wetness Index
    """
    nrows, ncols = dem.shape
    logger.info(f"Running hydrology pipeline on {nrows}×{ncols} DEM ({nrows*ncols} cells)")

    # Step 1: Depression filling (Priority-Flood)
    logger.info("Step 1/5: Depression filling...")
    filled_dem = priority_flood_fill(dem)
    fill_depth = filled_dem - dem

    n_depression_cells = np.sum(fill_depth > 0.01)
    logger.info(f"  Found {n_depression_cells} depression cells")

    # Step 2: D8 Flow direction (on filled DEM) - VECTORIZED
    logger.info("Step 2/5: Computing D8 flow directions...")
    flow_dir = compute_flow_direction_vectorized(filled_dem, resolution_m)

    # Step 3: Flow accumulation
    logger.info("Step 3/5: Computing flow accumulation...")
    flow_acc = compute_flow_accumulation(flow_dir, nrows, ncols)

    # Step 4: Slope computation (on original DEM for terrain characterization)
    logger.info("Step 4/5: Computing slope...")
    slope = compute_slope(dem, resolution_m)
    slope_deg = np.degrees(slope)

    # Step 5: TWI
    logger.info("Step 5/5: Computing Topographic Wetness Index...")
    twi = compute_twi(flow_acc, slope, resolution_m)

    logger.info("Hydrology pipeline complete")

    return {
        "filled_dem": filled_dem,
        "fill_depth": fill_depth,
        "flow_dir": flow_dir,
        "flow_acc": flow_acc,
        "slope": slope,
        "slope_deg": slope_deg,
        "twi": twi,
    }


# ─── Step 1: Priority-Flood Depression Filling ───────────────────────────────


def priority_flood_fill(dem: np.ndarray) -> np.ndarray:
    """
    Fill depressions in a DEM using the Priority-Flood algorithm.
    This is an efficient O(N log N) approach.

    The key insight: process boundary cells first (in ascending elevation),
    then flood inward. Any interior cell lower than its already-processed
    neighbor is a depression and gets raised.
    """
    nrows, ncols = dem.shape
    filled = dem.copy().astype(np.float64)
    visited = np.zeros((nrows, ncols), dtype=bool)

    # Priority queue: (elevation, row, col)
    pq = []

    # Initialize with all boundary cells
    for r in range(nrows):
        for c in [0, ncols - 1]:
            heapq.heappush(pq, (float(dem[r, c]), r, c))
            visited[r, c] = True
    for c in range(1, ncols - 1):
        for r in [0, nrows - 1]:
            heapq.heappush(pq, (float(dem[r, c]), r, c))
            visited[r, c] = True

    # Neighbor offsets (8-connected)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # Process cells in elevation order
    while pq:
        elev, r, c = heapq.heappop(pq)

        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc < ncols and not visited[nr, nc]:
                visited[nr, nc] = True
                if filled[nr, nc] < elev:
                    # This cell is in a depression — raise it
                    filled[nr, nc] = elev
                heapq.heappush(pq, (float(filled[nr, nc]), nr, nc))

    return filled.astype(np.float32)


# ─── Step 2: D8 Flow Direction (Vectorized) ──────────────────────────────────


def compute_flow_direction_vectorized(filled_dem: np.ndarray, resolution_m: float) -> np.ndarray:
    """
    Compute D8 flow direction using fully vectorized NumPy operations.
    Much faster than the per-cell loop version.
    """
    nrows, ncols = filled_dem.shape
    dem = filled_dem.astype(np.float64)

    # Direction codes and their (dr, dc) + distances
    directions = [
        (64,  -1,  0, 1.0),      # N
        (128, -1,  1, 1.4142),    # NE
        (1,    0,  1, 1.0),       # E
        (2,    1,  1, 1.4142),    # SE
        (4,    1,  0, 1.0),       # S
        (8,    1, -1, 1.4142),    # SW
        (16,   0, -1, 1.0),       # W
        (32,  -1, -1, 1.4142),    # NW
    ]

    # Compute slopes to all 8 neighbors
    max_slope = np.full((nrows, ncols), -np.inf, dtype=np.float64)
    flow_dir = np.zeros((nrows, ncols), dtype=np.uint8)

    for code, dr, dc, dist_factor in directions:
        # Define source and target slices
        # Source: the cells we're computing flow direction for
        # Target: the neighbor cells
        if dr < 0:
            src_r = slice(1, nrows)
            tgt_r = slice(0, nrows - 1)
        elif dr > 0:
            src_r = slice(0, nrows - 1)
            tgt_r = slice(1, nrows)
        else:
            src_r = slice(0, nrows)
            tgt_r = slice(0, nrows)

        if dc < 0:
            src_c = slice(1, ncols)
            tgt_c = slice(0, ncols - 1)
        elif dc > 0:
            src_c = slice(0, ncols - 1)
            tgt_c = slice(1, ncols)
        else:
            src_c = slice(0, ncols)
            tgt_c = slice(0, ncols)

        # Compute slope = drop / distance
        drop = dem[src_r, src_c] - dem[tgt_r, tgt_c]
        slope = drop / (dist_factor * resolution_m)

        # Update where this slope is steepest
        better = slope > max_slope[src_r, src_c]
        max_slope_view = max_slope[src_r, src_c]
        flow_dir_view = flow_dir[src_r, src_c]

        max_slope_view[better] = slope[better]
        flow_dir_view[better] = code

        max_slope[src_r, src_c] = max_slope_view
        flow_dir[src_r, src_c] = flow_dir_view

    return flow_dir


# ─── Step 3: Flow Accumulation ────────────────────────────────────────────────


def compute_flow_accumulation(flow_dir: np.ndarray, nrows: int, ncols: int) -> np.ndarray:
    """
    Compute flow accumulation using topological sort.
    Each cell receives the sum of all upstream cells that drain into it.
    """
    flow_acc = np.ones((nrows, ncols), dtype=np.float64)  # Each cell counts itself
    in_degree = np.zeros((nrows, ncols), dtype=np.int32)

    # Build in-degree map: for each cell, find where it flows to, increment that target's in-degree
    for code, (dr, dc) in D8_CODES.items():
        # Cells with this flow direction
        mask = (flow_dir == code)
        rows, cols = np.where(mask)
        # Their target cells
        tgt_rows = rows + dr
        tgt_cols = cols + dc
        # Filter in-bounds
        valid = (tgt_rows >= 0) & (tgt_rows < nrows) & (tgt_cols >= 0) & (tgt_cols < ncols)
        tgt_rows = tgt_rows[valid]
        tgt_cols = tgt_cols[valid]
        # Increment in-degree
        np.add.at(in_degree, (tgt_rows, tgt_cols), 1)

    # Initialize queue with cells that have no upstream contributors (ridgelines)
    queue = deque()
    zero_in = np.where(in_degree == 0)
    for r, c in zip(zero_in[0], zero_in[1]):
        queue.append((r, c))

    logger.info(f"  Flow accumulation: {len(queue)} source cells (ridgelines)")

    # Process in topological order
    while queue:
        r, c = queue.popleft()
        code = flow_dir[r, c]
        if code in D8_CODES:
            dr, dc = D8_CODES[code]
            nr, nc = r + dr, c + dc
            if 0 <= nr < nrows and 0 <= nc < ncols:
                flow_acc[nr, nc] += flow_acc[r, c]
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue.append((nr, nc))

    return flow_acc


# ─── Step 4: Slope ───────────────────────────────────────────────────────────


def compute_slope(dem: np.ndarray, resolution_m: float) -> np.ndarray:
    """
    Compute slope in radians using NumPy gradient (central differences).
    """
    dy, dx = np.gradient(dem, resolution_m)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    return slope


# ─── Step 5: Topographic Wetness Index ────────────────────────────────────────


def compute_twi(flow_acc: np.ndarray, slope: np.ndarray, resolution_m: float) -> np.ndarray:
    """
    Compute Topographic Wetness Index: TWI = ln(A / tan(β))

    Where:
        A = contributing area (flow_acc * cell_area)
        β = local slope (radians)

    High TWI = flat areas collecting water from large upstream areas.
    """
    cell_area = resolution_m ** 2
    contributing_area = flow_acc * cell_area

    # Avoid division by zero: clamp tan(slope) to a small positive value
    tan_slope = np.tan(slope)
    tan_slope = np.maximum(tan_slope, 0.001)

    twi = np.log(contributing_area / tan_slope)

    # Clamp extreme values
    twi = np.clip(twi, 0, 30)

    return twi
