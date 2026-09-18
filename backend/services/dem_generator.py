"""
DEM Generator Service.
Interpolates sparse contour vertices into a continuous elevation raster.

Uses a two-stage approach:
  1. LinearNDInterpolator (Delaunay triangulation) — fast, exact at data points
  2. Fills any NaN gaps (outside convex hull) with nearest-neighbor extrapolation

This is much faster than full RBF for large point sets (>10K vertices)
while still producing high-quality terrain surfaces suitable for hydrology.
"""

import logging

import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

from utils.geo_utils import create_transformers, lonlat_to_utm, compute_grid_params

logger = logging.getLogger(__name__)


def generate_dem(
    vertices: np.ndarray,
    resolution_m: float = 5.0,
    max_vertices: int = 20000,
    kernel: str = "thin_plate_spline",
) -> dict:
    """
    Generate a raster DEM from contour vertices using Delaunay-based interpolation.

    Uses LinearNDInterpolator (fast Delaunay triangulation) as the primary method,
    with NearestNDInterpolator to fill extrapolation gaps outside the convex hull.

    Args:
        vertices: Array of shape (N, 3) → [lon, lat, elevation]
        resolution_m: Output grid cell size in meters
        max_vertices: Maximum number of vertices to use (subsampled if exceeded)
        kernel: Unused (kept for API compatibility); Delaunay is always used

    Returns:
        dict with keys:
            - dem: 2D np.ndarray (float32) — the elevation grid
            - grid_params: dict with origin, size, resolution info
            - to_utm: pyproj Transformer (WGS84 → UTM)
            - to_wgs84: pyproj Transformer (UTM → WGS84)
            - utm_epsg: int — EPSG code of the UTM zone used
    """
    lons = vertices[:, 0]
    lats = vertices[:, 1]
    elevations = vertices[:, 2]

    # Compute center for UTM zone selection
    center_lon = (lons.min() + lons.max()) / 2
    center_lat = (lats.min() + lats.max()) / 2

    # Create CRS transformers
    to_utm, to_wgs84, utm_epsg = create_transformers(center_lon, center_lat)
    logger.info(f"Using UTM EPSG:{utm_epsg} for projection")

    # Project to UTM
    eastings, northings = lonlat_to_utm(lons, lats, to_utm)

    # Subsample if too many vertices (for interpolation performance)
    n_points = len(elevations)
    if n_points > max_vertices:
        logger.info(f"Subsampling {n_points} → {max_vertices} vertices for interpolation")
        # Use stratified sampling: pick evenly across the elevation range
        indices = _stratified_subsample(elevations, max_vertices)
        eastings_sub = eastings[indices]
        northings_sub = northings[indices]
        elevations_sub = elevations[indices]
    else:
        eastings_sub = eastings
        northings_sub = northings
        elevations_sub = elevations

    # Compute grid parameters
    grid_params = compute_grid_params(eastings, northings, resolution_m)
    nrows = grid_params["nrows"]
    ncols = grid_params["ncols"]

    logger.info(f"DEM grid: {nrows} × {ncols} = {nrows * ncols} cells at {resolution_m}m resolution")

    # Build the interpolation points (2D: easting, northing)
    points = np.column_stack([eastings_sub, northings_sub])

    # ── Stage 1: Delaunay-based linear interpolation (fast) ──────────────

    logger.info(f"Fitting LinearNDInterpolator ({len(points)} points)...")
    linear_interp = LinearNDInterpolator(points, elevations_sub)

    # Create evaluation grid
    x_grid, y_grid = np.meshgrid(grid_params["x_coords"], grid_params["y_coords"])
    eval_points = np.column_stack([x_grid.ravel(), y_grid.ravel()])

    logger.info("Evaluating interpolator on grid...")
    dem_flat = linear_interp(eval_points)
    dem = dem_flat.reshape(nrows, ncols).astype(np.float32)

    # ── Stage 2: Fill NaN gaps with nearest-neighbor ─────────────────────

    nan_mask = np.isnan(dem)
    nan_count = nan_mask.sum()
    if nan_count > 0:
        logger.info(f"Filling {nan_count} NaN cells ({100*nan_count/dem.size:.1f}%) with nearest-neighbor")
        nn_interp = NearestNDInterpolator(points, elevations_sub)
        nan_points = eval_points[nan_mask.ravel()]
        dem[nan_mask] = nn_interp(nan_points).astype(np.float32)

    # ── Post-processing ──────────────────────────────────────────────────

    # Light Gaussian smoothing to reduce triangulation artifacts at contour edges
    from scipy.ndimage import gaussian_filter
    dem = gaussian_filter(dem, sigma=0.8).astype(np.float32)

    # Clamp to reasonable range (contour min/max ± some tolerance)
    elev_min, elev_max = elevations.min(), elevations.max()
    elev_range = elev_max - elev_min
    dem = np.clip(dem, elev_min - elev_range * 0.1, elev_max + elev_range * 0.1)

    logger.info(
        f"DEM generated: shape={dem.shape}, "
        f"elevation range [{dem.min():.1f}, {dem.max():.1f}]m"
    )

    return {
        "dem": dem,
        "grid_params": grid_params,
        "to_utm": to_utm,
        "to_wgs84": to_wgs84,
        "utm_epsg": utm_epsg,
    }


def _stratified_subsample(elevations: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Stratified subsampling across elevation bins to preserve representation
    of all elevation levels (especially important for contour data where
    there are many more points at some elevations than others).
    """
    rng = np.random.default_rng(42)

    # Create elevation bins
    n_bins = min(50, len(np.unique(elevations)))
    bin_edges = np.linspace(elevations.min(), elevations.max() + 0.001, n_bins + 1)
    bin_indices = np.digitize(elevations, bin_edges) - 1

    # Allocate samples per bin proportionally, with a minimum per bin
    samples_per_bin = max(1, n_samples // n_bins)
    selected = []

    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = mask.sum()
        if bin_count == 0:
            continue
        bin_idx = np.where(mask)[0]
        n_take = min(bin_count, samples_per_bin)
        chosen = rng.choice(bin_idx, n_take, replace=False)
        selected.extend(chosen)

    # If we still need more samples, pick randomly from remaining
    selected_set = set(selected)
    remaining = [i for i in range(len(elevations)) if i not in selected_set]
    n_remaining = n_samples - len(selected)
    if n_remaining > 0 and remaining:
        extra = rng.choice(remaining, min(n_remaining, len(remaining)), replace=False)
        selected.extend(extra)

    return np.array(selected[:n_samples])
