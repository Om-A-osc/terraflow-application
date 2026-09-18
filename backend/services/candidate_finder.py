"""
Candidate Finder Service.
Identifies optimal pond locations using composite scoring of
terrain characteristics: depression depth, TWI, slope, and flow accumulation.

Uses non-maximum suppression to extract distinct, well-spaced candidate sites.
"""

import logging

import numpy as np
from scipy import ndimage

from config import AnalysisConfig
from utils.raster_utils import normalize_array

logger = logging.getLogger(__name__)


def find_candidates(
    hydro_results: dict,
    dem: np.ndarray,
    grid_params: dict,
    config: AnalysisConfig = None,
) -> list:
    """
    Find optimal pond candidate sites using composite terrain scoring.

    Args:
        hydro_results: Output from hydrology.run_hydrology_pipeline()
        dem: Original DEM array
        grid_params: Grid parameters (origin, resolution, etc.)
        config: Analysis configuration

    Returns:
        List of candidate dicts sorted by score (best first), each containing:
            - row, col: Grid position
            - score: Composite suitability score (0–1)
            - elevation: Elevation at the site
            - depression_depth: Maximum fill depth in the local depression
            - twi: Topographic Wetness Index
            - slope_deg: Local slope in degrees
            - flow_acc: Flow accumulation value
    """
    if config is None:
        config = AnalysisConfig()

    fill_depth = hydro_results["fill_depth"]
    twi = hydro_results["twi"]
    slope_deg = hydro_results["slope_deg"]
    flow_acc = hydro_results["flow_acc"]
    resolution_m = grid_params["resolution_m"]

    nrows, ncols = dem.shape

    # ── Compute composite suitability score ──────────────────────────────

    weights = config.score_weights

    # Normalize each layer to [0, 1]
    norm_depression = normalize_array(fill_depth)
    norm_twi = normalize_array(twi)
    norm_slope_inv = normalize_array(-slope_deg)  # Lower slope is better
    norm_acc = normalize_array(np.log1p(flow_acc))  # Log-scale accumulation

    # Weighted sum
    score = (
        weights["depression"] * norm_depression
        + weights["twi"] * norm_twi
        + weights["slope"] * norm_slope_inv
        + weights["accumulation"] * norm_acc
    )

    logger.info(
        f"Score stats: min={score.min():.3f}, max={score.max():.3f}, "
        f"mean={score.mean():.3f}"
    )

    # ── Non-maximum suppression ──────────────────────────────────────────

    # Minimum spacing in pixels
    spacing_px = max(3, int(config.candidate_min_spacing_m / resolution_m))

    # Apply minimum score threshold (top 5% of scores)
    score_threshold = np.percentile(score, 95)
    score_filtered = score.copy()
    score_filtered[score < score_threshold] = 0

    # Also require minimum depression depth
    score_filtered[fill_depth < config.min_depression_depth_m] *= 0.3

    # Label connected high-score regions
    candidates = _non_maximum_suppression(
        score_filtered, spacing_px, config.num_candidates * 3  # Get extras for filtering
    )

    # ── Build candidate details ──────────────────────────────────────────

    result = []
    for r, c, s in candidates:
        # Find the deepest point in the local depression neighborhood
        local_region = fill_depth[
            max(0, r - spacing_px) : min(nrows, r + spacing_px),
            max(0, c - spacing_px) : min(ncols, c + spacing_px),
        ]
        max_depth = float(local_region.max())

        result.append({
            "row": int(r),
            "col": int(c),
            "score": float(s),
            "elevation": float(dem[r, c]),
            "depression_depth": max_depth,
            "twi": float(twi[r, c]),
            "slope_deg": float(slope_deg[r, c]),
            "flow_acc": float(flow_acc[r, c]),
        })

    # Sort by score (descending) and take top N
    result.sort(key=lambda x: x["score"], reverse=True)
    result = result[: config.num_candidates]

    logger.info(f"Selected {len(result)} pond candidates")
    return result


def _non_maximum_suppression(
    score: np.ndarray, spacing_px: int, max_candidates: int
) -> list:
    """
    Extract local maxima from the score grid with minimum spacing.
    Uses the greedy approach: pick the highest score, suppress nearby, repeat.

    Args:
        score: 2D suitability score array
        spacing_px: Minimum spacing between candidates in pixels
        max_candidates: Maximum number of candidates to extract

    Returns:
        List of (row, col, score) tuples
    """
    # Find all local maxima using maximum filter
    from scipy.ndimage import maximum_filter

    # Detect local maxima
    local_max = maximum_filter(score, size=spacing_px)
    detected = (score == local_max) & (score > 0)

    # Get coordinates and scores of all local maxima
    max_rows, max_cols = np.where(detected)
    max_scores = score[max_rows, max_cols]

    # Sort by score (descending)
    order = np.argsort(-max_scores)
    max_rows = max_rows[order]
    max_cols = max_cols[order]
    max_scores = max_scores[order]

    # Greedy selection with spacing enforcement
    selected = []
    used = np.zeros_like(score, dtype=bool)

    for r, c, s in zip(max_rows, max_cols, max_scores):
        if used[r, c]:
            continue
        if s <= 0:
            continue

        selected.append((int(r), int(c), float(s)))

        # Suppress nearby cells
        r_lo = max(0, r - spacing_px)
        r_hi = min(score.shape[0], r + spacing_px + 1)
        c_lo = max(0, c - spacing_px)
        c_hi = min(score.shape[1], c + spacing_px + 1)
        used[r_lo:r_hi, c_lo:c_hi] = True

        if len(selected) >= max_candidates:
            break

    return selected
