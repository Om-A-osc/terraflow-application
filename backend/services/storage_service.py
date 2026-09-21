"""
Stage, area and storage of a pond basin.

Everything downstream reads this one curve: the capacity reported per site, the
chart in the sidebar, the water balance (which needs the water-spread area at a
given volume), and the earthwork optimiser (which needs dV/dS = A(S) to price a
raise of the spill level).

The curve is integrated on the *raw* DEM, never the filled one: filling is a
routing device, and using it would count storage that the terrain does not have.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

import config

logger = logging.getLogger(__name__)


def basin_at(
    dem: np.ndarray,
    row: int,
    col: int,
    fill_depth: np.ndarray | None = None,
    max_radius_cells: int = 60,
) -> np.ndarray | None:
    """
    The connected depression containing (row, col).

    Uses the fill-depth layer when available (cells that a flood fill raised are
    exactly the natural basins); otherwise grows a local region below the median
    elevation of its surroundings.
    """
    if fill_depth is not None:
        depressions = fill_depth > 0.01
        if depressions.any():
            labels, _ = ndimage.label(depressions)
            label = int(labels[row, col])
            if label == 0:
                # Search outward for the nearest depression
                r0 = max(0, row - max_radius_cells)
                r1 = min(dem.shape[0], row + max_radius_cells + 1)
                c0 = max(0, col - max_radius_cells)
                c1 = min(dem.shape[1], col + max_radius_cells + 1)
                window = labels[r0:r1, c0:c1]
                nonzero = window[window > 0]
                if nonzero.size:
                    label = int(np.bincount(nonzero).argmax())
            if label > 0:
                return labels == label
    return None


def flood_fill_at_level(dem: np.ndarray, row: int, col: int, level: float,
                        bounds_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Cells that would be underwater at ``level``, connected to the seed.

    This is the r.lake logic: threshold, label, keep the component containing
    the seed, so a puddle on the other side of a ridge is not counted.
    """
    below = dem < level
    if bounds_mask is not None:
        below &= bounds_mask
    if not below[row, col]:
        below[row, col] = True
    labels, _ = ndimage.label(below)
    seed = int(labels[row, col])
    if seed == 0:
        return np.zeros_like(below)
    return labels == seed


def stage_storage_curve(
    dem: np.ndarray,
    seed: tuple,
    cell_area_m2: float,
    spill_level: float | None = None,
    bounds_mask: np.ndarray | None = None,
    step_m: float = 0.1,
    max_levels: int = 120,
) -> dict:
    """
    Build the stage-area-storage curve for the basin around ``seed``.

    Returns arrays of level, area and volume plus the spill level and the
    capacity below it (less freeboard), in the shape the API and the chart use.
    """
    row, col = seed
    if spill_level is None:
        spill_level = _find_spill_level(dem, seed, bounds_mask)

    seed_elev = float(dem[row, col])
    if not np.isfinite(spill_level) or spill_level < seed_elev:
        return _degenerate_curve(seed_elev)

    # The bed is the lowest point of the pond, which is not necessarily the
    # cell the candidate sits on: a site found by score can be on the shoulder
    # of the hollow rather than at its bottom.
    full = flood_fill_at_level(dem, row, col, spill_level, bounds_mask)
    bed = float(dem[full].min()) if full.any() else seed_elev
    if spill_level <= bed:
        return _degenerate_curve(bed)

    n_steps = min(max_levels, max(2, int(np.ceil((spill_level - bed) / step_m)) + 1))
    levels = np.linspace(bed, spill_level, n_steps)

    cfg = config.AnalysisConfig()
    usable_level = max(bed, spill_level - cfg.freeboard_m)

    areas, volumes = [], []
    surface_mask = None
    best_gap = None
    for level in levels:
        wet = flood_fill_at_level(dem, row, col, level, bounds_mask)
        depth = np.where(wet, level - dem, 0.0)
        depth = np.maximum(depth, 0.0)
        areas.append(float(wet.sum()) * cell_area_m2)
        volumes.append(float(depth.sum()) * cell_area_m2)

        # Kept for drawing: the water surface at the level the pond actually
        # holds, which is the spill level less the freeboard.  Drawing the
        # brim-full extent instead made the pond look wider than the capacity
        # printed beside it.
        gap = abs(level - usable_level)
        if best_gap is None or gap < best_gap:
            best_gap, surface_mask = gap, wet
    capacity = float(np.interp(usable_level, levels, volumes))
    dead = float(np.interp(min(bed + cfg.dead_storage_m, usable_level), levels, volumes))

    return {
        "level_m": [round(float(v), 2) for v in levels],
        "area_m2": [round(v, 1) for v in areas],
        "volume_m3": [round(v, 1) for v in volumes],
        "bed_level_m": round(bed, 2),
        "spill_level_m": round(float(spill_level), 2),
        "usable_level_m": round(float(usable_level), 2),
        "capacity_m3": round(capacity, 1),
        "dead_storage_m3": round(dead, 1),
        "live_storage_m3": round(max(0.0, capacity - dead), 1),
        "surface_area_m2": round(float(np.interp(usable_level, levels, areas)), 1),
        "max_depth_m": round(float(usable_level - bed), 2),
        # The brim-full extent.  Containment and escape tests need this one: a
        # pond is only closed if the water cannot get out at its highest, not
        # at its working level.
        "mask": full,
        # The working water surface, which is what gets drawn.
        "surface_mask": surface_mask if surface_mask is not None else full,
    }


def _degenerate_curve(level: float) -> dict:
    """A basin with no storage: returned rather than raising, so ranking can drop it."""
    return {
        "level_m": [round(level, 2)],
        "area_m2": [0.0],
        "volume_m3": [0.0],
        "bed_level_m": round(level, 2),
        "spill_level_m": round(level, 2),
        "usable_level_m": round(level, 2),
        "capacity_m3": 0.0,
        "dead_storage_m3": 0.0,
        "live_storage_m3": 0.0,
        "surface_area_m2": 0.0,
        "max_depth_m": 0.0,
        "mask": None,
        "surface_mask": None,
    }


def _find_spill_level(dem: np.ndarray, seed: tuple, bounds_mask: np.ndarray | None) -> float:
    """
    Lowest level at which the water escapes the basin.

    The escape test floods *without* the containment mask and then asks whether
    the water left it, which is the whole point of the test.  Restricting the
    fill first would make escape impossible to detect and the search would run
    all the way to the surrounding hilltops.

    Bisection between the bed and the local maximum costs eighteen flood fills
    instead of one per candidate level.
    """
    row, col = seed
    bed = float(dem[row, col])

    radius = 80
    r0, r1 = max(0, row - radius), min(dem.shape[0], row + radius + 1)
    c0, c1 = max(0, col - radius), min(dem.shape[1], col + radius + 1)
    local_max = float(np.nanmax(dem[r0:r1, c0:c1]))
    if not np.isfinite(local_max) or local_max <= bed:
        return bed

    def escapes(level: float) -> bool:
        wet = flood_fill_at_level(dem, row, col, level, bounds_mask=None)
        if bounds_mask is not None and (wet & ~bounds_mask).any():
            return True
        return bool(
            wet[0, :].any() or wet[-1, :].any() or wet[:, 0].any() or wet[:, -1].any()
        )

    lo, hi = bed, local_max
    if not escapes(hi):
        return hi
    for _ in range(18):                    # ~0.1 mm over a 30 m range
        mid = (lo + hi) / 2
        if escapes(mid):
            hi = mid
        else:
            lo = mid
    return lo


def spill_level_from_fill(dem: np.ndarray, fill_depth: np.ndarray, basin: np.ndarray) -> float:
    """
    Spill level of a natural depression, read straight off the filled surface.

    Depression filling raises every cell of a basin to its outlet elevation, so
    the filled surface is flat at exactly the spill level.  That is both cheaper
    and more reliable than searching for it.
    """
    if basin is None or not basin.any():
        return float("nan")
    return float((dem + fill_depth)[basin].max())


def curve_for_payload(curve: dict, max_points: int = 40) -> dict:
    """Trim the curve for the API response and drop the raster mask."""
    levels = curve.get("level_m", [])
    if len(levels) <= max_points:
        subset = range(len(levels))
    else:
        subset = np.unique(np.linspace(0, len(levels) - 1, max_points).astype(int))
    return {
        "level_m": [levels[i] for i in subset],
        "area_m2": [curve["area_m2"][i] for i in subset],
        "volume_m3": [curve["volume_m3"][i] for i in subset],
        "bed_level_m": curve.get("bed_level_m"),
        "spill_level_m": curve.get("spill_level_m"),
        "usable_level_m": curve.get("usable_level_m"),
        "capacity_m3": curve.get("capacity_m3"),
        "dead_storage_m3": curve.get("dead_storage_m3"),
        "live_storage_m3": curve.get("live_storage_m3"),
    }


def capacity_band(capacity_m3: float, curve: dict, uncertainty_m: float | None = None) -> dict:
    """
    Low and high capacity for the stated DEM vertical uncertainty.

    A global 30 m DEM is accurate to a couple of metres, which is the same order
    as the design depth of a farm pond, so a single number would be misleading.
    """
    cfg = config.AnalysisConfig()
    uncertainty_m = uncertainty_m if uncertainty_m is not None else cfg.dem_vertical_uncertainty_m
    levels = curve.get("level_m") or []
    volumes = curve.get("volume_m3") or []
    if not levels:
        return {"low_m3": capacity_m3, "best_m3": capacity_m3, "high_m3": capacity_m3}

    usable = curve.get("usable_level_m", levels[-1])
    half = uncertainty_m / 2.0
    low = float(np.interp(max(levels[0], usable - half), levels, volumes))
    high = float(np.interp(min(levels[-1], usable + half), levels, volumes))
    return {
        "low_m3": round(min(low, capacity_m3), 1),
        "best_m3": round(capacity_m3, 1),
        "high_m3": round(max(high, capacity_m3), 1),
        "uncertainty_m": uncertainty_m,
    }
