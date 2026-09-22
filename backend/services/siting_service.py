"""
Where to put the pond.

Three generators propose candidates: natural depressions, embankment sites
across streams (a small-scale version of the CSIRO DamSite sweep), and dug-out
sites on flat unconstrained ground.  A Boolean constraint mask removes anything
that cannot be built on, and the survivors are scored by a weighted linear
combination with AHP weights.

The constraint mask multiplies the score rather than contributing to it: in a
compensatory sum a cell in the middle of a river would otherwise score highly
because every other criterion likes it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

import config
from services.hydrology_engine import HydroBundle
from services.storage_service import stage_storage_curve

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """One proposed site, before the yield and cost numbers are attached."""

    row: int
    col: int
    kind: str                       # depression | embankment | dugout
    score: float = 0.0
    criteria: dict = field(default_factory=dict)
    curve: dict | None = None
    basin_mask: np.ndarray | None = field(default=None, repr=False)
    bund: dict | None = None
    basin_label: int = -1

    @property
    def capacity_m3(self) -> float:
        return float((self.curve or {}).get("capacity_m3", 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Criterion layers
# ─────────────────────────────────────────────────────────────────────────────


def _membership_decreasing(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """1 below a, 0 above b, linear in between."""
    return np.clip((b - x) / max(1e-9, b - a), 0.0, 1.0)


def _membership_increasing(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.clip((x - a) / max(1e-9, b - a), 0.0, 1.0)


def build_criteria(
    bundle: HydroBundle,
    constraints: dict,
    streams: dict,
    region: np.ndarray,
    cn: float,
    cfg: config.AnalysisConfig | None = None,
) -> dict:
    """The 0..1 criterion rasters that the weighted sum consumes."""
    cfg = cfg or config.AnalysisConfig()

    # Runoff potential: upstream area on a log scale, 0.5 ha to 50 ha
    upa_ha = bundle.upstream_area / 10_000.0
    runoff = _membership_increasing(np.log10(np.maximum(upa_ha, 0.01)), np.log10(0.5), np.log10(50.0))

    # Slope: flat is good
    slope = _membership_decreasing(bundle.slope_pct, 1.0, cfg.max_slope_pct_pond)

    # Distance to a stream, in cells then metres
    stream_mask = streams["mask"]
    if stream_mask.any():
        dist_cells = ndimage.distance_transform_edt(~stream_mask)
        dist_m = dist_cells * bundle.resolution_m
        stream_score = _membership_decreasing(dist_m, 0.0, 300.0)
        # Prefer low-order channels: a farm pond belongs on order 1 or 2
        order = streams["order"]
        high_order = ndimage.maximum_filter((order >= 4).astype(np.float32), size=3)
        stream_score = stream_score * (1.0 - 0.4 * high_order)
    else:
        stream_score = np.zeros(bundle.shape, dtype=np.float32)

    # Natural storage already present in the terrain
    depression = _membership_increasing(bundle.fill_depth, 0.0, 2.0)

    # Soil: inferred from the catchment curve number when no soil grid is present
    from services.curve_number_service import hsg_from_cn

    soil_value = config.HSG_SUITABILITY.get(hsg_from_cn(cn), 0.8)
    soil = np.full(bundle.shape, soil_value, dtype=np.float32)

    landuse = constraints.get("landuse_score")
    if landuse is None:
        landuse = np.full(bundle.shape, config.LANDUSE_SUITABILITY["unknown"], dtype=np.float32)

    # Distance to settlement: useful nearby, but not on top of houses
    buildings = constraints.get("building")
    if buildings is not None and buildings.any():
        dist_m = ndimage.distance_transform_edt(~buildings) * bundle.resolution_m
        settlement = np.where(
            dist_m < 500, 1.0, np.clip(1.0 - (dist_m - 500) / 2500.0, 0.4, 1.0)
        ).astype(np.float32)
    else:
        settlement = np.full(bundle.shape, 0.7, dtype=np.float32)

    return {
        "runoff": runoff.astype(np.float32),
        "slope": slope.astype(np.float32),
        "stream": stream_score.astype(np.float32),
        "depression": depression.astype(np.float32),
        "soil": soil,
        "landuse": landuse.astype(np.float32),
        "settlement": settlement,
    }


def suitability_surface(
    criteria: dict,
    constraints: dict,
    region: np.ndarray,
    bundle: HydroBundle,
    weights: dict | None = None,
    cfg: config.AnalysisConfig | None = None,
) -> np.ndarray:
    """S(c) = (product of Boolean masks) x (weighted sum of memberships)."""
    cfg = cfg or config.AnalysisConfig()
    weights = weights or cfg.weights

    score = np.zeros(bundle.shape, dtype=np.float32)
    total_weight = 0.0
    for name, weight in weights.items():
        layer = criteria.get(name)
        if layer is None or float(np.ptp(layer)) < 1e-6 and name in ("soil",):
            # A layer that is constant inside the window carries no information
            continue
        score += weight * layer
        total_weight += weight
    if total_weight > 0:
        score /= total_weight

    allowed = region.copy()
    blocked = constraints.get("blocked")
    if blocked is not None:
        allowed &= ~blocked
    allowed &= bundle.slope_pct <= cfg.max_slope_pct_embankment
    return (score * allowed).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate generation
# ─────────────────────────────────────────────────────────────────────────────


def find_candidates(
    bundle: HydroBundle,
    region: np.ndarray,
    constraints: dict,
    streams: dict,
    score: np.ndarray,
    cfg: config.AnalysisConfig | None = None,
    limit: int = 20,
) -> list:
    """Generate, deduplicate and rank candidate sites inside the region."""
    cfg = cfg or config.AnalysisConfig()
    allowed = region & ~constraints.get("blocked", np.zeros(bundle.shape, bool))

    candidates: list[Candidate] = []
    candidates += _depression_candidates(bundle, allowed, score, cfg)
    candidates += _embankment_candidates(bundle, allowed, streams, score, cfg, region)
    # Excavated ponds are always offered: on the flat terrain that most Indian
    # villages sit on, a 30 m DEM simply has no basin deep enough to matter, and
    # the honest answer is "dig one here", sized by the budget filter.
    candidates += _dugout_candidates(bundle, allowed, score, cfg)

    if not candidates:
        return []

    # Non-maximum suppression so two proposals are never on top of each other
    spacing_px = max(2, int(cfg.candidate_min_spacing_m / bundle.resolution_m))
    candidates.sort(key=lambda c: c.score, reverse=True)
    kept: list[Candidate] = []
    taken = np.zeros(bundle.shape, dtype=bool)
    seen_basins: set = set()

    for cand in candidates:
        if taken[cand.row, cand.col]:
            continue
        if cand.basin_label >= 0 and cand.basin_label in seen_basins:
            continue
        kept.append(cand)
        if cand.basin_label >= 0:
            seen_basins.add(cand.basin_label)
        r0 = max(0, cand.row - spacing_px)
        r1 = min(bundle.shape[0], cand.row + spacing_px + 1)
        c0 = max(0, cand.col - spacing_px)
        c1 = min(bundle.shape[1], cand.col + spacing_px + 1)
        taken[r0:r1, c0:c1] = True
        if len(kept) >= limit:
            break

    logger.info(
        "Candidates: %d generated, %d kept (%s)",
        len(candidates), len(kept),
        ", ".join(sorted({c.kind for c in kept})) or "none",
    )
    return kept


def _depression_candidates(bundle, allowed, score, cfg) -> list:
    """Natural basins deep enough and large enough to be real, not DEM noise."""
    depressions = (bundle.fill_depth >= cfg.min_depression_depth_m) & allowed
    if not depressions.any():
        return []

    min_cells = max(1, int(cfg.min_depression_area_ha * 10_000 / bundle.cell_area_m2))
    labels, n = ndimage.label(depressions)
    if n == 0:
        return []

    out = []
    sizes = ndimage.sum(depressions, labels, index=range(1, n + 1))
    for label_id in range(1, n + 1):
        if sizes[label_id - 1] < min_cells:
            continue
        sel = labels == label_id
        # Site = deepest cell of the basin
        depth = np.where(sel, bundle.fill_depth, -1)
        idx = int(np.argmax(depth))
        row, col = divmod(idx, bundle.shape[1])
        out.append(Candidate(
            row=row, col=col, kind="depression",
            score=float(score[row, col]),
            basin_label=int(label_id),
            criteria={"fill_depth_m": round(float(bundle.fill_depth[row, col]), 2)},
        ))
    return out


def _embankment_candidates(bundle, allowed, streams, score, cfg, region=None) -> list:
    """
    Sweep crest heights across stream cells and keep the best ratio of ponded
    volume to bund volume, the DamSite idea at village scale.
    """
    stream_mask = streams["mask"] & allowed
    if not stream_mask.any():
        return []

    upa_ha = bundle.upstream_area / 10_000.0
    eligible = (
        stream_mask
        & (upa_ha >= cfg.embankment_min_upstream_ha)
        & (upa_ha <= cfg.embankment_max_upstream_ha)
    )
    if not eligible.any():
        return []

    rows, cols = np.where(eligible)
    # Evaluate the most promising cells only: the sweep is the expensive part
    cell_scores = score[rows, cols]
    order = np.argsort(cell_scores)[::-1][:40]
    out = []

    # A village structure impounds a few hundred metres of valley at most.  The
    # sweep is confined to a window of that size and any crest whose water
    # reaches the window edge is rejected: the reservoir would spill somewhere
    # else, which is the saddle-leak case DamSite handles with saddle dams.
    radius = max(4, int(round(cfg.embankment_max_extent_m / bundle.resolution_m)))

    for i in order:
        row, col = int(rows[i]), int(cols[i])
        # The pond must stay inside the area the user drew, as well as inside
        # the window: they asked for a pond there, not one that happens to
        # spill across the boundary.
        window = _local_window(bundle.shape, row, col, radius)
        if region is not None:
            window &= region
        best = None
        for height in cfg.embankment_heights_m:
            level = float(bundle.dem[row, col]) + height
            curve = stage_storage_curve(
                bundle.dem, (row, col), bundle.cell_area_m2,
                spill_level=level, bounds_mask=window,
                step_m=max(0.2, cfg.stage_step_m * 2),
            )
            volume = curve["capacity_m3"]
            mask = curve.get("mask")
            if volume <= 0 or mask is None:
                continue
            if _escapes_window(mask, window):
                break        # taller crests only leak more, stop the sweep here
            bund = _bund_estimate(bundle, curve, level, cfg)
            if bund is None or bund["volume_m3"] <= 0:
                continue
            if bund["max_height_m"] > config.GeometryConfig().bund_height_max_m:
                break
            ratio = volume / bund["volume_m3"]
            if best is None or ratio > best["ratio"]:
                best = {"ratio": ratio, "curve": curve, "bund": bund, "height": height}
        if best is None:
            continue
        out.append(Candidate(
            row=row, col=col, kind="embankment",
            score=float(score[row, col]) * float(np.clip(best["ratio"] / 20.0, 0.3, 1.2)),
            curve=best["curve"],
            bund=best["bund"],
            criteria={
                "crest_height_m": best["height"],
                "storage_per_bund_m3": round(best["ratio"], 2),
                "upstream_ha": round(float(upa_ha[row, col]), 2),
            },
        ))
    return out


def _local_window(shape_hw: tuple, row: int, col: int, radius: int) -> np.ndarray:
    """Boolean mask of a square window, used to confine a flood fill."""
    window = np.zeros(shape_hw, dtype=bool)
    r0, r1 = max(0, row - radius), min(shape_hw[0], row + radius + 1)
    c0, c1 = max(0, col - radius), min(shape_hw[1], col + radius + 1)
    window[r0:r1, c0:c1] = True
    return window


def _escapes_window(wet: np.ndarray, window: np.ndarray) -> bool:
    """
    True when the ponded area reaches the boundary of its containment window,
    which means the impoundment is not closed and the water would go elsewhere.
    """
    from scipy import ndimage

    boundary = window & ~ndimage.binary_erosion(window)
    return bool((wet & boundary).any())


def _dugout_candidates(bundle, allowed, score, cfg) -> list:
    """Flat, unconstrained ground for an excavated pond when nothing else fits."""
    flat = allowed & (bundle.slope_pct <= 2.0)
    if not flat.any():
        flat = allowed & (bundle.slope_pct <= 5.0)
    if not flat.any():
        return []

    masked = np.where(flat, score, -1)
    n_take = min(8, int(flat.sum()))
    idxs = np.argpartition(masked.ravel(), -n_take)[-n_take:]
    out = []
    for idx in idxs:
        if masked.ravel()[idx] <= 0:
            continue
        row, col = divmod(int(idx), bundle.shape[1])
        out.append(Candidate(
            row=row, col=col, kind="dugout",
            score=float(score[row, col]) * 0.8,   # no natural storage to start from
            criteria={"slope_pct": round(float(bundle.slope_pct[row, col]), 2)},
        ))
    return out


def _bund_estimate(bundle: HydroBundle, curve: dict, crest_level: float, cfg) -> dict | None:
    """
    Volume of the earth bund needed to hold water to ``crest_level``.

    The rim is the boundary of the ponded area; each rim cell contributes a
    trapezoidal cross-section of height (crest + freeboard - ground).
    """
    mask = curve.get("mask")
    if mask is None or not mask.any():
        return None

    geom = config.GeometryConfig()
    rim = ndimage.binary_dilation(mask) & ~mask
    if not rim.any():
        return None

    ground = bundle.dem[rim]
    design_h = np.maximum(0.0, crest_level + geom.freeboard_m - ground)
    design_h = design_h[design_h > 0]
    if design_h.size == 0:
        return None

    construction_h = design_h / (1.0 - geom.settlement_allowance)
    section = geom.bund_top_width_m * construction_h + geom.bund_side_slope * construction_h ** 2
    volume = float(section.sum() * bundle.resolution_m)

    return {
        "volume_m3": round(volume, 1),
        "length_m": round(float(design_h.size * bundle.resolution_m), 1),
        "max_height_m": round(float(construction_h.max()), 2),
        "mean_height_m": round(float(construction_h.mean()), 2),
        "rim_cells": int(design_h.size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confidence
# ─────────────────────────────────────────────────────────────────────────────


def weight_robustness(
    criteria: dict,
    candidates: list,
    weights: dict,
    top_k: int = 3,
    draws: int = 200,
    concentration: float = 60.0,
    seed: int = 12345,
) -> dict:
    """
    Share of perturbed weight sets in which each site stays in the top k.

    The score is linear in the weights, so the criterion values at the candidate
    cells can be sampled once and all draws evaluated in a single matrix
    multiply, which keeps this at a few milliseconds.
    """
    if not candidates:
        return {}

    names = [n for n in weights if n in criteria]
    if not names:
        return {i: 1.0 for i in range(len(candidates))}

    values = np.array([
        [float(criteria[n][c.row, c.col]) for n in names] for c in candidates
    ])                                              # (candidates, criteria)

    base = np.array([weights[n] for n in names], dtype=float)
    base = base / base.sum()

    rng = np.random.default_rng(seed)
    sampled = rng.dirichlet(base * concentration, size=draws)   # (draws, criteria)
    scores = values @ sampled.T                                  # (candidates, draws)

    k = min(top_k, len(candidates))
    ranks = np.argsort(np.argsort(-scores, axis=0), axis=0)
    in_top = (ranks < k).mean(axis=1)
    return {i: float(round(v, 3)) for i, v in enumerate(in_top)}


def confidence_label(value: float) -> str:
    if value >= 0.7:
        return "high"
    if value >= 0.4:
        return "medium"
    return "low"


def structure_type(catchment_ha: float, slope_pct: float, depth_m: float, kind: str) -> dict:
    """
    Label the site with the structure the Indian guidelines would call for
    (Ramakrishnan et al. 2009, after IMSD 1995 and the INCOH tables).
    """
    if kind == "embankment":
        preferred = "check_dam"
    elif catchment_ha >= 25:
        preferred = "percolation_tank"
    else:
        preferred = "farm_pond"

    rule = config.STRUCTURE_RULES[preferred]
    lo_c, hi_c = rule["catchment_ha"]
    fits = (
        rule["slope_pct"][0] <= slope_pct <= rule["slope_pct"][1]
        and lo_c * 0.5 <= catchment_ha <= hi_c * 2.0
    )
    return {
        "type": preferred,
        "label": rule["label"],
        "matches_guideline": bool(fits),
        "guideline_storage_m3": list(rule["storage_m3"]),
        "guideline_water_level_m": list(rule["water_level_m"]),
    }
