"""
Budget-constrained earthwork: how much more water can this site hold?

Level 1 is a two-variable search over (d, h): d is extra excavation depth over
the cells that already hold water, h is how far the spill level is raised by an
earth bund on the rim.  Both move in 0.1 m steps and each evaluation is a few
NumPy sums, so the whole grid of designs is priced in well under a second.

Level 2 is optional and prices *where* the soil goes: cut and fill cells are
aggregated into super-cells and the haul is solved as a transportation problem
with borrow and waste slack, using the HiGHS solver inside SciPy.

Volume bookkeeping is in bank cubic metres throughout.  Schedule-of-rates
excavation items are per bank cubic metre and banking items per compacted cubic
metre, so shrink converts a bund volume into the cut it consumes, and swell is
used only if someone wants truck counts.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

import config

logger = logging.getLogger(__name__)


def optimise(
    dem: np.ndarray,
    footprint_mask: np.ndarray,
    curve: dict,
    cell_area_m2: float,
    resolution_m: float,
    budget: float,
    costs: config.CostConfig | None = None,
    geometry: config.GeometryConfig | None = None,
    include_haul_plan: bool = False,
    seed: tuple | None = None,
    containment: np.ndarray | None = None,
) -> dict:
    """
    Maximise added storage subject to cost <= budget.

    Returns the chosen design, the cost breakdown, the balanced (cut equals
    fill) alternative, and any warnings a reviewer should see.
    """
    costs = (costs or config.CostConfig()).resolved()
    geometry = geometry or config.GeometryConfig()

    base_level = float(curve.get("usable_level_m") or curve.get("spill_level_m") or 0.0)
    spill_level = float(curve.get("spill_level_m") or base_level)
    base_capacity = float(curve.get("capacity_m3") or 0.0)

    wet = footprint_mask & (dem < spill_level)
    if not wet.any():
        # A dug-out pond on flat ground: treat the footprint itself as the floor
        wet = footprint_mask.copy()
    n_wet = int(wet.sum())
    if n_wet == 0:
        return _empty_result(budget, base_capacity, "The selected footprint has no area to deepen")

    rim = ndimage.binary_dilation(footprint_mask) & ~footprint_mask
    rim_ground = dem[rim] if rim.any() else np.array([spill_level])

    # Grids of the two decision variables
    depths = np.round(np.arange(0.0, geometry.max_dig_depth_m + 1e-9, 0.1), 2)
    raises = np.round(np.arange(0.0, geometry.max_spill_raise_m + 1e-9, 0.1), 2)

    # Raising the spill level is evaluated on the DEM itself, inside a
    # containment window, rather than extrapolated off the end of the
    # stage-storage curve.  Anything else lets the reported volume and the
    # footprint drawn on the map disagree: above the natural spill the water
    # spreads far wider than the basin the curve was built from.
    if seed is None:
        flat = int(np.argmax(np.where(footprint_mask, -dem, -np.inf)))
        seed = (flat // dem.shape[1], flat % dem.shape[1])
    if containment is None:
        containment = _window_around(dem.shape, seed[0], seed[1], resolution_m)

    raise_table = _raise_table(
        dem, seed, containment, base_level, raises, cell_area_m2, resolution_m, geometry
    )
    feasible_raises = [entry for entry in raise_table if not entry["escapes"]]
    if not feasible_raises:
        feasible_raises = raise_table[:1]        # the do-nothing option always stands

    results = []
    for entry in feasible_raises:
        h = entry["raise_m"]
        bund = entry["bund"]
        fill_needed_bank = bund["compacted_m3"] / max(1e-9, 1.0 - geometry.shrink)
        gain_raise = entry["added_m3"]

        for d in depths:
            # Deepening gains a cubic metre of storage per cubic metre cut, over
            # whatever area is wet at this level.
            wet_cells = max(n_wet, entry["wet_cells"])
            cut = wet_cells * cell_area_m2 * float(d)      # bank m3
            gain = cut + gain_raise

            cost = _cost_of(cut, fill_needed_bank, bund["compacted_m3"], costs, resolution_m, wet_cells)
            results.append((float(d), float(h), gain, cost, bund, cut, fill_needed_bank))

    affordable = [r for r in results if r[3]["total"] <= budget]
    if affordable:
        best = max(affordable, key=lambda r: r[2])
    else:
        best = min(results, key=lambda r: r[3]["total"])    # cheapest, still over budget

    d, h, gain, cost, bund, cut, fill_bank = best
    within_budget = cost["total"] <= budget

    # Balanced design: cut equals the bund's demand, the FAO cut-and-fill pond
    def storage_from_raise(h: float) -> float:
        """Added volume for a raise, read off the table built from the DEM."""
        if h <= 0:
            return 0.0
        best_entry = min(raise_table, key=lambda e: abs(e["raise_m"] - h))
        return 0.0 if best_entry["escapes"] else best_entry["added_m3"]

    balanced = _balanced_design(
        n_wet, cell_area_m2, rim_ground, base_level, resolution_m, geometry, costs, storage_from_raise
    )

    warnings = []
    if bund["max_height_m"] > geometry.bund_height_max_m:
        warnings.append(
            f"Bund height {bund['max_height_m']:.1f} m exceeds the {geometry.bund_height_max_m:.0f} m "
            "limit for a small earthen structure; treat this as a dam design, not a farm pond"
        )
    elif bund["max_height_m"] > geometry.bund_height_warn_m:
        warnings.append(
            f"Bund height {bund['max_height_m']:.1f} m is above {geometry.bund_height_warn_m:.0f} m; "
            "it needs a spillway and a proper foundation"
        )
    if not within_budget:
        warnings.append(
            "Even the smallest viable design costs more than the budget; the figures shown are "
            f"for the cheapest option at Rs {cost['total']:,.0f}"
        )
    if h > 0 and not rim.any():
        warnings.append("No closed rim was found, so the raised level may spill elsewhere")
    if any(entry["escapes"] for entry in raise_table if entry["raise_m"] <= h):
        warnings.append(
            "Above this level the water escapes the basin, so the raise has been capped"
        )

    uncertainty = config.AnalysisConfig().dem_vertical_uncertainty_m
    band_factor = 1.0 + uncertainty / max(1.0, (base_level - float(curve.get("bed_level_m", base_level - 1)) + d + h))

    result = {
        "within_budget": within_budget,
        "budget": round(budget, 2),
        "design": {
            "extra_depth_m": round(d, 2),
            "spill_raise_m": round(h, 2),
            "new_usable_level_m": round(base_level + h, 2),
            "excavation_m3": round(cut, 1),
            "bund_volume_m3": round(bund["compacted_m3"], 1),
            "bund_fill_bank_m3": round(fill_bank, 1),
            "bund_max_height_m": bund["max_height_m"],
            "bund_length_m": bund["length_m"],
            "wetted_cells": n_wet,
        },
        "storage": {
            "base_capacity_m3": round(base_capacity, 1),
            "added_m3": round(gain, 1),
            "new_capacity_m3": round(base_capacity + gain, 1),
            "added_low_m3": round(gain / band_factor, 1),
            "added_high_m3": round(gain * band_factor, 1),
            "from_deepening_m3": round(cut, 1),
            "from_raising_m3": round(gain - cut, 1),
        },
        "cost": cost,
        "cost_per_m3_stored": round(cost["total"] / gain, 2) if gain > 0 else None,
        "balanced_design": balanced,
        "warnings": warnings,
        "assumptions": {
            "excavation_per_m3": costs.excavation_per_m3,
            "fill_per_m3": costs.fill_per_m3,
            "haul_per_m3_per_50m": costs.haul_per_m3_per_50m,
            "borrow_per_m3": costs.borrow_per_m3,
            "manual_rates": costs.manual_rates,
            "bund_top_width_m": geometry.bund_top_width_m,
            "bund_side_slope": geometry.bund_side_slope,
            "freeboard_m": geometry.freeboard_m,
            "settlement_allowance": geometry.settlement_allowance,
            "shrink": geometry.shrink,
            "source": "CPWD DSR 2023 chapter 2; FAO pond construction manual",
        },
    }

    if include_haul_plan and cut > 0:
        result["haul_plan"] = haul_plan(
            dem, wet, rim, d, base_level + h, cell_area_m2, resolution_m, costs, geometry
        )
    return result


def _cost_of(cut_bank, fill_bank_needed, fill_compacted, costs, resolution_m, n_wet) -> dict:
    """Price a design; soil moves from the dig to the bund, the rest is borrowed or wasted."""
    reused = min(cut_bank, fill_bank_needed)
    borrow = max(0.0, fill_bank_needed - cut_bank)
    waste = max(0.0, cut_bank - fill_bank_needed)

    # Mean haul distance: roughly the radius of the wetted area plus the bund offset
    radius_m = float(np.sqrt(max(1.0, n_wet)) * resolution_m / 2.0)
    haul_distance_m = max(0.0, radius_m - costs.free_lead_m)
    slabs = np.ceil(haul_distance_m / 50.0) if haul_distance_m > 0 else 0.0
    haul_rate = slabs * costs.haul_per_m3_per_50m

    excavation = cut_bank * costs.excavation_per_m3
    placement = fill_compacted * costs.fill_per_m3
    haulage = reused * haul_rate
    borrowing = borrow * costs.borrow_per_m3
    disposal = waste * costs.waste_per_m3

    total = excavation + placement + haulage + borrowing + disposal
    return {
        "excavation": round(excavation, 2),
        "placement": round(placement, 2),
        "haulage": round(haulage, 2),
        "borrow": round(borrowing, 2),
        "disposal": round(disposal, 2),
        "total": round(total, 2),
        "haul_distance_m": round(haul_distance_m, 1),
        "reused_m3": round(reused, 1),
        "borrowed_m3": round(borrow, 1),
        "wasted_m3": round(waste, 1),
    }


def _bund_volume(rim_ground: np.ndarray, crest_level: float, resolution_m: float, geometry) -> dict:
    """
    Trapezoidal bund along the rim.

    Section area = T x CH + m x CH^2 for equal side slopes; construction height
    carries the settlement allowance.  Cost grows roughly with the square of the
    height, which is why the optimiser is capped rather than left to push it.
    """
    design_h = np.maximum(0.0, crest_level + geometry.freeboard_m - np.asarray(rim_ground, dtype=float))
    positive = design_h[design_h > 0]
    if positive.size == 0:
        return {"compacted_m3": 0.0, "max_height_m": 0.0, "mean_height_m": 0.0, "length_m": 0.0}

    construction_h = positive / (1.0 - geometry.settlement_allowance)
    section = geometry.bund_top_width_m * construction_h + geometry.bund_side_slope * construction_h ** 2
    return {
        "compacted_m3": float(section.sum() * resolution_m),
        "max_height_m": round(float(construction_h.max()), 2),
        "mean_height_m": round(float(construction_h.mean()), 2),
        "length_m": round(float(positive.size * resolution_m), 1),
    }


def _balanced_design(
    n_wet, cell_area_m2, rim_ground, base_level, resolution_m, geometry, costs, storage_from_raise
) -> dict:
    """
    The FAO cut-and-fill pond: dig exactly as much as the bund needs, so no soil
    is bought or dumped.  Found by bisection on the raise, both sides monotone.
    """
    lo, hi = 0.0, geometry.max_spill_raise_m
    best = None
    for _ in range(24):
        h = (lo + hi) / 2
        bund = _bund_volume(rim_ground, base_level + h, resolution_m, geometry)
        fill_bank = bund["compacted_m3"] / max(1e-9, 1.0 - geometry.shrink)
        d = fill_bank / max(1e-9, n_wet * cell_area_m2)
        if d > geometry.max_dig_depth_m:
            hi = h
        else:
            lo = h
            best = (h, d, bund, fill_bank)
    if best is None:
        return {}
    h, d, bund, fill_bank = best
    cut = n_wet * cell_area_m2 * d
    cost = _cost_of(cut, fill_bank, bund["compacted_m3"], costs, resolution_m, n_wet)
    gain = cut + storage_from_raise(h)
    return {
        "extra_depth_m": round(d, 2),
        "spill_raise_m": round(h, 2),
        "added_storage_m3": round(gain, 1),
        "cost": cost["total"],
        "note": "Cut balances the bund, so no soil is bought or dumped",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: where the soil goes
# ─────────────────────────────────────────────────────────────────────────────


def haul_plan(
    dem, wet_mask, rim_mask, depth_m, crest_level, cell_area_m2, resolution_m,
    costs, geometry, max_nodes: int = 300, time_limit_s: float = 5.0,
) -> dict:
    """
    Least-cost allocation of cut to fill, as a transportation linear programme.

    Cut and fill cells are aggregated into super-cells so the problem stays near
    a few hundred by a few hundred; a 400 by 400 instance solves in about 1.6 s
    with HiGHS, and the optimal basis has at most m + n - 1 non-zero flows, so
    the arrows drawn on the map stay few.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    supplies, supply_xy = _aggregate(wet_mask, dem, resolution_m, depth_m * cell_area_m2, max_nodes)
    if not supplies:
        return {"solved": False, "reason": "No excavation in the chosen design"}

    demand_heights = np.maximum(
        0.0, crest_level + geometry.freeboard_m - dem[rim_mask]
    ) / (1.0 - geometry.settlement_allowance)
    if demand_heights.size == 0 or demand_heights.max() <= 0:
        return {"solved": False, "reason": "No bund is required in the chosen design"}

    section = (
        geometry.bund_top_width_m * demand_heights
        + geometry.bund_side_slope * demand_heights ** 2
    ) * resolution_m / max(1e-9, 1.0 - geometry.shrink)
    demands, demand_xy = _aggregate_values(rim_mask, section, resolution_m, max_nodes)
    if not demands:
        return {"solved": False, "reason": "No bund cells to fill"}

    m, n = len(supplies), len(demands)
    sx = np.array([p[0] for p in supply_xy]); sy = np.array([p[1] for p in supply_xy])
    dx = np.array([p[0] for p in demand_xy]); dy = np.array([p[1] for p in demand_xy])
    dist = np.sqrt((sx[:, None] - dx[None, :]) ** 2 + (sy[:, None] - dy[None, :]) ** 2)

    slabs = np.ceil(np.maximum(0.0, dist - costs.free_lead_m) / 50.0)
    haul_cost = np.minimum(slabs * costs.haul_per_m3_per_50m,
                           costs.haul_per_m3_per_km * dist / 1000.0 + costs.haul_per_m3_per_50m)
    unit_cost = costs.excavation_per_m3 + costs.fill_per_m3 + haul_cost

    # Variables: x_ij (m*n), waste_i (m), borrow_j (n)
    c = np.concatenate([
        unit_cost.ravel(),
        np.full(m, costs.excavation_per_m3 + costs.waste_per_m3),
        np.full(n, costs.borrow_per_m3),
    ])

    rows_supply = sparse.hstack([
        sparse.kron(sparse.eye(m), np.ones((1, n))),
        sparse.eye(m),
        sparse.csr_matrix((m, n)),
    ])
    rows_demand = sparse.hstack([
        sparse.kron(np.ones((1, m)), sparse.eye(n)) * (1.0 - geometry.shrink),
        sparse.csr_matrix((n, m)),
        sparse.eye(n),
    ])
    a_eq = sparse.vstack([rows_supply, rows_demand]).tocsr()
    b_eq = np.concatenate([np.array(supplies), np.array(demands)])

    res = linprog(
        c, A_eq=a_eq, b_eq=b_eq, bounds=(0, None),
        method="highs", options={"time_limit": time_limit_s, "presolve": True},
    )
    if not res.success or res.x is None:
        logger.warning("Haul LP did not solve: %s", res.message)
        return {"solved": False, "reason": res.message}

    flows = res.x[: m * n].reshape(m, n)
    arrows = []
    for i, j in zip(*np.nonzero(flows > 1.0)):
        arrows.append({
            "from": [float(sx[i]), float(sy[i])],
            "to": [float(dx[j]), float(dy[j])],
            "volume_m3": round(float(flows[i, j]), 1),
            "distance_m": round(float(dist[i, j]), 1),
        })
    arrows.sort(key=lambda a: -a["volume_m3"])

    return {
        "solved": True,
        "objective_rupees": round(float(res.fun), 2),
        "cut_nodes": m,
        "fill_nodes": n,
        "arrows": arrows[:60],
        "wasted_m3": round(float(res.x[m * n: m * n + m].sum()), 1),
        "borrowed_m3": round(float(res.x[m * n + m:].sum()), 1),
        "mean_haul_m": round(
            float((flows * dist).sum() / max(1e-9, flows.sum())), 1
        ),
    }


def _aggregate(mask, dem, resolution_m, per_cell_volume, max_nodes) -> tuple:
    """Group mask cells into super-cells carrying an equal volume each."""
    values = np.where(mask, per_cell_volume, 0.0)
    return _aggregate_values(mask, values[mask], resolution_m, max_nodes)


def _aggregate_values(mask, values, resolution_m, max_nodes) -> tuple:
    rows, cols = np.where(mask)
    if rows.size == 0:
        return [], []
    values = np.asarray(values, dtype=float).ravel()
    if values.size != rows.size:
        values = np.full(rows.size, float(values.sum()) / rows.size)

    # Block size chosen so the node count lands under the cap
    factor = max(1, int(np.ceil(np.sqrt(rows.size / max(1, max_nodes)))))
    keys = (rows // factor) * 100000 + (cols // factor)
    order = np.argsort(keys)
    keys, rows, cols, values = keys[order], rows[order], cols[order], values[order]
    boundaries = np.flatnonzero(np.diff(keys)) + 1

    totals, centres = [], []
    for chunk_rows, chunk_cols, chunk_vals in zip(
        np.split(rows, boundaries), np.split(cols, boundaries), np.split(values, boundaries)
    ):
        total = float(chunk_vals.sum())
        if total <= 0:
            continue
        totals.append(total)
        centres.append((
            float(chunk_cols.mean() * resolution_m),
            float(chunk_rows.mean() * resolution_m),
        ))
    return totals, centres


def _empty_result(budget: float, base_capacity: float, reason: str) -> dict:
    return {
        "within_budget": True,
        "budget": budget,
        "design": {"extra_depth_m": 0.0, "spill_raise_m": 0.0, "excavation_m3": 0.0},
        "storage": {
            "base_capacity_m3": round(base_capacity, 1),
            "added_m3": 0.0,
            "new_capacity_m3": round(base_capacity, 1),
        },
        "cost": {"total": 0.0},
        "warnings": [reason],
    }


def _window_around(shape_hw: tuple, row: int, col: int, resolution_m: float,
                   extent_m: float = 400.0) -> np.ndarray:
    """Square containment mask, the scale of a village structure."""
    radius = max(4, int(round(extent_m / resolution_m)))
    window = np.zeros(shape_hw, dtype=bool)
    r0, r1 = max(0, row - radius), min(shape_hw[0], row + radius + 1)
    c0, c1 = max(0, col - radius), min(shape_hw[1], col + radius + 1)
    window[r0:r1, c0:c1] = True
    return window


def _raise_table(dem, seed, containment, base_level, raises, cell_area_m2, resolution_m, geometry):
    """
    For each candidate raise of the spill level, flood the DEM and measure what
    the water actually does: how much more it holds, how wide it spreads, what
    bund would be needed, and whether it escapes the containment window.
    """
    from services.storage_service import flood_fill_at_level

    row, col = int(seed[0]), int(seed[1])
    boundary = containment & ~ndimage.binary_erosion(containment)

    base_wet = flood_fill_at_level(dem, row, col, base_level, bounds_mask=containment)
    base_volume = float(np.maximum(base_level - dem[base_wet], 0.0).sum()) * cell_area_m2 if base_wet.any() else 0.0

    table = []
    escaped = False
    for h in raises:
        level = base_level + float(h)
        if escaped:
            table.append({"raise_m": float(h), "escapes": True, "added_m3": 0.0,
                          "wet_cells": 0, "bund": _empty_bund()})
            continue

        wet = flood_fill_at_level(dem, row, col, level, bounds_mask=containment)
        if not wet.any():
            table.append({"raise_m": float(h), "escapes": False, "added_m3": 0.0,
                          "wet_cells": 0, "bund": _empty_bund()})
            continue

        if float(h) > 0 and bool((wet & boundary).any()):
            escaped = True
            table.append({"raise_m": float(h), "escapes": True, "added_m3": 0.0,
                          "wet_cells": int(wet.sum()), "bund": _empty_bund()})
            continue

        volume = float(np.maximum(level - dem[wet], 0.0).sum()) * cell_area_m2
        rim = ndimage.binary_dilation(wet) & ~wet
        bund = (
            _bund_volume(dem[rim], level, resolution_m, geometry)
            if rim.any() and h > 0
            else _empty_bund()
        )
        table.append({
            "raise_m": float(h),
            "escapes": False,
            "added_m3": max(0.0, volume - base_volume),
            "wet_cells": int(wet.sum()),
            "bund": bund,
        })
    return table


def _empty_bund() -> dict:
    return {"compacted_m3": 0.0, "max_height_m": 0.0, "mean_height_m": 0.0, "length_m": 0.0}


def optimise_dugout(
    base_area_m2: float,
    base_depth_m: float,
    base_side_m: float,
    slope_factor: float,
    budget: float,
    costs: config.CostConfig | None = None,
    geometry: config.GeometryConfig | None = None,
    resolution_m: float = 30.0,
    area_for_side=None,
    max_side_m: float | None = None,
) -> dict:
    """
    The budget question for an excavated pond.

    A dug-out pond has no basin and no bund: it is a hole of a chosen size, so
    the only decisions are how much deeper and how much wider to dig.  Storage
    is area x depth x a side-slope factor, the spoil is the same volume and all
    of it is carted away, so the cost is excavation plus disposal.  The search
    is a grid over extra depth and new side, and the answer is the largest pond
    the money buys.

    `area_for_side` lets the caller supply the real plan area of a pond of a
    given side, clipped to the user's selection; without it the area scales
    with the square of the side.
    """
    costs = (costs or config.CostConfig()).resolved()
    geometry = geometry or config.GeometryConfig()

    base_capacity = base_area_m2 * base_depth_m * slope_factor
    if base_side_m <= 0 or base_area_m2 <= 0:
        return _empty_result(budget, base_capacity, "The pond has no footprint to enlarge")

    max_side = float(max_side_m or geometry.dugout_max_side_m)
    sides = np.round(np.arange(base_side_m, max(base_side_m, max_side) + 1e-9, 2.0), 1)
    depths = np.round(np.arange(0.0, geometry.max_dig_depth_m + 1e-9, 0.1), 2)

    def area_of(side: float) -> float:
        if area_for_side is not None:
            return float(area_for_side(float(side)))
        return base_area_m2 * (side / base_side_m) ** 2

    results = []
    for side in sides:
        area = area_of(side)
        if area <= 0:
            continue
        for d in depths:
            depth = base_depth_m + float(d)
            if depth > geometry.dugout_max_depth_m + 1e-9:
                continue
            capacity = area * depth * slope_factor
            gain = capacity - base_capacity
            if gain < -1e-6:
                continue
            cut = max(0.0, gain)           # the spoil is the water it makes room for
            n_cells = max(1, int(round(area / (resolution_m ** 2))))
            cost = _cost_of(cut, 0.0, 0.0, costs, resolution_m, n_cells)
            results.append((float(side), float(d), area, depth, gain, cut, cost))

    if not results:
        return _empty_result(budget, base_capacity, "No larger pond fits inside the selection")

    affordable = [r for r in results if r[6]["total"] <= budget]
    if affordable:
        # Most storage for the money.  Every cubic metre costs the same to dig
        # whether it goes down or out, so designs within a couple of percent of
        # the best are the same purchase; among those, take the smallest water
        # spread, because a deeper pond loses less to evaporation and takes
        # less land.
        top = max(r[4] for r in affordable)
        near = [r for r in affordable if r[4] >= 0.98 * top]
        best = min(near, key=lambda r: (r[2], r[6]["total"]))
    else:
        best = min(results, key=lambda r: r[6]["total"])
    side, d, area, depth, gain, cut, cost = best

    warnings = []
    steps = [r for r in results if r[4] > 0]
    if gain <= 0 and steps and budget > 0:
        cheapest = min(steps, key=lambda r: r[6]["total"])
        warnings.append(
            f"The smallest step, 10 cm deeper over the existing pond, costs Rs {cheapest[6]['total']:,.0f}, "
            "more than this budget"
        )
    at_max_depth = depth >= geometry.dugout_max_depth_m - 1e-6
    at_max_side = side >= max_side - 1e-6
    if at_max_depth and at_max_side and cost["total"] < budget * 0.9:
        warnings.append(
            f"The pond is at the largest size the planner allows here ({side:.0f} m a side, "
            f"{depth:.1f} m deep); Rs {budget - cost['total']:,.0f} of the budget is unspent"
        )
    if at_max_side and max_side < geometry.dugout_max_side_m - 1e-6 and cost["total"] < budget * 0.9:
        warnings.append("A wider pond would cross the edge of the selected area, so the design grows down rather than out")

    from_deepening = base_area_m2 * float(d) * slope_factor
    result = {
        "within_budget": cost["total"] <= budget,
        "budget": round(budget, 2),
        "design": {
            "designed": True,
            "extra_depth_m": round(d, 2),
            "spill_raise_m": 0.0,
            "new_usable_level_m": None,
            "excavation_m3": round(cut, 1),
            "bund_volume_m3": 0.0,
            "bund_fill_bank_m3": 0.0,
            "bund_max_height_m": 0.0,
            "bund_length_m": 0.0,
            "wetted_cells": max(1, int(round(area / (resolution_m ** 2)))),
            "base_side_m": round(base_side_m, 1),
            "new_side_m": round(side, 1),
            "base_depth_m": round(base_depth_m, 2),
            "new_depth_m": round(depth, 2),
            "new_area_m2": round(area, 1),
        },
        "storage": {
            "base_capacity_m3": round(base_capacity, 1),
            "added_m3": round(gain, 1),
            "new_capacity_m3": round(base_capacity + gain, 1),
            # A designed shape's uncertainty is the side slope, not the DEM
            "added_low_m3": round(gain * 0.85, 1),
            "added_high_m3": round(gain * 1.15, 1),
            "from_deepening_m3": round(min(gain, from_deepening), 1),
            "from_raising_m3": 0.0,
            "from_widening_m3": round(max(0.0, gain - from_deepening), 1),
        },
        "cost": cost,
        "cost_per_m3_stored": round(cost["total"] / gain, 2) if gain > 0 else None,
        "balanced_design": {},
        "warnings": warnings,
        "assumptions": {
            "excavation_per_m3": costs.excavation_per_m3,
            "fill_per_m3": costs.fill_per_m3,
            "haul_per_m3_per_50m": costs.haul_per_m3_per_50m,
            "borrow_per_m3": costs.borrow_per_m3,
            "manual_rates": costs.manual_rates,
            "side_slope_factor": slope_factor,
            "max_depth_m": geometry.dugout_max_depth_m,
            "max_side_m": max_side,
            "source": "CPWD DSR 2023 chapter 2; MGNREGA farm pond model",
        },
    }
    return result
