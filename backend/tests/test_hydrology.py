"""
Tests for the terrain and hydrology core.

These use synthetic terrain with a known answer rather than real elevation
data, so they run offline and fail loudly when the maths drifts.
"""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from services import hydrology_engine as he
from services.storage_service import (
    flood_fill_at_level, spill_level_from_fill, stage_storage_curve,
)

RES = 10.0
TRANSFORM = Affine(RES, 0.0, 500000.0, 0.0, -RES, 2350000.0)
EPSG = 32644


def tilted_plane_with_pit(n: int = 60, pit_depth: float = 3.0) -> np.ndarray:
    """A plane sloping west, with one square hollow of a known volume."""
    y, x = np.mgrid[0:n, 0:n]
    dem = (100.0 - 0.02 * x * RES).astype(np.float32)
    dem[28:34, 28:34] -= pit_depth        # 6 x 6 cells, 3 m deep
    return dem


def bowl(n: int = 60, depth: float = 4.0) -> np.ndarray:
    """A radially symmetric bowl inside a flat plain."""
    y, x = np.mgrid[0:n, 0:n]
    r = np.hypot(x - n / 2, y - n / 2)
    dem = np.full((n, n), 100.0, dtype=np.float32)
    inside = r < 12
    dem[inside] = 100.0 - depth * (1 - r[inside] / 12.0)
    return dem.astype(np.float32)


# ── Fill and routing ─────────────────────────────────────────────────────────


def test_fill_raises_only_the_pit():
    dem = tilted_plane_with_pit()
    bundle = he.build_bundle(dem, TRANSFORM, EPSG, RES, "synthetic")

    assert bundle.fill_depth.max() == pytest.approx(3.0, abs=0.3)
    # Nothing outside the hollow should have been raised
    outside = bundle.fill_depth.copy()
    outside[26:36, 26:36] = 0
    assert outside.max() < 0.05
    # Filling never lowers the ground
    assert np.all(bundle.filled >= bundle.dem - 1e-4)


def test_flow_accumulates_downslope():
    dem = tilted_plane_with_pit(pit_depth=0.0)
    bundle = he.build_bundle(dem, TRANSFORM, EPSG, RES, "synthetic")

    # Elevation falls as the column index rises, so water runs east and the
    # eastern edge carries everything that fell upslope of it.
    west = bundle.upstream_area[:, 1].mean()
    east = bundle.upstream_area[:, -2].mean()
    assert east > west * 5
    # Every cell drains at least itself
    assert bundle.upstream_area.min() >= RES * RES - 1


def test_slope_matches_the_plane():
    dem = tilted_plane_with_pit(pit_depth=0.0)
    bundle = he.build_bundle(dem, TRANSFORM, EPSG, RES, "synthetic")
    interior = bundle.slope_pct[5:-5, 5:-5]
    # 0.02 m fall per metre = 2 percent
    assert interior.mean() == pytest.approx(2.0, abs=0.3)


def test_catchment_of_a_region_covers_upslope_ground():
    dem = tilted_plane_with_pit(pit_depth=0.0)
    bundle = he.build_bundle(dem, TRANSFORM, EPSG, RES, "synthetic")

    region = np.zeros(dem.shape, dtype=bool)
    region[20:40, 45:55] = True              # a strip near the downslope edge
    mask, info = he.catchment_of_region(bundle, region)

    assert mask.sum() > region.sum()
    assert np.all(mask[region])              # the selection is part of its own catchment
    # The catchment reaches upslope (west), not downslope (east)
    cols = np.where(mask.any(axis=0))[0]
    assert cols.min() < 45


def test_bundle_survives_a_cache_round_trip():
    dem = tilted_plane_with_pit()
    bundle = he.build_bundle(dem, TRANSFORM, EPSG, RES, "synthetic")
    restored = he.HydroBundle.from_payload(bundle.to_payload())

    assert restored.shape == bundle.shape
    assert np.allclose(restored.dem, bundle.dem)
    assert np.allclose(restored.upstream_area, bundle.upstream_area)
    assert restored.flw() is not None        # the flow object rebuilds from D8


# ── Stage, area and storage ──────────────────────────────────────────────────


def test_storage_of_a_known_pit():
    dem = tilted_plane_with_pit(pit_depth=3.0)
    bundle = he.build_bundle(dem, TRANSFORM, EPSG, RES, "synthetic")

    basin = bundle.fill_depth > 0.01
    spill = spill_level_from_fill(bundle.dem, bundle.fill_depth, basin)
    seed = np.unravel_index(int(np.argmin(np.where(basin, dem, np.inf))), dem.shape)

    curve = stage_storage_curve(
        dem, seed, RES * RES, spill_level=spill, bounds_mask=basin, step_m=0.1,
    )

    # The hollow is 6x6 cells of 100 m2; the plane inside it falls ~0.2 m across,
    # so the volume at the spill level is close to 36 * 100 * (3 - 0.1) m3.
    assert 8000 < curve["volume_m3"][-1] < 12000
    # Storage must never decrease as the level rises
    assert np.all(np.diff(curve["volume_m3"]) >= -1e-6)
    assert np.all(np.diff(curve["area_m2"]) >= -1e-6)
    # Freeboard is taken off the usable capacity
    assert curve["capacity_m3"] < curve["volume_m3"][-1]


def test_curve_gradient_is_the_water_spread():
    """dV/dS must equal A(S): the optimiser prices a raise with it."""
    dem = bowl(depth=4.0)
    seed = (30, 30)
    curve = stage_storage_curve(dem, seed, RES * RES, spill_level=100.0, step_m=0.2)

    levels = np.array(curve["level_m"])
    volumes = np.array(curve["volume_m3"])
    areas = np.array(curve["area_m2"])

    gradient = np.diff(volumes) / np.diff(levels)
    midpoint_area = (areas[:-1] + areas[1:]) / 2
    usable = midpoint_area > 0
    # Within a cell-quantisation tolerance
    assert np.allclose(gradient[usable], midpoint_area[usable], rtol=0.35)


def test_flood_fill_respects_containment():
    dem = bowl(depth=4.0)
    window = np.zeros(dem.shape, dtype=bool)
    window[25:35, 25:35] = True

    unbounded = flood_fill_at_level(dem, (30, 30)[0], (30, 30)[1], 99.9)
    bounded = flood_fill_at_level(dem, 30, 30, 99.9, bounds_mask=window)

    assert bounded.sum() <= unbounded.sum()
    assert not (bounded & ~window).any()


def test_spill_level_is_read_off_the_filled_surface():
    dem = tilted_plane_with_pit(pit_depth=2.0)
    bundle = he.build_bundle(dem, TRANSFORM, EPSG, RES, "synthetic")
    basin = bundle.fill_depth > 0.01

    spill = spill_level_from_fill(bundle.dem, bundle.fill_depth, basin)
    # The spill sits above the deepest point and at or below the surrounding ground
    assert spill > dem[basin].min()
    assert spill <= float(dem[28:34, 28:34].max()) + 2.1
