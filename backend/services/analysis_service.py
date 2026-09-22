"""
The analysis of a selected area, end to end.

Takes the polygon the user drew and returns everything the map and the sidebar
need: the catchment of the whole selection, ranked pond sites with their
footprint and their own catchment, the storage each can hold, the water each
can expect to collect in a year, and the stage-storage curve that the budget
filter later works on.

Results are cached on a key built from the rounded polygon and the parameters,
so redrawing what looks like the same area is free.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
from shapely.geometry import Point, shape

import config
from core.cache import bump, cached_call, get_cache, polygon_key
from core.resilience import DataQuality, Timer
from services import siting_service as sit
from services import hydrology_engine as he
from services.bundle_service import (
    cells_for, choose_resolution, get_bundle, window_for_selection,
)
from services.curve_number_service import catchment_curve_number
from services.evaporation_service import monthly_et0
from services.osm_service import constraints_geojson
from services.rainfall_service import get_daily_rainfall, rainfall_summary
from services.runoff_service import annual_inflow, daily_runoff, strange_runoff, water_balance
from services.storage_service import (
    basin_at, capacity_band, curve_for_payload, spill_level_from_fill,
    stage_storage_curve,
)
from utils.vector_utils import (
    mask_to_feature, pixel_to_lonlat, polygon_area_m2, rasterize_polygon,
    stream_features_to_geojson,
)

logger = logging.getLogger(__name__)


def _clip_to_selection(feature: dict | None, selection) -> dict | None:
    """
    Trim a drawn result to the area the user selected.

    Even after the raster work is confined, a 30 m cell whose centre sits just
    inside the boundary still sticks half a cell out when it is vectorised.
    That reads as the app ignoring the boundary, so the published geometry is
    intersected with it.
    """
    if feature is None or selection is None:
        return feature
    try:
        from shapely.geometry import mapping, shape

        geom = shape(feature["geometry"])
        if not geom.is_valid:
            geom = geom.buffer(0)
        clipped = geom.intersection(selection)
        if clipped.is_empty:
            return None
        return {**feature, "geometry": mapping(clipped)}
    except Exception as exc:  # noqa: BLE001 - never lose a result to a clip
        logger.debug("Could not clip a feature to the selection: %s", exc)
        return feature


class AnalysisError(ValueError):
    """A problem the user can fix, such as too large a selection."""


def analyze_polygon(
    geometry: dict,
    num_sites: int = 5,
    resolution_m: float | None = None,
    dem_source: str = "copernicus",
    hydrology: config.HydrologyParams | None = None,
    hsg: str = config.DEFAULT_HSG,
    cfg: config.AnalysisConfig | None = None,
    use_cache: bool = True,
) -> dict:
    """Analyse a GeoJSON polygon (WGS84) and return the full result payload."""
    cfg = cfg or config.AnalysisConfig()
    hydrology = hydrology or config.HydrologyParams()

    geom = shape(geometry)
    if geom.is_empty:
        raise AnalysisError("The selected area is empty")
    if not geom.is_valid:
        geom = geom.buffer(0)
        if geom.is_empty:
            raise AnalysisError("The selected polygon is self-intersecting; redraw it")

    area_m2 = polygon_area_m2(geometry)
    if area_m2 < cfg.min_area_ha * 10_000:
        raise AnalysisError(
            f"The selection is {area_m2 / 10_000:.2f} ha; the minimum is {cfg.min_area_ha:g} ha"
        )
    if area_m2 > cfg.max_area_km2 * 1e6:
        raise AnalysisError(
            f"The selection is {area_m2 / 1e6:.1f} km2; the maximum is {cfg.max_area_km2:g} km2. "
            "Draw a smaller area."
        )

    coords = list(geom.exterior.coords) if hasattr(geom, "exterior") else list(geom.bounds)
    key = polygon_key(
        [list(c[:2]) for c in coords],
        num_sites=num_sites,
        resolution=resolution_m,
        dem_source=dem_source,
        lam=hydrology.lambda_ia,
        cn=hydrology.cn_override,
        seepage=hydrology.seepage_mm_day,
        lined=hydrology.lined,
        black_soil=hydrology.black_soil,
        hsg=hsg,
    )

    def produce():
        return _analyze_uncached(
            geometry, geom, area_m2, num_sites, resolution_m, dem_source, hydrology, hsg, cfg
        )

    if not use_cache:
        return produce()
    return cached_call(key, produce, expire=config.CACHE_TTL_RESULT)


def _analyze_uncached(
    geometry, geom, area_m2, num_sites, resolution_m, dem_source, hydrology, hsg, cfg
) -> dict:
    timer = Timer()
    quality = DataQuality()

    # ── 1. Working window ───────────────────────────────────────────────────
    selection_bounds = geom.bounds                       # (w, s, e, n)
    # One shared window per grid cell, so the village warm-up actually helps
    bounds = window_for_selection(selection_bounds)
    requested = resolution_m or cfg.dem_resolution_m
    if area_m2 <= cfg.fine_resample_max_km2 * 1e6 and requested > cfg.fine_resolution_m:
        requested = min(requested, cfg.dem_resolution_m)
    working_res = choose_resolution(bounds, requested, cfg)

    data = get_bundle(bounds, working_res, dem_source, quality)
    bundle: he.HydroBundle = data["bundle"]
    constraints = data["constraints"]
    timer.mark("bundle_s")

    # ── 2. Catchment of the whole selection, growing the pad if truncated ──
    region = rasterize_polygon(geometry, bundle.shape, bundle.transform, bundle.epsg)
    if not region.any():
        raise AnalysisError("The selected area did not intersect the elevation grid")

    catchment_mask, info = he.catchment_of_region(bundle, region)
    # A catchment that runs off the edge of the window is only a lower bound, so
    # the window is widened and the routing redone.  Each attempt refetches the
    # elevation and re-queries OpenStreetMap, so the growth is deliberately
    # small and short: one extra ring, then two.  Growing by the old schedule
    # reached nine rings, about eighty kilometres across, and a large selection
    # simply ran out of time before finishing.
    # A catchment that leaves the window is a lower bound, and the window can
    # be widened to try to capture it.  That is worth at most one attempt:
    # each one roughly doubles the grid and repeats the elevation and
    # OpenStreetMap fetches, and three attempts took a large area past the
    # request budget with nothing to show.
    #
    # It is also only worth trying for a small selection.  An 800 ha area here
    # drains 19,700 ha, tens of kilometres upstream, so no window this service
    # would build could ever contain it; widening just spends the budget before
    # reporting the same lower bound.
    widen_worth_trying = (
        info.get("truncated")
        and area_m2 < 2e6                       # under about 200 ha
        and timer.total() < 15.0
    )
    if info.get("truncated") and not widen_worth_trying:
        logger.info(
            "Reporting a truncated catchment: %.0f ha selection, %.0f s spent already",
            area_m2 / 1e4, timer.total(),
        )
    elif widen_worth_trying:
        wider = window_for_selection(selection_bounds, rings=2)
        res2 = choose_resolution(wider, requested, cfg)
        if res2 <= working_res * 1.5 and cells_for(wider, res2) <= cfg.max_cells:
            try:
                data = get_bundle(wider, res2, dem_source, quality)
                bundle = data["bundle"]
                constraints = data["constraints"]
                working_res = res2
                region = rasterize_polygon(
                    geometry, bundle.shape, bundle.transform, bundle.epsg
                )
                catchment_mask, info = he.catchment_of_region(bundle, region)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not widen the window: %s", exc)

    quality.catchment_truncated = bool(info.get("truncated"))
    if quality.catchment_truncated:
        quality.note(
            "The catchment reaches the edge of the analysed window, so its area is a lower bound"
        )
    timer.mark("catchment_s")

    catchment_area_m2 = float(catchment_mask.sum()) * bundle.cell_area_m2
    centroid = geom.centroid
    lat, lon = float(centroid.y), float(centroid.x)

    # ── 3. Rainfall, curve number, evaporation ──────────────────────────────
    cn_info = catchment_curve_number(
        catchment_mask, bundle.transform, bundle.epsg,
        landuse_class=constraints.get("landuse_class"),
        hsg=hsg, override=hydrology.cn_override,
    )
    quality.cn_source = cn_info["source"]

    try:
        rainfall = get_daily_rainfall(lat, lon, quality=quality)
        rain_summary = rainfall_summary(rainfall)
        runoff = daily_runoff(rainfall, cn_info["cn"], hydrology)
        inflow = annual_inflow(runoff, catchment_area_m2, hydrology.dependability_pct)
        strange = strange_runoff(rainfall, catchment_area_m2)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rainfall stage failed: %s", exc)
        rainfall, rain_summary = None, {}
        runoff, inflow, strange = None, {}, {}
        quality.note(f"Water yield could not be computed: {exc}")

    evaporation = monthly_et0(lat, lon, quality)
    timer.mark("climate_s")

    # ── 4. Streams, criteria, suitability ───────────────────────────────────
    streams = he.stream_network(bundle, cfg.stream_threshold_ha)
    criteria = sit.build_criteria(bundle, constraints, streams, region, cn_info["cn"], cfg)
    score = sit.suitability_surface(criteria, constraints, region, bundle, cfg.weights, cfg)
    timer.mark("scoring_s")

    # ── 5. Candidates ───────────────────────────────────────────────────────
    num_sites = int(np.clip(num_sites, 1, cfg.max_candidates))
    candidates = sit.find_candidates(
        bundle, region, constraints, streams, score, cfg, limit=max(num_sites * 3, 10)
    )
    robustness = sit.weight_robustness(criteria, candidates, cfg.weights, top_k=num_sites)
    timer.mark("candidates_s")

    # Belt and braces: whatever the raster says, a site is only reported if its
    # published coordinate really falls inside the polygon the user drew.
    selection = geom

    sites = []
    for index, cand in enumerate(candidates):
        site = _build_site(
            cand, index, bundle, region, constraints, streams, cfg, hydrology,
            runoff, evaporation, robustness.get(index, 0.5), quality, selection,
        )
        if site is not None:
            point = Point(site["location"]["lon"], site["location"]["lat"])
            if not selection.covers(point):
                logger.info(
                    "Dropping site at (%.5f, %.5f): outside the selected area",
                    site["location"]["lat"], site["location"]["lon"],
                )
            else:
                sites.append(site)
        if len(sites) >= num_sites:
            break

    sites.sort(key=lambda s: -s["score"])
    for rank, site in enumerate(sites, start=1):
        site["rank"] = rank
    timer.mark("sites_s")

    # ── 6. Overlays ─────────────────────────────────────────────────────────
    catchment_feature = mask_to_feature(
        catchment_mask, bundle.transform, bundle.epsg,
        properties={
            "area_km2": round(catchment_area_m2 / 1e6, 4),
            "area_ha": round(catchment_area_m2 / 1e4, 2),
            "truncated": quality.catchment_truncated,
            "outlets": info.get("outlets", 0),
        },
        simplify_m=bundle.resolution_m,
    )
    streams_fc = stream_features_to_geojson(
        streams.get("features", []), bundle.epsg, clip_to=selection
    )
    timer.mark("overlays_s")

    analysis_id = uuid.uuid4().hex[:16]
    result = {
        "status": "success",
        "analysis_id": analysis_id,
        "selection": {
            "area_ha": round(area_m2 / 1e4, 2),
            "area_km2": round(area_m2 / 1e6, 4),
            "centroid": {"lat": round(lat, 6), "lon": round(lon, 6)},
            "geometry": geometry,
        },
        "grid": {
            "resolution_m": working_res,
            "rows": int(bundle.shape[0]),
            "cols": int(bundle.shape[1]),
            "cells": int(bundle.dem.size),
            "epsg": bundle.epsg,
            "dem_source": bundle.dem_source,
            "engine": bundle.engine,
        },
        "catchment": {
            "area_km2": round(catchment_area_m2 / 1e6, 4),
            "area_ha": round(catchment_area_m2 / 1e4, 2),
            "truncated": quality.catchment_truncated,
            "outlets": info.get("outlets", 0),
            "feature": catchment_feature,
        },
        "rainfall": rain_summary,
        "runoff": {
            "curve_number": cn_info["cn"],
            "cn_source": cn_info["source"],
            "hsg": cn_info["hsg"],
            "landuse_breakdown": cn_info.get("breakdown", {}),
            "lambda": hydrology.lambda_ia,
            "scs_cn": inflow,
            "strange_cross_check": strange,
        },
        "evaporation": evaporation,
        "sites": sites,
        "streams_geojson": streams_fc,
        "constraints_geojson": constraints_geojson(data["osm"]),
        "data_quality": quality.to_dict(),
        "assumptions": {
            "lambda_ia": hydrology.lambda_ia,
            "black_soil": hydrology.black_soil,
            "curve_number": cn_info["cn"],
            "seepage_mm_day": (
                hydrology.seepage_mm_day_lined if hydrology.lined else hydrology.seepage_mm_day
            ),
            "open_water_kc": hydrology.open_water_kc,
            "dependability_pct": hydrology.dependability_pct,
            "freeboard_m": cfg.freeboard_m,
            "dead_storage_m": cfg.dead_storage_m,
            "dem_vertical_uncertainty_m": cfg.dem_vertical_uncertainty_m,
            "weights": cfg.weights,
        },
        "timings": timer.to_dict(),
    }

    # Keep the raster context so the earthwork endpoint can work on the same grid
    _remember_context(analysis_id, bundle, region, constraints, sites, selection)
    bump("analyses")
    logger.info(
        "Analysis %s: %.2f ha selection, catchment %.2f km2, %d sites, %.2f s",
        analysis_id, area_m2 / 1e4, catchment_area_m2 / 1e6, len(sites), timer.total(),
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# One site
# ─────────────────────────────────────────────────────────────────────────────


def _build_site(
    cand, index, bundle, region, constraints, streams, cfg, hydrology,
    runoff, evaporation, robustness, quality, selection=None,
) -> dict | None:
    """Attach geometry, capacity, catchment and yield to a candidate."""
    row, col = cand.row, cand.col

    # Stage-storage curve: the embankment sweep already built one
    if cand.curve is None:
        basin = basin_at(bundle.dem, row, col, bundle.fill_depth)
        # For a natural depression the spill level is the flat top of the
        # filled surface, which is exact and costs nothing to read.
        spill = spill_level_from_fill(bundle.dem, bundle.fill_depth, basin)
        # Confine the water to the area the user drew.  A natural hollow does
        # not stop at their boundary, but a pond they can actually build does,
        # and reporting storage that lies outside the selection would not be
        # storage they can have.
        bounds = basin & region if basin is not None else region
        curve = stage_storage_curve(
            bundle.dem, (row, col), bundle.cell_area_m2,
            spill_level=spill if np.isfinite(spill) else None,
            bounds_mask=bounds, step_m=cfg.stage_step_m,
        )
    else:
        curve = cand.curve

    capacity = float(curve.get("capacity_m3", 0.0))
    # Drawn at the working water level, not brim-full
    footprint_mask = curve.get("surface_mask")
    if footprint_mask is None:
        footprint_mask = curve.get("mask")

    # A hollow this shallow is DEM noise, not a pond.  Rather than dropping the
    # location, offer it as an excavated pond: that is the honest answer on the
    # flat terrain where most villages sit, and the budget filter sizes it.
    kind = cand.kind
    surface_area = float(curve.get("surface_area_m2") or 0.0)
    mean_depth = capacity / surface_area if surface_area > 0 else 0.0

    # The same floors apply to every kind.  A bund holding a quarter of a metre
    # of water is not a check dam, and calling it one would flatter the result.
    too_small = capacity < cfg.min_useful_capacity_m3
    too_shallow = surface_area > 0 and mean_depth < cfg.min_mean_depth_m
    if too_small or too_shallow:
        if too_shallow and not too_small:
            logger.info(
                "Site at (%d, %d) averages %.2f m deep over %.2f ha: offering an "
                "excavated pond instead of the natural hollow",
                row, col, mean_depth, surface_area / 1e4,
            )
        kind = "dugout"
        capacity = 0.0

    if capacity <= 0 and kind != "dugout":
        return None
    if capacity > cfg.max_site_capacity_m3:
        # A basin this large is a reservoir, not a village pond: almost always
        # the flood fill escaping down an open valley rather than real storage.
        logger.info(
            "Dropping site at (%d, %d): capacity %.0f m3 exceeds the village-scale cap",
            row, col, capacity,
        )
        return None

    # A dug-out pond has no natural basin: size it to the MGNREGA model and let
    # the budget filter grow it.
    if kind == "dugout" and capacity <= 0:
        # The pond is a designed shape, so its capacity comes from the shape
        # that gets drawn, not from the raster cell that happens to hold it.
        # A 20 m pond is smaller than one 30 m cell, and sizing it by the cell
        # would overstate the storage by more than double.
        lon0, lat0 = pixel_to_lonlat(row, col, bundle.transform, bundle.epsg)
        dugout_feature = _clip_to_selection(
            _dugout_footprint(lon0, lat0, cfg, 0.0), selection
        )
        area = polygon_area_m2(dugout_feature["geometry"]) if dugout_feature else 0.0

        side_cells = max(1, int(round(cfg.dugout_side_m / bundle.resolution_m)))
        footprint_mask = np.zeros(bundle.shape, dtype=bool)
        r0 = max(0, row - side_cells // 2); r1 = min(bundle.shape[0], r0 + side_cells)
        c0 = max(0, col - side_cells // 2); c1 = min(bundle.shape[1], c0 + side_cells)
        footprint_mask[r0:r1, c0:c1] = True
        footprint_mask &= region
        if not footprint_mask.any():
            footprint_mask[row, col] = True
        if area <= 0:
            area = float(footprint_mask.sum()) * bundle.cell_area_m2

        depth = cfg.dugout_depth_m
        capacity = area * depth * cfg.dugout_side_slope_factor
        bed = float(bundle.dem[row, col]) - depth
        curve = {
            "level_m": [round(bed, 2), round(bed + depth, 2)],
            "area_m2": [0.0, round(area, 1)],
            "volume_m3": [0.0, round(capacity, 1)],
            "bed_level_m": round(bed, 2),
            "spill_level_m": round(bed + depth, 2),
            "usable_level_m": round(bed + depth - cfg.freeboard_m, 2),
            "capacity_m3": round(capacity, 1),
            "dead_storage_m3": round(area * cfg.dead_storage_m * 0.6, 1),
            "live_storage_m3": round(capacity - area * cfg.dead_storage_m * 0.6, 1),
            "surface_area_m2": round(area, 1),
            "max_depth_m": depth,
            "mask": footprint_mask,
            "note": "Excavated pond sized to the MGNREGA model; the terrain offers no usable basin",
        }

    lon, lat = pixel_to_lonlat(row, col, bundle.transform, bundle.epsg)

    # Catchment of this site
    site_mask, (srow, scol) = he.catchment_of_point(bundle, row, col)
    site_catchment_m2 = float(site_mask.sum()) * bundle.cell_area_m2

    # Yield for this site's own catchment
    inflow, balance = {}, {}
    if runoff is not None and site_catchment_m2 > 0:
        inflow = annual_inflow(runoff, site_catchment_m2, hydrology.dependability_pct)
        balance = water_balance(
            runoff, site_catchment_m2, curve, capacity, evaporation, hydrology
        )

    slope_pct = float(bundle.slope_pct[row, col])
    catchment_ha = site_catchment_m2 / 1e4
    depth_m = float(curve.get("max_depth_m", 0.0))

    structure = sit.structure_type(catchment_ha, slope_pct, depth_m, kind)
    band = capacity_band(capacity, curve)

    # Confidence: data quality x weight robustness x depth against DEM noise
    depth_factor = float(np.clip(depth_m / max(0.5, cfg.dem_vertical_uncertainty_m), 0.2, 1.0))
    confidence_value = quality.score() * max(0.2, robustness) * depth_factor
    confidence_value = float(np.clip(confidence_value, 0.0, 1.0))

    footprint_feature = mask_to_feature(
        footprint_mask, bundle.transform, bundle.epsg,
        properties={"kind": "pond_footprint", "capacity_m3": round(capacity, 1)},
        simplify_m=bundle.resolution_m / 2,
    ) if footprint_mask is not None else None

    # An excavated pond is a designed shape, not a found one, so it is drawn at
    # the size it would actually be built rather than as the 30 m cell that
    # happens to contain it.
    if kind == "dugout":
        footprint_feature = _clip_to_selection(
            _dugout_footprint(lon, lat, cfg, capacity), selection
        )
    else:
        footprint_feature = _clip_to_selection(footprint_feature, selection)

    catchment_feature = mask_to_feature(
        site_mask, bundle.transform, bundle.epsg,
        properties={"area_ha": round(catchment_ha, 2)},
        simplify_m=bundle.resolution_m,
    )

    return {
        "site_id": f"s{index}",
        "rank": index + 1,
        "kind": kind,
        "structure": structure,
        "score": round(float(cand.score), 4),
        "confidence": round(confidence_value, 3),
        "confidence_label": sit.confidence_label(confidence_value),
        "weight_robustness": round(float(robustness), 3),
        "location": {"lat": round(lat, 6), "lon": round(lon, 6)},
        "cell": {"row": int(row), "col": int(col)},
        "elevation_m": round(float(bundle.dem[row, col]), 2),
        "slope_pct": round(slope_pct, 2),
        "twi": round(float(bundle.twi[row, col]), 2),
        "storage": {
            "capacity_m3": round(capacity, 1),
            "capacity_low_m3": band["low_m3"],
            "capacity_high_m3": band["high_m3"],
            "dead_storage_m3": curve.get("dead_storage_m3", 0.0),
            "live_storage_m3": curve.get("live_storage_m3", capacity),
            "surface_area_m2": curve.get("surface_area_m2", 0.0),
            "max_depth_m": depth_m,
            "spill_level_m": curve.get("spill_level_m"),
            "note": curve.get("note"),
        },
        "catchment": {
            "area_ha": round(catchment_ha, 2),
            "area_km2": round(site_catchment_m2 / 1e6, 4),
            "feature": catchment_feature,
        },
        "yield": {
            "annual_inflow_mean_m3": inflow.get("mean_m3", 0.0),
            "annual_inflow_dependable_m3": inflow.get("dependable_m3", 0.0),
            "harvestable_mean_m3": balance.get("harvestable_mean_m3", 0.0),
            "harvestable_dependable_m3": balance.get("harvestable_dependable_m3", 0.0),
            "mean_wet_months": balance.get("mean_wet_months", 0.0),
            "monthly_storage_m3": balance.get("monthly_storage_m3", []),
            "month_labels": balance.get("month_labels", []),
            "years": inflow.get("years", 0),
        },
        "stage_storage": curve_for_payload(curve),
        "bund": cand.bund,
        "footprint": footprint_feature,
        "criteria": cand.criteria,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Context reuse for the earthwork endpoint
# ─────────────────────────────────────────────────────────────────────────────


def _remember_context(analysis_id: str, bundle, region, constraints, sites, selection=None) -> None:
    """
    Keep just enough raster context for the budget filter to run on the same
    grid without recomputing hydrology.
    """
    payload = {
        "dem": bundle.dem,
        "transform": tuple(bundle.transform)[:6],
        "epsg": bundle.epsg,
        "resolution_m": bundle.resolution_m,
        "fill_depth": bundle.fill_depth,
        # The budget filter has to respect the same boundary the analysis did
        "region": region,
        # An excavated pond is re-drawn at its new size, so its shape has to be
        # clipped to the same boundary the analysis used
        "selection_wkt": selection.wkt if selection is not None else None,
        "sites": {
            s["site_id"]: {
                "cell": s["cell"],
                "kind": s.get("kind"),
                "location": s["location"],
                "capacity_m3": s["storage"]["capacity_m3"],
                "surface_area_m2": s["storage"].get("surface_area_m2", 0.0),
                "max_depth_m": s["storage"].get("max_depth_m", 0.0),
                "stage_storage": s["stage_storage"],
                "spill_level_m": s["storage"].get("spill_level_m"),
            }
            for s in sites
        },
    }
    get_cache().set(f"ctx:{config.ALGO_VERSION}:{analysis_id}", payload, expire=config.CACHE_TTL_RESULT)


def load_context(analysis_id: str) -> dict | None:
    return get_cache().get(f"ctx:{config.ALGO_VERSION}:{analysis_id}", default=None)


def _dugout_footprint(lon: float, lat: float, cfg, capacity_m3: float, side_m: float | None = None) -> dict:
    """
    The plan shape of an excavated pond, at its designed size.

    A 20 m pond cannot be represented by a 30 m raster cell, and drawing the
    cell makes the pond look both bigger and squarer than it is.  The corners
    are cut so it reads as a pond rather than a pixel.
    """
    import math

    side_m = float(side_m or cfg.dugout_side_m)
    half = side_m / 2.0
    dlat = half / 111_320.0
    dlon = half / (111_320.0 * max(0.2, math.cos(math.radians(lat))))

    # An octagon: a square with the corners taken off
    k = 0.42
    ring = [
        (lon - dlon * k, lat - dlat), (lon + dlon * k, lat - dlat),
        (lon + dlon, lat - dlat * k), (lon + dlon, lat + dlat * k),
        (lon + dlon * k, lat + dlat), (lon - dlon * k, lat + dlat),
        (lon - dlon, lat + dlat * k), (lon - dlon, lat - dlat * k),
    ]
    ring.append(ring[0])
    return {
        "type": "Feature",
        "properties": {
            "kind": "pond_footprint",
            "capacity_m3": round(capacity_m3, 1),
            "designed": True,
            "side_m": side_m,
        },
        "geometry": {"type": "Polygon", "coordinates": [[[x, y] for x, y in ring]]},
    }
