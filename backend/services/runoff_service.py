"""
Runoff and the pond water balance.

Runoff uses the Indian form of the SCS Curve Number method as written in the
CGWB Manual on Artificial Recharge, section 4.2.2, run on the *daily* rainfall
series.  Daily is not a detail: the equation is convex, so feeding it monthly or
annual totals inflates runoff several times over.

Three numbers come out of here:
  * annual inflow, mean and 75 percent dependable over the record;
  * harvestable volume, what a pond of a given capacity actually captures after
    evaporation, seepage and spill, year by year;
  * a Strange-table cross-check, because the accepted Indian methods disagree by
    up to a factor of two and a single number would be false precision.
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np

import config
from services.curve_number_service import convert_cn
from services.rainfall_service import weibull_dependable

logger = logging.getLogger(__name__)

# Strange's monsoon runoff percentage for an average catchment (CGWB Table 4.8a),
# as (monsoon rainfall mm, percent of rainfall appearing as runoff).
STRANGE_AVERAGE = [
    (0, 0.0), (286, 11.25), (500, 19.7), (714, 28.1),
    (762, 30.6), (935, 36.8), (1143, 45.0), (1524, 55.0),
]


def daily_runoff(
    rainfall: dict,
    cn2: float,
    params: config.HydrologyParams | None = None,
) -> dict:
    """
    Daily runoff depth (mm) from daily rainfall by the SCS-CN method.

    The antecedent moisture condition is set from the previous five days' rain
    (CGWB Table 4.13), with different thresholds inside and outside the growing
    season, and the curve number is converted per day accordingly.
    """
    params = params or config.HydrologyParams()
    dates = rainfall["dates"]
    rain = np.asarray(rainfall["values"], dtype=np.float64)
    n = len(rain)
    if n == 0:
        return {"dates": dates, "runoff_mm": np.zeros(0), "cn2": cn2,
                "water_years": np.zeros(0, dtype=int),
                "monthly_mm": np.zeros((0, 12))}

    # Five-day antecedent rainfall, excluding the current day
    kernel = np.ones(5)
    padded = np.concatenate([np.zeros(5), rain])
    p5 = np.convolve(padded, kernel, mode="valid")[:n]

    months = rainfall["month"]
    water_years = rainfall["water_year"]
    growing = np.isin(months, params.growing_months)

    amc = np.full(n, 2, dtype=np.int8)
    low = np.where(growing, params.amc_growing_low, params.amc_dormant_low)
    high = np.where(growing, params.amc_growing_high, params.amc_dormant_high)
    amc[p5 < low] = 1
    amc[p5 > high] = 3

    cn = np.empty(n, dtype=np.float64)
    for level in (1, 2, 3):
        sel = amc == level
        if sel.any():
            cn[sel] = convert_cn(cn2, level)

    s = 25400.0 / np.clip(cn, 1.0, 100.0) - 254.0

    # Lambda: 0.3 in India, 0.1 for black soils in the wetter conditions
    lam = np.full(n, params.lambda_ia, dtype=np.float64)
    if params.black_soil:
        lam[np.isin(amc, (2, 3))] = 0.1

    ia = lam * s
    excess = rain - ia
    runoff = np.where(excess > 0, excess ** 2 / (excess + s), 0.0)

    # Aggregate once to (hydrological year, month) so that the per-site water
    # balance is a table lookup rather than another pass over 12,000 days.
    order = [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5]
    unique_years = np.unique(water_years)
    full = np.array([
        y for y in unique_years if (water_years == y).sum() >= 300
    ], dtype=int)
    monthly = np.zeros((len(full), 12), dtype=np.float64)
    days = np.zeros((len(full), 12), dtype=np.int32)
    for row, y in enumerate(full):
        in_year = water_years == y
        for col, m in enumerate(order):
            sel = in_year & (months == m)
            if sel.any():
                monthly[row, col] = runoff[sel].sum()
                days[row, col] = int(sel.sum())

    return {
        "dates": dates,
        "runoff_mm": runoff.astype(np.float32),
        "rain_mm": rain.astype(np.float32),
        "cn2": float(cn2),
        "lambda": float(params.lambda_ia),
        "amc_counts": {int(k): int((amc == k).sum()) for k in (1, 2, 3)},
        "water_years": full,
        "monthly_mm": monthly,          # rows = years, cols = Jun..May
        "monthly_days": days,
        "month_order": order,
    }


def water_year(date: dt.date) -> int:
    """Hydrological year starting in June, labelled by its first calendar year."""
    return date.year if date.month >= 6 else date.year - 1


def annual_inflow(
    runoff: dict,
    catchment_area_m2: float,
    dependability_pct: float = 75.0,
) -> dict:
    """Inflow volume per hydrological year, with mean and dependable values."""
    monthly = np.asarray(runoff.get("monthly_mm"))
    years_arr = np.asarray(runoff.get("water_years", []), dtype=int)
    if monthly.size == 0 or catchment_area_m2 <= 0:
        return {"mean_m3": 0.0, "dependable_m3": 0.0, "per_year": {}, "years": 0}

    annual_mm = monthly.sum(axis=1)
    volumes = {
        int(y): float(mm * catchment_area_m2 / 1000.0)
        for y, mm in zip(years_arr, annual_mm)
    }
    series = np.array(list(volumes.values()))
    if series.size == 0:
        return {"mean_m3": 0.0, "dependable_m3": 0.0, "per_year": {}, "years": 0}

    return {
        "mean_m3": round(float(series.mean()), 1),
        "dependable_m3": round(float(weibull_dependable(series, dependability_pct)), 1),
        "min_m3": round(float(series.min()), 1),
        "max_m3": round(float(series.max()), 1),
        "per_year": {k: round(v, 1) for k, v in volumes.items()},
        "years": int(series.size),
        "dependability_pct": dependability_pct,
    }


def _partial_edges(dates, years, unique) -> bool:
    first = (years == unique[0]).sum()
    last = (years == unique[-1]).sum()
    return first < 300 or last < 300


def strange_runoff(rainfall: dict, catchment_area_m2: float) -> dict:
    """
    Cross-check with Strange's monsoon table (CGWB Table 4.8a, average catchment).

    Reported beside the curve-number figure as the upper end of a range.
    """
    values = np.asarray(rainfall["values"], dtype=np.float64)
    if len(values) == 0:
        return {"mean_m3": 0.0, "percent": 0.0}

    months = rainfall["month"]
    years = rainfall["water_year"]
    monsoon = np.isin(months, (6, 7, 8, 9, 10))

    volumes = []
    for y in np.unique(years):
        rain_mm = float(values[(years == y) & monsoon].sum())
        if rain_mm <= 0:
            continue
        pct = float(np.interp(
            rain_mm,
            [p[0] for p in STRANGE_AVERAGE],
            [p[1] for p in STRANGE_AVERAGE],
        ))
        volumes.append(rain_mm * pct / 100.0 * catchment_area_m2 / 1000.0)

    if not volumes:
        return {"mean_m3": 0.0, "percent": 0.0}
    arr = np.array(volumes)
    return {
        "mean_m3": round(float(arr.mean()), 1),
        "dependable_m3": round(float(weibull_dependable(arr, 75.0)), 1),
        "percent": round(float(np.mean([v / max(1e-9, arr.max()) for v in arr]) * 100), 1),
        "method": "Strange monsoon table (average catchment)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Monthly water balance
# ─────────────────────────────────────────────────────────────────────────────


def water_balance(
    runoff: dict,
    catchment_area_m2: float,
    stage_curve: dict,
    capacity_m3: float,
    evaporation: dict,
    params: config.HydrologyParams | None = None,
) -> dict:
    """
    Step a pond of ``capacity_m3`` through every hydrological year of the record.

    Storage gains the month's inflow and loses evaporation and seepage over the
    *current* water-spread area, which is read back from the stage-storage
    curve, so a nearly empty pond loses far less than a full one.  Anything
    above capacity spills.

    Returns the harvested volume (what the pond actually captured) as a mean and
    a dependable value, plus a representative monthly trace for the chart.
    """
    params = params or config.HydrologyParams()
    monthly = np.asarray(runoff.get("monthly_mm"))
    year_labels = np.asarray(runoff.get("water_years", []), dtype=int)
    if monthly.size == 0 or capacity_m3 <= 0:
        return _empty_balance()

    seepage_mm_day = params.seepage_mm_day_lined if params.lined else params.seepage_mm_day
    evap_month = evaporation.get("open_water_mm_month", [0.0] * 12)
    order = runoff.get("month_order", [6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5])
    days_per_month = np.asarray(
        runoff.get("monthly_days", np.full(monthly.shape, 30)), dtype=float
    )

    captured_per_year, spill_per_year, loss_per_year = {}, {}, {}
    wet_months_per_year, end_storage_per_year = {}, {}
    trace_sum = np.zeros(12)
    trace_n = 0

    for row, y in enumerate(year_labels):
        storage = 0.0
        captured = spilled = lost = 0.0
        wet_months = 0
        monthly_storage = []

        for col, m in enumerate(order):
            inflow = float(monthly[row, col] * catchment_area_m2 / 1000.0)
            ndays = float(days_per_month[row, col]) or 30.0

            # Losses act on the water spread of the *current* storage, so a
            # nearly empty pond loses far less than a full one.
            area = area_at_volume(stage_curve, storage)
            loss_mm = evap_month[m - 1] + seepage_mm_day * ndays
            losses = min(storage + inflow, loss_mm / 1000.0 * area)

            gross = storage + inflow - losses
            spill = max(0.0, gross - capacity_m3)
            storage = max(0.0, min(gross, capacity_m3))

            captured += max(0.0, inflow - spill)
            spilled += spill
            lost += losses
            if storage > 0.05 * capacity_m3:
                wet_months += 1
            monthly_storage.append(storage)

        captured_per_year[int(y)] = round(captured, 1)
        spill_per_year[int(y)] = round(spilled, 1)
        loss_per_year[int(y)] = round(lost, 1)
        wet_months_per_year[int(y)] = wet_months
        end_storage_per_year[int(y)] = round(storage, 1)
        trace_sum += np.array(monthly_storage[:12])
        trace_n += 1

    if not captured_per_year:
        return _empty_balance()

    captured_arr = np.array(list(captured_per_year.values()))
    spill_arr = np.array(list(spill_per_year.values()))
    loss_arr = np.array(list(loss_per_year.values()))
    wet = np.array(list(wet_months_per_year.values()))
    trace = (trace_sum / max(1, trace_n)).round(1).tolist()

    return {
        # What the pond actually takes in over a year: inflow that did not spill.
        # A pond smaller than its catchment fills and spills repeatedly, so this
        # can exceed the capacity several times over.
        "harvestable_mean_m3": round(float(captured_arr.mean()), 1),
        "harvestable_dependable_m3": round(
            float(weibull_dependable(captured_arr, params.dependability_pct)), 1
        ),
        "harvestable_min_m3": round(float(captured_arr.min()), 1),
        "harvestable_max_m3": round(float(captured_arr.max()), 1),
        "spill_mean_m3": round(float(spill_arr.mean()), 1),
        "losses_mean_m3": round(float(loss_arr.mean()), 1),
        "mean_wet_months": round(float(wet.mean()), 1),
        "monthly_storage_m3": trace,
        "month_labels": ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                          "Dec", "Jan", "Feb", "Mar", "Apr", "May"],
        "per_year": captured_per_year,
        "years": int(captured_arr.size),
        "seepage_mm_day": seepage_mm_day,
        "evaporation_annual_mm": evaporation.get("annual_mm", 0),
    }


def _empty_balance() -> dict:
    return {
        "harvestable_mean_m3": 0.0,
        "harvestable_dependable_m3": 0.0,
        "mean_wet_months": 0.0,
        "monthly_storage_m3": [0.0] * 12,
        "month_labels": ["Jun", "Jul", "Aug", "Sep", "Oct", "Nov",
                          "Dec", "Jan", "Feb", "Mar", "Apr", "May"],
        "per_year": {},
        "years": 0,
    }


def area_at_volume(stage_curve: dict, volume_m3: float) -> float:
    """Water-spread area for a stored volume, by interpolation on the curve."""
    volumes = stage_curve.get("volume_m3") or []
    areas = stage_curve.get("area_m2") or []
    if not volumes or not areas:
        return 0.0
    return float(np.interp(volume_m3, volumes, areas))


def _days_in_month(year: int, month: int) -> int:
    import calendar

    return calendar.monthrange(year, month)[1]
