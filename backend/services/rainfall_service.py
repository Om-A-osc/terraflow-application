"""
Daily rainfall for a point.

Primary source is the IMD 0.25 degree gauge-interpolated daily grid, downloaded
once by ``scripts/download_imd.py`` and read here from a local memmap, so a
lookup is an array slice with no network call.  When that archive is not
present the NASA POWER daily point API supplies the series instead; POWER is
reanalysis, so it smooths daily intensity and biases curve-number runoff low.
That difference is recorded in data quality and lowers the site confidence.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np

import config
from core.cache import bump, cached_call, make_key
from core.resilience import DataQuality, call_source, get

logger = logging.getLogger(__name__)

_memmap = None
_meta: dict | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Local IMD archive
# ─────────────────────────────────────────────────────────────────────────────


def _load_archive():
    """Open the IMD memmap once per process; returns (array, meta) or (None, None)."""
    global _memmap, _meta
    if _memmap is not None:
        return _memmap, _meta
    if not (config.RAINFALL_MEMMAP.exists() and config.RAINFALL_META.exists()):
        return None, None
    try:
        meta = json.loads(config.RAINFALL_META.read_text())
        shape = tuple(meta["shape"])              # (days, nlat, nlon)
        arr = np.memmap(config.RAINFALL_MEMMAP, dtype=meta.get("dtype", "int16"), mode="r", shape=shape)
        _memmap, _meta = arr, meta
        logger.info("IMD rainfall archive: %s days from %s", shape[0], meta["start_date"])
        return _memmap, _meta
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not open the IMD archive: %s", exc)
        return None, None


def has_local_archive() -> bool:
    arr, _ = _load_archive()
    return arr is not None


def _imd_cell(lat: float, lon: float) -> tuple:
    i = int(round((lat - config.IMD_LAT0) / config.IMD_STEP))
    j = int(round((lon - config.IMD_LON0) / config.IMD_STEP))
    i = max(0, min(config.IMD_NLAT - 1, i))
    j = max(0, min(config.IMD_NLON - 1, j))
    return i, j


def _read_imd(lat: float, lon: float) -> dict | None:
    arr, meta = _load_archive()
    if arr is None:
        return None
    i, j = _imd_cell(lat, lon)
    series = np.asarray(arr[:, i, j], dtype=np.float32)
    scale = float(meta.get("scale", 10.0))
    series = series / scale
    series[series < 0] = np.nan

    # Nearest valid cell if this one is sea / outside India
    if np.all(np.isnan(series)):
        for radius in (1, 2, 3):
            found = False
            for di in range(-radius, radius + 1):
                for dj in range(-radius, radius + 1):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < config.IMD_NLAT and 0 <= jj < config.IMD_NLON:
                        cand = np.asarray(arr[:, ii, jj], dtype=np.float32) / scale
                        cand[cand < 0] = np.nan
                        if not np.all(np.isnan(cand)):
                            series, found = cand, True
                            break
                if found:
                    break
            if found:
                break
    if np.all(np.isnan(series)):
        return None

    start = np.datetime64(meta["start_date"], "D")
    dates = start + np.arange(len(series), dtype="timedelta64[D]")
    bump("rainfall_local")
    return _with_calendar({
        "dates": dates,
        "values": np.nan_to_num(series, nan=0.0),
        "source": "imd_0.25deg",
        "label": "IMD 0.25 degree gridded daily rainfall",
    })


# ─────────────────────────────────────────────────────────────────────────────
# NASA POWER fallback
# ─────────────────────────────────────────────────────────────────────────────


def _read_power(lat: float, lon: float, start_year: int, end_year: int) -> dict | None:
    end_year = min(end_year, dt.date.today().year - 1)
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start": f"{start_year}0101",
        "end": f"{end_year}1231",
        "format": "JSON",
    }

    def _call():
        resp = get(config.NASA_POWER_DAILY, params=params, timeout=(3.05, 60.0))
        return resp.json()

    data = call_source("nasa_power_daily", _call)
    if not data:
        return None
    try:
        series = data["properties"]["parameter"]["PRECTOTCORR"]
    except (KeyError, TypeError):
        logger.warning("Unexpected NASA POWER payload")
        return None

    keys = sorted(series)
    if not keys:
        return None
    dates = np.array([f"{k[:4]}-{k[4:6]}-{k[6:8]}" for k in keys], dtype="datetime64[D]")
    values = np.array([series[k] for k in keys], dtype=np.float64)
    values[values <= -900] = 0.0

    bump("rainfall_power")
    return _with_calendar({
        "dates": dates,
        "values": values.astype(np.float32),
        "source": "nasa_power",
        "label": "NASA POWER daily precipitation (MERRA-2 reanalysis)",
    })


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def get_daily_rainfall(
    lat: float,
    lon: float,
    start_year: int | None = None,
    end_year: int | None = None,
    quality: DataQuality | None = None,
) -> dict:
    """
    Daily rainfall series for the grid cell containing (lat, lon).

    Returns ``{dates, values (mm), source, label, years}``.  Raises only if
    every source fails, which also fails the analysis loudly rather than
    inventing rainfall.
    """
    quality = quality or DataQuality()
    params = config.HydrologyParams()
    start_year = start_year or params.record_start_year
    end_year = end_year or params.record_end_year

    local = _read_imd(lat, lon)
    if local is not None:
        result = _clip_years(local, start_year, end_year)
        quality.rainfall_source = result["source"]
        return result

    # NASA POWER is a 0.5 degree grid, so 0.1 degree keys are ample and let every
    # polygon in a village share one entry.
    key = make_key("rain_power", round(lat, 1), round(lon, 1), start_year, end_year)
    payload = cached_call(
        key,
        lambda: _serialise(_read_power(lat, lon, start_year, end_year)),
        expire=config.CACHE_TTL_CLIMATE,
    )
    remote = _deserialise(payload)
    if remote is None:
        quality.rainfall_source = "none"
        quality.note("No rainfall source was reachable; yield could not be computed")
        raise RuntimeError("No rainfall source is available for this location")

    quality.rainfall_source = remote["source"]
    quality.note(
        "Rainfall from NASA POWER reanalysis, not the IMD gauge grid; "
        "daily intensities are smoothed and runoff is biased low"
    )
    return remote


def _with_calendar(series: dict) -> dict:
    """
    Attach year, month and hydrological-year arrays.

    Every downstream calculation needs these, and deriving them per call with a
    Python loop over ~12,800 dates was the single slowest step in an analysis.
    NumPy does the same work on the whole array at once.
    """
    dates = np.asarray(series["dates"], dtype="datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    months = dates.astype("datetime64[M]").astype(int) % 12 + 1
    return {
        **series,
        "dates": dates,
        "year": years.astype(np.int32),
        "month": months.astype(np.int16),
        # Hydrological year runs June to May, labelled by its first calendar year
        "water_year": np.where(months >= 6, years, years - 1).astype(np.int32),
    }


def _clip_years(series: dict, start_year: int, end_year: int) -> dict:
    series = series if "year" in series else _with_calendar(series)
    years = series["year"]
    keep = (years >= start_year) & (years <= end_year)
    clipped = {
        **series,
        "dates": series["dates"][keep],
        "values": series["values"][keep],
        "year": years[keep],
        "month": series["month"][keep],
        "water_year": series["water_year"][keep],
    }
    clipped["years"] = (
        (int(years[keep].min()), int(years[keep].max())) if keep.any() else (0, 0)
    )
    return clipped


def _serialise(series: dict | None) -> dict | None:
    if series is None:
        return None
    return {
        "start_date": str(series["dates"][0]),
        "values": np.asarray(series["values"], dtype=np.float32),
        "source": series["source"],
        "label": series["label"],
    }


def _deserialise(payload: dict | None) -> dict | None:
    if payload is None:
        return None
    values = np.asarray(payload["values"], dtype=np.float32)
    start = np.datetime64(payload["start_date"], "D")
    dates = start + np.arange(len(values), dtype="timedelta64[D]")
    series = _with_calendar({
        "dates": dates,
        "values": values,
        "source": payload["source"],
        "label": payload["label"],
    })
    years = series["year"]
    series["years"] = (int(years.min()), int(years.max())) if len(years) else (0, 0)
    return series


def rainfall_summary(series: dict) -> dict:
    """Mean annual, 75 percent dependable annual, and the monthly profile."""
    values = series["values"]
    if len(values) == 0:
        return {"mean_annual_mm": 0.0, "dependable_annual_mm": 0.0, "monthly_mm": [0.0] * 12, "years": 0}

    years = series.get("year")
    months = series.get("month")
    if years is None or months is None:
        series = _with_calendar(series)
        years, months = series["year"], series["month"]

    annual = np.array([values[years == y].sum() for y in np.unique(years)])
    monthly = [float(values[months == m].sum() / max(1, len(np.unique(years)))) for m in range(1, 13)]

    return {
        "mean_annual_mm": round(float(annual.mean()), 1),
        "dependable_annual_mm": round(float(weibull_dependable(annual, 75.0)), 1),
        "min_annual_mm": round(float(annual.min()), 1),
        "max_annual_mm": round(float(annual.max()), 1),
        "monthly_mm": [round(v, 1) for v in monthly],
        "years": int(len(annual)),
        "source": series.get("label", series.get("source", "unknown")),
    }


def weibull_dependable(values: np.ndarray, percent: float = 75.0) -> float:
    """
    Value exceeded in ``percent`` of years, by the Weibull plotting position
    P = m / (N + 1) with the series ranked largest first.

    Linear interpolation between the neighbouring ranks keeps the estimate
    stable when the record is short.
    """
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    n = len(values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(values[0])
    ordered = np.sort(values)[::-1]              # rank 1 = largest
    position = (percent / 100.0) * (n + 1)       # fractional rank
    if position <= 1:
        return float(ordered[0])
    if position >= n:
        return float(ordered[-1])
    lo = int(np.floor(position)) - 1
    frac = position - np.floor(position)
    return float(ordered[lo] + frac * (ordered[lo + 1] - ordered[lo]))
