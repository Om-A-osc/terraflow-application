"""
Open-water evaporation for the pond water balance.

Reference evapotranspiration is computed locally with the FAO-56 Penman-Monteith
equation from the NASA POWER monthly climatology (POWER publishes no ET0
parameter of its own; EVLAND is actual land evaporation, which is a different
quantity).  Without wind, humidity or radiation the Hargreaves temperature-only
equation is used instead, and if the network is unavailable a coarse Indian
monthly climatology keeps the analysis running with a visible warning.

Pond evaporation is ET0 times the open-water coefficient 1.05 that FAO-56
Table 12 gives for shallow tropical water.
"""

from __future__ import annotations

import logging
import math

import numpy as np

import config
from core.cache import cached_call, make_key
from core.resilience import DataQuality, call_source, get

logger = logging.getLogger(__name__)

DAYS_IN_MONTH = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Fallback monthly ET0 (mm/day) for the Indian plains, used only when no
# climatology can be fetched.  Deliberately mid-range and flagged in notes.
FALLBACK_ET0_MM_DAY = [3.2, 4.1, 5.3, 6.4, 7.0, 5.6, 4.1, 3.8, 4.0, 4.1, 3.4, 3.0]


def monthly_et0(lat: float, lon: float, quality: DataQuality | None = None) -> dict:
    """
    Monthly reference evapotranspiration (mm/day) and open-water evaporation
    (mm/month) for a location.
    """
    quality = quality or DataQuality()
    key = make_key("climatology", round(lat, 1), round(lon, 1))
    payload = cached_call(
        key, lambda: _fetch_climatology(lat, lon), expire=config.CACHE_TTL_CLIMATE
    )

    if payload is None:
        et0 = list(FALLBACK_ET0_MM_DAY)
        method = "fallback_climatology"
        quality.note(
            "Evaporation from a generic Indian climatology; NASA POWER was unreachable"
        )
    else:
        et0, method = _et0_from_climatology(payload, lat)

    kc = config.HydrologyParams().open_water_kc
    monthly_mm = [round(e * kc * DAYS_IN_MONTH[m], 1) for m, e in enumerate(et0)]
    return {
        "et0_mm_day": [round(e, 2) for e in et0],
        "open_water_mm_month": monthly_mm,
        "annual_mm": round(sum(monthly_mm), 0),
        "method": method,
        "kc": kc,
    }


def _fetch_climatology(lat: float, lon: float) -> dict | None:
    params = {
        "parameters": "T2M_MAX,T2M_MIN,T2M,RH2M,WS2M,ALLSKY_SFC_SW_DWN",
        "community": "AG",
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "format": "JSON",
    }

    def _call():
        resp = get(config.NASA_POWER_CLIMATOLOGY, params=params, timeout=(3.05, 30.0))
        return resp.json()

    data = call_source("nasa_power_climatology", _call)
    if not data:
        return None
    try:
        return data["properties"]["parameter"]
    except (KeyError, TypeError):
        return None


MONTH_KEYS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _series(param: dict, name: str) -> list | None:
    block = param.get(name)
    if not isinstance(block, dict):
        return None
    out = []
    for key in MONTH_KEYS:
        value = block.get(key)
        if value is None or value <= -900:
            return None
        out.append(float(value))
    return out


def _et0_from_climatology(param: dict, lat: float) -> tuple:
    tmax = _series(param, "T2M_MAX")
    tmin = _series(param, "T2M_MIN")
    tmean = _series(param, "T2M")
    rh = _series(param, "RH2M")
    wind = _series(param, "WS2M")
    rs = _series(param, "ALLSKY_SFC_SW_DWN")

    if tmax is None or tmin is None:
        return list(FALLBACK_ET0_MM_DAY), "fallback_climatology"
    if tmean is None:
        tmean = [(a + b) / 2 for a, b in zip(tmax, tmin)]

    # Representative day of each month, for extraterrestrial radiation
    doys = [17, 46, 75, 105, 135, 162, 198, 228, 258, 288, 318, 344]

    if rh is not None and wind is not None and rs is not None:
        et0 = [
            _penman_monteith(tmax[m], tmin[m], tmean[m], rh[m], wind[m], rs[m], lat, doys[m])
            for m in range(12)
        ]
        return et0, "fao56_penman_monteith"

    et0 = [_hargreaves(tmax[m], tmin[m], tmean[m], lat, doys[m]) for m in range(12)]
    return et0, "hargreaves"


def _extraterrestrial_radiation(lat_deg: float, doy: int) -> float:
    """Ra in MJ m-2 day-1 (FAO-56 eq. 21)."""
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    delta = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    arg = max(-1.0, min(1.0, -math.tan(phi) * math.tan(delta)))
    ws = math.acos(arg)
    return (24 * 60 / math.pi) * 0.0820 * dr * (
        ws * math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.sin(ws)
    )


def _penman_monteith(tmax, tmin, tmean, rh, u2, rs, lat, doy, elevation_m: float = 200.0) -> float:
    """FAO-56 eq. 6, daily step, grass reference."""
    # Vapour pressures (kPa)
    def es_t(t):
        return 0.6108 * math.exp(17.27 * t / (t + 237.3))

    es = (es_t(tmax) + es_t(tmin)) / 2
    ea = es * max(0.0, min(100.0, rh)) / 100.0

    delta = 4098 * es_t(tmean) / (tmean + 237.3) ** 2
    pressure = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26
    gamma = 0.000665 * pressure

    ra = _extraterrestrial_radiation(lat, doy)
    rso = (0.75 + 2e-5 * elevation_m) * ra
    rns = (1 - 0.23) * rs                         # albedo 0.23
    sigma = 4.903e-9
    tmaxk, tmink = tmax + 273.16, tmin + 273.16
    rel = min(1.0, rs / rso) if rso > 0 else 0.5
    rnl = (
        sigma * ((tmaxk ** 4 + tmink ** 4) / 2)
        * (0.34 - 0.14 * math.sqrt(max(ea, 0.0)))
        * (1.35 * rel - 0.35)
    )
    rn = rns - rnl

    numerator = 0.408 * delta * rn + gamma * (900 / (tmean + 273)) * u2 * (es - ea)
    denominator = delta + gamma * (1 + 0.34 * u2)
    return max(0.0, numerator / denominator)


def _hargreaves(tmax, tmin, tmean, lat, doy) -> float:
    """FAO-56 eq. 52, temperature-only."""
    ra = _extraterrestrial_radiation(lat, doy)
    return max(0.0, 0.0023 * (tmean + 17.8) * math.sqrt(max(0.0, tmax - tmin)) * ra * 0.408)
