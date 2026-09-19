"""
Resilience helpers for the external data sources.

Every outside service (DEM bucket, Overpass, NASA POWER, Photon) sits behind its
own circuit breaker: three consecutive failures open it for sixty seconds, so a
host that is down or blocked on the campus network costs one timeout, not one
per request.  Callers record what degraded in a DataQuality object which the API
returns and the confidence score reads.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pybreaker
import requests

import config
from core.cache import bump

logger = logging.getLogger(__name__)

_breakers: dict[str, pybreaker.CircuitBreaker] = {}


def breaker(name: str) -> pybreaker.CircuitBreaker:
    """One breaker per external source."""
    if name not in _breakers:
        _breakers[name] = pybreaker.CircuitBreaker(
            fail_max=config.BREAKER_FAIL_MAX,
            reset_timeout=config.BREAKER_RESET_TIMEOUT_S,
            name=name,
        )
    return _breakers[name]


def breaker_states() -> dict[str, str]:
    return {name: b.current_state for name, b in _breakers.items()}


class SourceUnavailable(RuntimeError):
    """Raised when a source failed and no fallback produced data."""


# A source that just failed is marked down in the shared cache for this long.
# Each analysis runs in its own worker process, so a per-process breaker has to
# be taught the same lesson eight times over, and every worker pays the full
# timeout learning it.  A shared marker means one failure is enough.
SOURCE_DOWN_S = 90


def _down_key(name: str) -> str:
    return f"down:{name}"


def is_source_down(name: str) -> bool:
    try:
        from core.cache import get_cache

        return bool(get_cache().get(_down_key(name), default=False))
    except Exception:  # pragma: no cover - never fail a request on this
        return False


def mark_source_down(name: str, seconds: int = SOURCE_DOWN_S) -> None:
    try:
        from core.cache import get_cache

        get_cache().set(_down_key(name), True, expire=seconds)
    except Exception:  # pragma: no cover
        pass


def call_source(name: str, fn: Callable[[], Any], default: Any = None) -> Any:
    """
    Run ``fn`` behind the named breaker.

    Returns ``default`` instead of raising when the source fails or the breaker
    is open, so callers can move to their next fallback.  A recent failure
    recorded by any worker short-circuits the call, which is what keeps a dead
    host from costing every process its own timeout.
    """
    if is_source_down(name):
        bump("breaker_open")
        logger.info("Skipping %s: marked down by an earlier failure", name)
        return default

    brk = breaker(name)
    try:
        return brk.call(fn)
    except pybreaker.CircuitBreakerError:
        bump("breaker_open")
        logger.warning("Circuit breaker open for %s, skipping", name)
        return default
    except Exception as exc:  # noqa: BLE001 - deliberately broad, it is a boundary
        logger.warning("Source %s failed: %s", name, exc)
        mark_source_down(name)
        return default


_session: requests.Session | None = None


def http() -> requests.Session:
    """Shared session so connections to the same host are reused."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": config.HTTP_USER_AGENT})
        adapter = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=32)
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    return _session


def get(url: str, *, timeout: tuple | None = None, **kwargs) -> requests.Response:
    resp = http().get(url, timeout=timeout or config.HTTP_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp


def post(url: str, *, timeout: tuple | None = None, **kwargs) -> requests.Response:
    resp = http().post(url, timeout=timeout or config.HTTP_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp


@dataclass
class DataQuality:
    """
    What the analysis actually managed to use.

    Every degraded path appends a note here; the notes are returned to the
    client and lower the confidence label of each suggested site.
    """

    dem_source: str = "unknown"
    rainfall_source: str = "unknown"
    cn_source: str = "unknown"
    osm_available: bool = False
    terrain_corrected: bool = False
    catchment_truncated: bool = False
    notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)
        logger.info("data quality: %s", message)

    def score(self) -> float:
        """0..1 factor used by the site confidence calculation."""
        s = 1.0
        if self.dem_source.startswith("copernicus"):
            s *= 1.0
        elif self.dem_source.startswith("terrain"):
            s *= 0.85          # SRTM 2000-vintage, integer metres
        elif self.dem_source == "contour_kml":
            s *= 1.0           # a real survey beats any global DEM
        else:
            s *= 0.6
        if not self.osm_available:
            s *= 0.75          # constraints unknown
        if not self.terrain_corrected:
            s *= 0.95
        if self.rainfall_source.startswith("imd"):
            s *= 1.0
        elif self.rainfall_source.startswith("power"):
            s *= 0.85          # reanalysis smooths daily intensity
        else:
            s *= 0.7
        if self.cn_source == "gcn250":
            s *= 1.0
        else:
            s *= 0.9
        if self.catchment_truncated:
            s *= 0.8
        return round(min(1.0, s), 3)

    def to_dict(self) -> dict:
        return {
            "dem_source": self.dem_source,
            "rainfall_source": self.rainfall_source,
            "cn_source": self.cn_source,
            "osm_available": self.osm_available,
            "terrain_corrected": self.terrain_corrected,
            "catchment_truncated": self.catchment_truncated,
            "score": self.score(),
            "notes": list(self.notes),
        }


class Timer:
    """Tiny stopwatch used to fill the ``timings`` block of a response."""

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.marks: dict[str, float] = {}
        self._last = self.t0

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.marks[name] = round(now - self._last, 3)
        self._last = now

    def total(self) -> float:
        return round(time.perf_counter() - self.t0, 3)

    def to_dict(self) -> dict:
        return {**self.marks, "total_s": self.total()}
