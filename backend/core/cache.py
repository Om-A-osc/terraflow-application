"""
Disk cache shared by the uvicorn workers and the process pool of one node.

diskcache is SQLite-backed, so it is process-safe on a local filesystem (never
put it on NFS).  Keys are tuples; every key that depends on the algorithm is
versioned with ALGO_VERSION so a code change invalidates it automatically.

Note on stampedes: diskcache's ``memoize_stampede`` only guards *expiry*, not a
cold miss, so first-time computations are serialised here with ``cache.lock``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from typing import Any, Callable

from diskcache import FanoutCache

import config

logger = logging.getLogger(__name__)

_cache: FanoutCache | None = None


def get_cache() -> FanoutCache:
    """Process-local handle to the shared on-disk cache."""
    global _cache
    if _cache is None:
        _cache = FanoutCache(
            directory=str(config.CACHE_DIR),
            shards=config.CACHE_SHARDS,
            timeout=5,
            size_limit=config.CACHE_SIZE_LIMIT_BYTES,
            eviction_policy="least-recently-used",
        )
        logger.info("Disk cache at %s (%d shards)", config.CACHE_DIR, config.CACHE_SHARDS)
    return _cache


def _norm(value: Any) -> Any:
    """Make a value stable and hashable for key building."""
    if isinstance(value, dict):
        return {k: _norm(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, float):
        # Avoid 0.1 + 0.2 style drift in keys
        return round(value, 10)
    return value


def make_key(namespace: str, *parts: Any, **kwargs: Any) -> str:
    """
    Build a versioned cache key.

    Floats are rounded and dict keys sorted, so the same request expressed
    slightly differently still hits the same entry.
    """
    payload = json.dumps(
        {"parts": _norm(list(parts)), "kwargs": _norm(kwargs)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha1(payload.encode()).hexdigest()
    return f"{namespace}:{config.ALGO_VERSION}:{digest}"


def polygon_key(coords: list, **params: Any) -> str:
    """
    Result-cache key for a drawn polygon.

    Coordinates are rounded to 1e-5 degrees (about 1 m), so redrawing what looks
    like the same polygon still hits the cache.
    """
    rounded = [[round(float(x), 5), round(float(y), 5)] for x, y in coords]
    return make_key("result", rounded, **params)


@contextmanager
def compute_lock(key: str, expire: int = 120, timeout: float = 60.0):
    """
    Cross-process mutex so two users asking for the same cold analysis do not
    both compute it.

    ``FanoutCache`` has no ``.lock`` helper, so this uses the atomic ``add``
    primitive: only the caller whose insert succeeds holds the lock, and the
    entry expires on its own if that process dies.  Falls through *without* the
    lock once ``timeout`` passes, which is better than failing the request.
    """
    import time

    cache = get_cache()
    lock_key = f"lock:{key}"
    acquired = False
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                acquired = bool(cache.add(lock_key, True, expire=expire, retry=False))
            except Exception:  # pragma: no cover - never fail a request on the lock
                acquired = False
                break
            if acquired or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        yield acquired
    finally:
        if acquired:
            try:
                cache.delete(lock_key, retry=False)
            except Exception:  # pragma: no cover
                pass


def cached_call(
    key: str,
    producer: Callable[[], Any],
    expire: int | None = None,
    use_lock: bool = True,
) -> Any:
    """
    Get ``key`` from the cache, or compute it with ``producer`` and store it.

    Counts hits and misses for /metrics.
    """
    cache = get_cache()
    hit = cache.get(key, default=None)
    if hit is not None:
        bump("cache_hits")
        return hit

    if not use_lock:
        value = producer()
        cache.set(key, value, expire=expire)
        bump("cache_misses")
        return value

    with compute_lock(key):
        # Someone else may have produced it while we waited for the lock.
        hit = cache.get(key, default=None)
        if hit is not None:
            bump("cache_hits")
            return hit
        value = producer()
        cache.set(key, value, expire=expire)
        bump("cache_misses")
        return value


# ─────────────────────────────────────────────────────────────────────────────
# Counters for /metrics — stored in the cache so every worker contributes
# ─────────────────────────────────────────────────────────────────────────────


def bump(name: str, amount: int = 1) -> None:
    try:
        get_cache().incr(f"metric:{name}", amount, default=0, retry=False)
    except Exception:  # pragma: no cover - metrics must never break a request
        pass


METRIC_NAMES = (
    "cache_hits",
    "cache_misses",
    "analyses",
    "analyses_failed",
    "earthwork_runs",
    "dem_fetch_copernicus",
    "dem_fetch_terrain_tiles",
    "dem_fetch_failed",
    "osm_fetch",
    "osm_failed",
    "geocode_local",
    "geocode_photon",
    "rainfall_local",
    "rainfall_power",
    "breaker_open",
    "timeouts",
)


def read_metrics() -> dict[str, int]:
    cache = get_cache()
    out: dict[str, int] = {}
    for name in METRIC_NAMES:
        try:
            out[name] = int(cache.get(f"metric:{name}", default=0) or 0)
        except Exception:
            out[name] = 0
    return out
