"""
CPU work off the event loop.

Analysis is NumPy and numba work that holds the GIL, so running it inline would
block every other request on the worker.  It goes to a ProcessPoolExecutor
sized to the physical cores: measured throughput on this class of machine
plateaus at eight workers, and more only adds memory pressure.

Python 3.10 has no ``max_tasks_per_child``, so the pool is rebuilt after a set
number of jobs to keep NumPy and numba fragmentation in check.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import config

logger = logging.getLogger(__name__)

_pool: ProcessPoolExecutor | None = None
_submitted = 0
_inflight = 0
_RECYCLE_AFTER = 500


def _initializer() -> None:
    """Warm each worker so the first real request does not pay for the JIT."""
    os.environ.setdefault("NUMBA_CACHE_DIR", str(config.NUMBA_CACHE_DIR))
    try:
        from services.hydrology_engine import warm_up

        warm_up()
    except Exception as exc:  # pragma: no cover - a cold worker still works
        logging.getLogger(__name__).warning("Worker warm-up failed: %s", exc)


def get_pool() -> ProcessPoolExecutor:
    global _pool, _submitted
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=config.POOL_WORKERS, initializer=_initializer)
        _submitted = 0
        logger.info("Process pool started with %d workers", config.POOL_WORKERS)
    return _pool


def shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
        logger.info("Process pool shut down")


def stats() -> dict:
    return {
        "workers": config.POOL_WORKERS,
        "submitted": _submitted,
        "inflight": _inflight,
        "started": _pool is not None,
    }


class ComputeTimeout(TimeoutError):
    """Raised when a job exceeds the compute budget."""


async def run(fn, *args, timeout: float | None = None, **kwargs):
    """
    Run ``fn(*args, **kwargs)`` in the pool with a hard timeout.

    Falls back to a thread if the pool cannot start (for example in a sandbox
    that forbids forking), so the service degrades rather than failing.
    """
    global _submitted, _inflight

    timeout = timeout or config.COMPUTE_TIMEOUT_S
    loop = asyncio.get_running_loop()

    if kwargs:
        from functools import partial

        call = partial(fn, *args, **kwargs)
        args = ()
    else:
        call = fn

    try:
        pool = get_pool()
        _submitted += 1
        _inflight += 1
        if _submitted >= _RECYCLE_AFTER:
            logger.info("Recycling the process pool after %d jobs", _submitted)
            shutdown_pool()
            pool = get_pool()
        future = loop.run_in_executor(pool, call, *args)
    except Exception as exc:  # noqa: BLE001 - forking may be unavailable
        _inflight = max(0, _inflight - 1)
        logger.warning("Process pool unavailable (%s); running in a thread", exc)
        return await asyncio.wait_for(
            loop.run_in_executor(None, call, *args), timeout=timeout
        )

    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError as exc:
        from core.cache import bump

        bump("timeouts")
        raise ComputeTimeout(
            f"The analysis did not finish within {timeout:.0f} seconds. "
            "This village's data is still being fetched: wait for the preparation "
            "to finish, or select a smaller area."
        ) from exc
    finally:
        _inflight = max(0, _inflight - 1)
