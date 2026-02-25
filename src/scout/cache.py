"""
Reusable Caching Decorator for Async and Sync Functions

Provides a simple, thread-safe caching decorator with TTL support.
Designed for use with MCP tools to reduce redundant API calls.

Usage:
    from scout.cache import simple_cache

    # For async functions
    @mcp.tool()
    @simple_cache(ttl_seconds=5)
    async def my_async_tool(param1: str, param2: int) -> str:
        # Expensive operation here
        ...

    # For sync functions
    @mcp.tool()
    @simple_cache(ttl_seconds=5)
    def my_sync_tool(param1: str) -> str:
        # Expensive operation here
        ...
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import threading
import time
from functools import wraps
from typing import Any, Callable, TypeVar

from scout.cache_deps import (
    ensure_watcher_started,
    register_dependency,
    resolve_dependencies,
)

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    import logging

    # Only configure if no handlers exist (avoid overriding centralized config)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    logger = logging.getLogger(__name__)


def _log_debug(msg: str, **kwargs) -> None:
    """Log debug message, handling both structlog and standard logging."""
    if hasattr(logger, "bind"):
        logger.debug(msg, **kwargs)
    else:
        logger.debug(f"{msg}: {kwargs}")


def _log_info(msg: str, **kwargs) -> None:
    """Log info message, handling both structlog and standard logging."""
    if hasattr(logger, "bind"):
        logger.info(msg, **kwargs)
    else:
        logger.info(f"{msg}: {kwargs}")


def _log_warning(msg: str, **kwargs) -> None:
    """Log warning message, handling both structlog and standard logging."""
    if hasattr(logger, "bind"):
        logger.warning(msg, **kwargs)
    else:
        logger.warning(f"{msg}: {kwargs}")


T = TypeVar("T")


def _make_cache_key(*args: Any, **kwargs: Any) -> str:
    """Generate a deterministic cache key from args and kwargs.

    Excludes transient metadata like timestamps, session IDs, and other
    non-content-affecting parameters to reduce cache misses.
    """
    # Fields to exclude from cache key as they are transient/non-deterministic
    TRANSIENT_FIELDS = {
        "timestamp",
        "time",
        "datetime",
        "session_id",
        "sessionId",
        "request_id",
        "requestId",
        "id",
        "uuid",
        "token",
        "nonce",
        "_timestamp",
        "_time",
        "_cache",
        "cache",
        "stats",
    }

    key_parts = []

    for arg in args:
        # Skip None, booleans (often used as flags), and known transient types
        if arg is None or isinstance(arg, bool):
            continue
        if hasattr(arg, "__dict__"):
            try:
                # Filter out transient fields from __dict__
                filtered = {
                    k: v
                    for k, v in arg.__dict__.items()
                    if k not in TRANSIENT_FIELDS and v is not None
                }
                if filtered:
                    key_parts.append(str(filtered))
            except Exception:
                key_parts.append(repr(arg))
        else:
            # Skip common transient types
            if isinstance(arg, (int, float)) and arg > 1e15:  # Likely a timestamp in ms
                continue
            key_parts.append(str(arg))

    # Filter kwargs to exclude transient fields
    filtered_kwargs = {
        k: v
        for k, v in sorted(kwargs.items())
        if k not in TRANSIENT_FIELDS and v is not None
    }

    for k, v in filtered_kwargs.items():
        if hasattr(v, "__dict__"):
            try:
                filtered = {
                    key: val
                    for key, val in v.__dict__.items()
                    if key not in TRANSIENT_FIELDS and val is not None
                }
                if filtered:
                    key_parts.append(f"{k}={filtered}")
            except Exception:
                key_parts.append(f"{k}={repr(v)}")
        else:
            # Skip common transient types
            if isinstance(v, (int, float)) and v > 1e15:  # Likely a timestamp in ms
                continue
            key_parts.append(f"{k}={v}")

    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]


def simple_cache(
    ttl_seconds: float = 5.0,
    max_size: int = 1000,
    dependencies: list[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Simple caching decorator with TTL supporting both sync and async functions.

    Args:
        ttl_seconds: Time-to-live for cached results in seconds.
        max_size: Maximum number of entries to store in cache.
        dependencies: List of glob patterns for files this cache depends on.
                      When any of these files change, the cache entry is invalidated.

    Returns:
        A decorator that wraps functions with caching.

    Features:
        - Works with both sync and async functions
        - Thread-safe with threading.Lock for sync, asyncio.Lock for async
        - Tracks hits/misses for monitoring
        - Automatic expiration based on TTL
        - LRU-style eviction when max_size is reached
        - Optional dependency tracking for file-based invalidation
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        is_async = inspect.iscoroutinefunction(func)

        cache: dict[str, tuple[Any, float]] = {}
        stats: dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}

        # Use appropriate lock based on function type
        if is_async:
            lock = asyncio.Lock()
        else:
            lock = threading.Lock()

        def _check_cache(key: str, now: float) -> tuple[Any, bool]:
            """Check if key exists and is not expired. Returns (result, was_hit)."""
            if key in cache:
                result, timestamp = cache[key]
                age = now - timestamp
                if age < ttl_seconds:
                    return result, True
                else:
                    del cache[key]
            return None, False

        def _store_cache(key: str, result: Any, now: float) -> None:
            """Store result in cache with eviction if needed."""
            if len(cache) >= max_size:
                oldest_key = min(cache.keys(), key=lambda k: cache[k][1])
                del cache[oldest_key]
                stats["evictions"] += 1
            cache[key] = (result, now)

        if is_async:
            # Per-key events for waiting on in-flight requests
            in_flight_events: dict[str, asyncio.Event] = {}

            @wraps(func)
            async def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Ensure file watcher is started if dependencies are configured
                if dependencies:
                    ensure_watcher_started()

                key = _make_cache_key(*args, **kwargs)

                while True:  # Retry loop for in-flight wait
                    now = time.time()

                    async with lock:
                        result, was_hit = _check_cache(key, now)
                        if was_hit:
                            stats["hits"] += 1
                            _log_debug(
                                "cache_hit",
                                function=func.__name__,
                                key=key[:8],
                                age_seconds=round(now - cache[key][1], 3),
                            )
                            return result

                        # Check if another request is already executing for this key
                        if key in in_flight_events:
                            # Wait for the in-flight request to complete
                            event = in_flight_events[key]
                            # Release lock and wait - need to do this outside
                            wait_event = event
                        else:
                            # Mark this key as in-flight
                            in_flight_events[key] = asyncio.Event()
                            stats["misses"] += 1
                            break  # We own this execution

                    # If there's an in-flight request, wait for it
                    if "wait_event" in locals():
                        await wait_event.wait()
                        # Loop back to check cache again

                try:
                    result = await func(*args, **kwargs)
                except Exception as e:
                    async with lock:
                        if key in in_flight_events:
                            in_flight_events[key].set()
                            in_flight_events.pop(key, None)
                    _log_warning(
                        "cache_skip_error", function=func.__name__, error=str(e)
                    )
                    raise

                now = time.time()
                async with lock:
                    _store_cache(key, result, now)
                    # Register dependencies for this cache entry
                    if dependencies:
                        file_paths = resolve_dependencies(
                            dependencies,
                            exclude_patterns=["tests/", "test_", "_test.py", "/tests/"],
                        )
                        if file_paths:
                            register_dependency((wrapper, key), file_paths)
                    _log_debug(
                        "cache_miss_stored",
                        function=func.__name__,
                        key=key[:8],
                        cache_size=len(cache),
                    )
                    # Signal any waiting requests and clear in-flight
                    if key in in_flight_events:
                        in_flight_events[key].set()
                        in_flight_events.pop(key, None)

                return result

        else:
            # Per-key events for waiting on in-flight requests (sync version)
            in_flight_events: dict[str, threading.Event] = {}

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Ensure file watcher is started if dependencies are configured
                if dependencies:
                    ensure_watcher_started()

                key = _make_cache_key(*args, **kwargs)

                while True:  # Retry loop for in-flight wait
                    now = time.time()

                    with lock:
                        result, was_hit = _check_cache(key, now)
                        if was_hit:
                            stats["hits"] += 1
                            _log_debug(
                                "cache_hit",
                                function=func.__name__,
                                key=key[:8],
                                age_seconds=round(now - cache[key][1], 3),
                            )
                            return result

                        # Check if another request is already executing for this key
                        if key in in_flight_events:
                            # Wait for the in-flight request to complete
                            event = in_flight_events[key]
                            wait_event = event
                        else:
                            # Mark this key as in-flight
                            in_flight_events[key] = threading.Event()
                            stats["misses"] += 1
                            break  # We own this execution

                    # If there's an in-flight request, wait for it
                    if "wait_event" in locals():
                        wait_event.wait()
                        # Loop back to check cache again

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    with lock:
                        if key in in_flight_events:
                            in_flight_events[key].set()
                            in_flight_events.pop(key, None)
                    _log_warning(
                        "cache_skip_error", function=func.__name__, error=str(e)
                    )
                    raise

                now = time.time()
                with lock:
                    _store_cache(key, result, now)
                    # Register dependencies for this cache entry
                    if dependencies:
                        file_paths = resolve_dependencies(
                            dependencies,
                            exclude_patterns=["tests/", "test_", "_test.py", "/tests/"],
                        )
                        if file_paths:
                            register_dependency((wrapper, key), file_paths)
                    _log_debug(
                        "cache_miss_stored",
                        function=func.__name__,
                        key=key[:8],
                        cache_size=len(cache),
                    )
                    # Signal any waiting requests and clear in-flight
                    if key in in_flight_events:
                        in_flight_events[key].set()
                        in_flight_events.pop(key, None)

                return result

        # Attach cache dict to wrapper for external invalidation
        wrapper._cache = cache
        wrapper.cache_stats = stats
        wrapper.cache_clear = lambda: cache.clear()
        wrapper.cache_info = lambda: _get_cache_info(cache, stats, ttl_seconds)

        return wrapper

    return decorator


def _get_cache_info(
    cache: dict[str, tuple[Any, float]], stats: dict[str, int], ttl_seconds: float
) -> dict[str, Any]:
    """Get cache statistics."""
    total_requests = stats["hits"] + stats["misses"]
    hit_rate = (stats["hits"] / total_requests * 100) if total_requests > 0 else 0.0

    return {
        "size": len(cache),
        "hits": stats["hits"],
        "misses": stats["misses"],
        "evictions": stats["evictions"],
        "hit_rate_percent": round(hit_rate, 2),
        "ttl_seconds": ttl_seconds,
    }


class CacheStats:
    """Container for cache statistics with convenience methods."""

    def __init__(self, stats: dict[str, int], ttl_seconds: float):
        self._stats = stats
        self._ttl = ttl_seconds

    @property
    def hits(self) -> int:
        return self._stats["hits"]

    @property
    def misses(self) -> int:
        return self._stats["misses"]

    @property
    def evictions(self) -> int:
        return self._stats["evictions"]

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1f}%, evictions={self.evictions})"
        )


def cached(
    ttl_seconds: float = 5.0,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Alias for simple_cache for more intuitive usage.

    Example:
        @cached(ttl_seconds=10)
        async def expensive_call(x: int) -> int:
            return x * 2
    """
    return simple_cache(ttl_seconds=ttl_seconds)
