"""Tests for the full cache system including dependency tracking."""
import asyncio
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestSimpleCacheBasic:
    """Basic tests for simple_cache decorator (same as before)."""

    @pytest.mark.asyncio
    async def test_async_cache_miss_first_call(self):
        """First call should miss cache and execute function."""
        from scout.cache import simple_cache

        call_count = 0

        @simple_cache(ttl_seconds=5)
        async def expensive_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result = await expensive_operation(5)
        assert result == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_cache_hit(self):
        """Second call with same args should hit cache."""
        from scout.cache import simple_cache

        call_count = 0

        @simple_cache(ttl_seconds=5)
        async def expensive_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await expensive_operation(5)
        await expensive_operation(5)

        assert call_count == 1

    def test_sync_cache(self):
        """Cache should also work with sync functions."""
        from scout.cache import simple_cache

        call_count = 0

        @simple_cache(ttl_seconds=5)
        def sync_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        result = sync_operation(5)
        assert result == 10
        assert call_count == 1

        result2 = sync_operation(5)
        assert result2 == 10
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Cache entries should expire after TTL."""
        from scout.cache import simple_cache

        call_count = 0

        @simple_cache(ttl_seconds=0.1)
        async def timed_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await timed_operation(5)
        assert call_count == 1

        await asyncio.sleep(0.15)

        await timed_operation(5)
        assert call_count == 2

    def test_cache_info(self):
        """Cache info should report statistics."""
        from scout.cache import simple_cache

        call_count = 0

        @simple_cache(ttl_seconds=5)
        def tracked_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        tracked_operation(5)
        tracked_operation(5)
        tracked_operation(10)

        info = tracked_operation.cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 2


class TestCacheDependencies:
    """Tests for cache dependency tracking features."""

    def test_resolve_dependencies(self):
        """Test glob pattern resolution."""
        from scout.cache_deps import resolve_dependencies

        # Test with simple glob pattern
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            Path(tmpdir, "test1.py").touch()
            Path(tmpdir, "test2.py").touch()
            Path(tmpdir, "readme.txt").touch()

            # Resolve Python files
            patterns = [os.path.join(tmpdir, "*.py")]
            resolved = resolve_dependencies(patterns)

            # Should find 2 Python files
            assert len(resolved) == 2
            assert any("test1.py" in r for r in resolved)
            assert any("test2.py" in r for r in resolved)

    def test_resolve_dependencies_with_exclusions(self):
        """Test resolve_dependencies with exclude patterns."""
        from scout.cache_deps import resolve_dependencies

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files including test files
            Path(tmpdir, "main.py").touch()
            Path(tmpdir, "test_main.py").touch()

            # Resolve with exclusions
            patterns = [os.path.join(tmpdir, "*.py")]
            resolved = resolve_dependencies(patterns, exclude_patterns=["test_"])

            # Should only find main.py
            assert len(resolved) == 1
            assert any("main.py" in r for r in resolved)

    def test_register_dependency(self):
        """Test registering cache dependencies."""
        from scout.cache_deps import _dependency_map, register_dependency

        # Clear any existing dependencies
        _dependency_map.clear()

        # Create mock wrapper
        mock_wrapper = MagicMock()
        mock_wrapper._cache = {}

        # Register dependency
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir, "test.py")
            test_file.touch()

            register_dependency((mock_wrapper, "test_key"), [str(test_file)])

            # Check dependency was registered
            assert str(test_file) in _dependency_map
            assert (mock_wrapper, "test_key") in _dependency_map[str(test_file)]

    @pytest.mark.asyncio
    async def test_invalidate_for_path(self):
        """Test cache invalidation for a specific file path."""
        from scout.cache_deps import (
            _dependency_map,
            _do_invalidate,
            register_dependency,
            set_event_loop,
        )

        # Clear dependencies
        _dependency_map.clear()

        # Create mock wrapper with cache
        mock_wrapper = MagicMock()
        cache_dict = {"test_key": ("cached_value", time.time())}
        mock_wrapper._cache = cache_dict

        # Register dependency
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir, "test.py")
            test_file.touch()

            register_dependency((mock_wrapper, "test_key"), [str(test_file)])

            # Set event loop for async invalidation
            set_event_loop(asyncio.get_event_loop())

            # Invalidate for the file (async version)
            await _do_invalidate(str(test_file))

            # Check cache was cleared (key should be removed)
            assert "test_key" not in cache_dict

    def test_simple_cache_with_dependencies(self):
        """Test simple_cache with file dependencies (without file watcher)."""
        from scout.cache import simple_cache
        from scout.cache_deps import _dependency_map

        # Clear dependencies
        _dependency_map.clear()

        call_count = 0

        @simple_cache(ttl_seconds=5, dependencies=["*.py"])
        def operation_with_deps(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call
        result = operation_with_deps(5)
        assert result == 10
        assert call_count == 1

        # Second call should hit cache
        result = operation_with_deps(5)
        assert result == 10
        assert call_count == 1

        # Check dependencies were registered
        # (The actual file paths depend on what glob resolves)


class TestCacheStats:
    """Tests for CacheStats class."""

    def test_cache_stats_properties(self):
        """Test CacheStats class properties."""
        from scout.cache import CacheStats

        stats = {"hits": 80, "misses": 20, "evictions": 5}
        cache_stats = CacheStats(stats, ttl_seconds=10)

        assert cache_stats.hits == 80
        assert cache_stats.misses == 20
        assert cache_stats.evictions == 5
        assert cache_stats.hit_rate == 80.0

    def test_cache_stats_zero_misses(self):
        """Test CacheStats with zero misses (100% hit rate)."""
        from scout.cache import CacheStats

        stats = {"hits": 100, "misses": 0, "evictions": 0}
        cache_stats = CacheStats(stats, ttl_seconds=10)

        assert cache_stats.hit_rate == 100.0


class TestCachedAlias:
    """Tests for the cached() alias."""

    @pytest.mark.asyncio
    async def test_cached_alias(self):
        """Test that cached() is an alias for simple_cache."""
        from scout.cache import cached, simple_cache

        call_count = 0

        @cached(ttl_seconds=5)
        async def cached_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 3

        # First call
        await cached_operation(5)
        assert call_count == 1

        # Second call should hit cache
        await cached_operation(5)
        assert call_count == 1


class TestCacheConcurrency:
    """Tests for cache concurrency handling."""

    @pytest.mark.asyncio
    async def test_concurrent_async_calls(self):
        """Test that concurrent async calls are properly handled."""
        from scout.cache import simple_cache

        call_count = 0

        @simple_cache(ttl_seconds=5)
        async def concurrent_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            # Simulate some work
            await asyncio.sleep(0.05)
            return x * 2

        # Launch multiple concurrent calls
        results = await asyncio.gather(
            concurrent_operation(5),
            concurrent_operation(5),
            concurrent_operation(5),
        )

        # Should only have executed once due to in-flight coalescing
        assert call_count == 1
        # All results should be the same
        assert all(r == 10 for r in results)

    def test_concurrent_sync_calls(self):
        """Test that concurrent sync calls are properly handled."""
        from scout.cache import simple_cache
        import threading

        call_count = 0

        @simple_cache(ttl_seconds=5)
        def sync_concurrent_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            time.sleep(0.05)
            return x * 2

        threads = []
        for _ in range(3):
            t = threading.Thread(target=lambda: sync_concurrent_operation(5))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should only have executed once due to in-flight coalescing
        assert call_count == 1
