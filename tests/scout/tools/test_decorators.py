"""Tests for tool decorators (simple_cache and log_tool_invocation)."""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSimpleCache:
    """Tests for the simple_cache decorator."""

    @pytest.mark.asyncio
    async def test_async_cache_miss_first_call(self):
        """First call should miss cache and execute function."""
        from scout.tools import simple_cache

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
        from scout.tools import simple_cache

        call_count = 0

        @simple_cache(ttl_seconds=5)
        async def expensive_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await expensive_operation(5)
        await expensive_operation(5)

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_cache_different_args(self):
        """Different arguments should result in separate cache entries."""
        from scout.tools import simple_cache

        call_count = 0

        @simple_cache(ttl_seconds=5)
        async def expensive_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        await expensive_operation(5)
        await expensive_operation(10)

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_sync_cache(self):
        """Cache should also work with sync functions."""
        from scout.tools import simple_cache

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
        from scout.tools import simple_cache

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
        from scout.tools import simple_cache

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

    def test_cache_clear(self):
        """Cache should be clearable."""
        from scout.tools import simple_cache

        call_count = 0

        @simple_cache(ttl_seconds=5)
        def clearable_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        clearable_operation(5)
        assert call_count == 1

        clearable_operation.cache_clear()
        clearable_operation(5)
        assert call_count == 2


class TestLogToolInvocation:
    """Tests for the log_tool_invocation decorator."""

    @pytest.mark.asyncio
    async def test_logs_successful_invocation(self):
        """Should log successful tool invocations."""
        from scout.tools import log_tool_invocation

        mock_audit = MagicMock()
        mock_audit.log = MagicMock()

        with patch("scout.tools.get_audit", return_value=mock_audit):

            @log_tool_invocation
            async def my_tool(x: int) -> int:
                return x * 2

            result = await my_tool(5)
            assert result == 10
            mock_audit.log.assert_called_once()
            call_kwargs = mock_audit.log.call_args[1]
            assert call_kwargs["tool"] == "my_tool"
            assert call_kwargs["success"] is True
            assert "duration_ms" in call_kwargs

    @pytest.mark.asyncio
    async def test_logs_failed_invocation(self):
        """Should log failed tool invocations with error."""
        from scout.tools import log_tool_invocation

        mock_audit = MagicMock()
        mock_audit.log = MagicMock()

        with patch("scout.tools.get_audit", return_value=mock_audit):

            @log_tool_invocation
            async def failing_tool() -> None:
                raise ValueError("Something went wrong")

            with pytest.raises(ValueError):
                await failing_tool()

            mock_audit.log.assert_called_once()
            call_kwargs = mock_audit.log.call_args[1]
            assert call_kwargs["tool"] == "failing_tool"
            assert call_kwargs["success"] is False
            assert call_kwargs["error"] == "Something went wrong"

    def test_logs_sync_invocation(self):
        """Should work with sync functions too."""
        from scout.tools import log_tool_invocation

        mock_audit = MagicMock()
        mock_audit.log = MagicMock()

        with patch("scout.tools.get_audit", return_value=mock_audit):

            @log_tool_invocation
            def sync_tool(x: int) -> int:
                return x + 1

            result = sync_tool(5)
            assert result == 6
            mock_audit.log.assert_called_once()

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        """Decorated function should preserve its name and docstring."""
        from scout.tools import log_tool_invocation

        @log_tool_invocation
        async def documented_tool(x: int) -> int:
            """This is the docstring."""
            return x * 2

        assert documented_tool.__name__ == "documented_tool"
        assert documented_tool.__doc__ == "This is the docstring."
