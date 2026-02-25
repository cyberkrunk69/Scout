"""Tests for retry.py module."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

from scout.retry import (
    RetryConfig,
    RetryStats,
    calculate_backoff,
    is_retryable,
    RetryContext,
    with_retry_async,
    with_retry_sync,
    retry_async,
    retry_sync,
)


class TestRetryConfig:
    """Tests for RetryConfig class."""

    def test_default_values(self):
        """Test default retry config values."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.base_delay_ms == 1000.0
        assert config.max_delay_ms == 30000.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter == 0.1

    def test_custom_values(self):
        """Test custom retry config values."""
        config = RetryConfig(
            max_retries=5,
            base_delay_ms=500.0,
            max_delay_ms=10000.0,
        )

        assert config.max_retries == 5
        assert config.base_delay_ms == 500.0


class TestRetryStats:
    """Tests for RetryStats class."""

    def test_creation(self):
        """Test creating RetryStats."""
        stats = RetryStats()

        assert stats.attempts == 0
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.total_delay_ms == 0.0
        assert stats.last_error is None


class TestCalculateBackoff:
    """Tests for calculate_backoff function."""

    def test_first_attempt(self):
        """Test backoff for first attempt."""
        config = RetryConfig(base_delay_ms=1000.0, max_delay_ms=10000.0, jitter=0.0)

        delay = calculate_backoff(0, config)

        assert delay == 1000.0

    def test_exponential_growth(self):
        """Test exponential backoff growth."""
        config = RetryConfig(base_delay_ms=1000.0, max_delay_ms=100000.0, jitter=0.0)

        delay0 = calculate_backoff(0, config)
        delay1 = calculate_backoff(1, config)
        delay2 = calculate_backoff(2, config)

        assert delay1 > delay0
        assert delay2 > delay1

    def test_max_delay_cap(self):
        """Test that max delay is capped."""
        config = RetryConfig(base_delay_ms=1000.0, max_delay_ms=5000.0, jitter=0.0)

        delay = calculate_backoff(10, config)  # Would be much higher without cap

        assert delay == 5000.0


class TestIsRetryable:
    """Tests for is_retryable function."""

    def test_retryable_connection_error(self):
        """Test that ConnectionError is retryable."""
        config = RetryConfig()

        assert is_retryable(ConnectionError("connection failed"), config) is True

    def test_retryable_timeout_error(self):
        """Test that TimeoutError is retryable."""
        config = RetryConfig()

        assert is_retryable(TimeoutError("timed out"), config) is True

    def test_retryable_asyncio_timeout(self):
        """Test that asyncio.TimeoutError is retryable."""
        config = RetryConfig()

        assert is_retryable(asyncio.TimeoutError("timeout"), config) is True

    def test_not_retryable_value_error(self):
        """Test that ValueError is not retryable."""
        config = RetryConfig()

        assert is_retryable(ValueError("invalid value"), config) is False

    def test_retryable_with_status_code(self):
        """Test retryable with status code attribute."""
        config = RetryConfig()

        class MockError(Exception):
            def __init__(self, status_code):
                self.status_code = status_code

        assert is_retryable(MockError(429), config) is True
        assert is_retryable(MockError(500), config) is True
        assert is_retryable(MockError(404), config) is False

    def test_retryable_keyword_matching(self):
        """Test retryable matching by keyword in message."""
        config = RetryConfig()

        assert is_retryable(Exception("connection timeout"), config) is True
        assert is_retryable(Exception("network error"), config) is True
        assert is_retryable(Exception("temporarily unavailable"), config) is True


class TestRetryContext:
    """Tests for RetryContext class."""

    def test_creation(self):
        """Test creating RetryContext."""
        config = RetryConfig()
        ctx = RetryContext("task1", config)

        assert ctx.task_id == "task1"
        assert ctx.config == config
        assert ctx.reporter is None
        assert ctx.stats.attempts == 0

    def test_can_retry_not_cancelled(self):
        """Test can_retry when not cancelled."""
        config = RetryConfig(max_retries=3)
        ctx = RetryContext("task1", config)

        assert ctx.can_retry is True

    def test_can_retry_cancelled(self):
        """Test can_retry when cancelled."""
        config = RetryConfig()
        ctx = RetryContext("task1", config)
        ctx.cancel()

        assert ctx.can_retry is False

    def test_can_retry_max_attempts(self):
        """Test can_retry when max attempts reached."""
        config = RetryConfig(max_retries=2)
        ctx = RetryContext("task1", config)
        ctx.stats.attempts = 3

        assert ctx.can_retry is False

    def test_attempt_property(self):
        """Test attempt property."""
        config = RetryConfig()
        ctx = RetryContext("task1", config)
        ctx.stats.attempts = 5

        assert ctx.attempt == 5


class TestWithRetryAsync:
    """Tests for with_retry_async function."""

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        """Test successful call on first try."""
        async def success_func():
            return "result"

        result = await with_retry_async(success_func, task_id="test")

        assert result == "result"

    @pytest.mark.asyncio
    async def test_failure_then_success(self):
        """Test failure then success after retry."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("try again")
            return "success"

        result = await with_retry_async(flaky_func, task_id="test")

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """Test when all retries are exhausted."""
        async def always_fail():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            await with_retry_async(always_fail, task_id="test")


class TestWithRetrySync:
    """Tests for with_retry_sync function."""

    def test_success_first_try(self):
        """Test successful call on first try."""
        def success_func():
            return "result"

        result = with_retry_sync(success_func, task_id="test")

        assert result == "result"

    def test_failure_then_success(self):
        """Test failure then success after retry."""
        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("try again")
            return "success"

        result = with_retry_sync(flaky_func, task_id="test")

        assert result == "success"
        assert call_count == 2

    def test_all_retries_exhausted(self):
        """Test when all retries are exhausted."""
        def always_fail():
            raise ConnectionError("always fails")

        with pytest.raises(ConnectionError):
            with_retry_sync(always_fail, task_id="test")


class TestRetryDecorators:
    """Tests for retry_async and retry_sync decorators."""

    @pytest.mark.asyncio
    async def test_retry_async_decorator(self):
        """Test retry_async decorator."""
        call_count = 0

        @retry_async(task_id="decorated")
        async def flaky_async():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("retry")
            return "success"

        result = await flaky_async()

        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_sync_decorator(self):
        """Test retry_sync decorator."""
        call_count = 0

        @retry_sync(task_id="decorated")
        def flaky_sync():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("retry")
            return "success"

        result = flaky_sync()

        assert result == "success"
