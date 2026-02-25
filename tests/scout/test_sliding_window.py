"""Tests for sliding_window.py module."""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock

from scout.sliding_window import (
    RateLimitConfig,
    MiniMaxRateLimitConfig,
    TokenUsage,
    SlidingWindowRateLimiter,
    MultiProviderRateLimiter,
    get_rate_limiter,
    RateLimitError,
)


class TestRateLimitConfig:
    """Tests for RateLimitConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = RateLimitConfig()

        assert config.requests_per_minute == 60
        assert config.tokens_per_minute == 100000

    def test_for_deepseek(self):
        """Test DeepSeek-specific config."""
        config = RateLimitConfig.for_deepseek()

        assert config.requests_per_minute == 50
        assert config.tokens_per_minute == 2_000_000

    def test_for_minimax(self):
        """Test MiniMax-specific config."""
        config = RateLimitConfig.for_minimax()

        assert isinstance(config, MiniMaxRateLimitConfig)
        assert config.requests_per_minute == 100
        assert config.tokens_per_minute == 200_000


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_creation(self):
        """Test TokenUsage creation."""
        usage = TokenUsage(timestamp=1234567890.0, tokens=100)

        assert usage.timestamp == 1234567890.0
        assert usage.tokens == 100


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter class."""

    @pytest.mark.asyncio
    async def test_acquire_first_request(self):
        """Test acquiring first request."""
        config = RateLimitConfig(requests_per_minute=10)
        limiter = SlidingWindowRateLimiter(config)

        result = await limiter.acquire(tokens=0)

        assert result is True
        assert len(limiter._request_times) == 1

    @pytest.mark.asyncio
    async def test_acquire_with_tokens(self):
        """Test acquiring with token count."""
        config = RateLimitConfig(tokens_per_minute=1000)
        limiter = SlidingWindowRateLimiter(config)

        result = await limiter.acquire(tokens=100)

        assert result is True
        assert len(limiter._token_usage) == 1
        assert limiter._token_usage[0].tokens == 100

    @pytest.mark.asyncio
    async def test_acquire_request_limit_exceeded_no_wait(self):
        """Test acquire when request limit exceeded without wait."""
        config = RateLimitConfig(requests_per_minute=1)
        limiter = SlidingWindowRateLimiter(config)

        # First request
        await limiter.acquire(tokens=0)

        # Second request should fail without wait
        result = await limiter.acquire(tokens=0, wait=False)

        assert result is False

    @pytest.mark.asyncio
    async def test_acquire_token_limit_exceeded_no_wait(self):
        """Test acquire when token limit exceeded without wait."""
        config = RateLimitConfig(tokens_per_minute=100)
        limiter = SlidingWindowRateLimiter(config)

        # First request uses all tokens
        await limiter.acquire(tokens=100)

        # Second request should fail
        result = await limiter.acquire(tokens=10, wait=False)

        assert result is False

    def test_get_status(self):
        """Test getting status."""
        config = RateLimitConfig(requests_per_minute=10, tokens_per_minute=1000)
        limiter = SlidingWindowRateLimiter(config)

        status = limiter.get_status()

        assert "requests_this_minute" in status
        assert "requests_limit" in status
        assert "requests_remaining" in status
        assert "tokens_this_minute" in status
        assert "tokens_limit" in status
        assert "tokens_remaining" in status
        assert status["requests_limit"] == 10
        assert status["tokens_limit"] == 1000

    def test_reset(self):
        """Test resetting the limiter."""
        config = RateLimitConfig()
        limiter = SlidingWindowRateLimiter(config)

        # Add some fake entries
        limiter._request_times.append(time.time())
        limiter._token_usage.append(TokenUsage(time.time(), 100))

        limiter.reset()

        assert len(limiter._request_times) == 0
        assert len(limiter._token_usage) == 0


class TestMultiProviderRateLimiter:
    """Tests for MultiProviderRateLimiter class."""

    def test_get_limiter_creates_new(self):
        """Test that get_limiter creates new limiter."""
        multi = MultiProviderRateLimiter()

        limiter = multi.get_limiter("deepseek")

        assert isinstance(limiter, SlidingWindowRateLimiter)

    def test_get_limiter_returns_same(self):
        """Test that get_limiter returns same instance."""
        multi = MultiProviderRateLimiter()

        limiter1 = multi.get_limiter("deepseek")
        limiter2 = multi.get_limiter("deepseek")

        assert limiter1 is limiter2

    def test_get_limiter_deepseek_config(self):
        """Test DeepSeek config is applied."""
        multi = MultiProviderRateLimiter()

        limiter = multi.get_limiter("deepseek")

        assert limiter.config.requests_per_minute == 50

    def test_get_limiter_minimax_config(self):
        """Test MiniMax config is applied."""
        multi = MultiProviderRateLimiter()

        limiter = multi.get_limiter("minimax")

        assert limiter.config.requests_per_minute == 100

    def test_get_limiter_unknown_provider(self):
        """Test unknown provider uses default config."""
        multi = MultiProviderRateLimiter()

        limiter = multi.get_limiter("unknown")

        # Default is for_deepseek() which has 50 requests_per_minute
        assert limiter.config.requests_per_minute == 50

    def test_get_all_status(self):
        """Test getting status of all limiters."""
        multi = MultiProviderRateLimiter()

        multi.get_limiter("deepseek")
        multi.get_limiter("minimax")

        status = multi.get_all_status()

        assert "deepseek" in status
        assert "minimax" in status

    def test_reset_all(self):
        """Test resetting all limiters."""
        multi = MultiProviderRateLimiter()

        limiter = multi.get_limiter("deepseek")
        limiter._request_times.append(time.time())

        multi.reset_all()

        assert len(limiter._request_times) == 0


class TestGetRateLimiter:
    """Tests for get_rate_limiter function."""

    def test_singleton_pattern(self):
        """Test that get_rate_limiter returns singleton."""
        # Reset global
        import scout.sliding_window as sw_module
        sw_module._rate_limiter = None

        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()

        assert limiter1 is limiter2


class TestRateLimitError:
    """Tests for RateLimitError exception."""

    def test_exception(self):
        """Test RateLimitError can be raised."""
        with pytest.raises(RateLimitError):
            raise RateLimitError("Rate limit exceeded")
