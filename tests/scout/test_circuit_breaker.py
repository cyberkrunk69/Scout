"""Tests for circuit_breaker.py module."""

import pytest
import asyncio
import time
import os
from unittest.mock import patch, MagicMock, AsyncMock

from scout.circuit_breaker import (
    CircuitState,
    CircuitBreakerOpen,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitBreakerManager,
    get_circuit_breaker_manager,
)


class TestCircuitState:
    """Tests for CircuitState enum."""

    def test_states(self):
        """Test CircuitState values."""
        assert CircuitState.CLOSED.value == "closed"
        assert CircuitState.OPEN.value == "open"
        assert CircuitState.HALF_OPEN.value == "half_open"


class TestCircuitBreakerOpen:
    """Tests for CircuitBreakerOpen exception."""

    def test_exception(self):
        """Test CircuitBreakerOpen can be raised."""
        with pytest.raises(CircuitBreakerOpen):
            raise CircuitBreakerOpen("Circuit is open")

    def test_exception_message(self):
        """Test exception message."""
        err = CircuitBreakerOpen("Test circuit open")
        assert "Test circuit open" in str(err)


class TestCircuitBreakerConfig:
    """Tests for CircuitBreakerConfig class."""

    def test_default_values(self):
        """Test default config values."""
        config = CircuitBreakerConfig()

        assert config.failure_threshold == 5
        assert config.success_threshold == 2
        assert config.timeout == 30.0
        assert config.half_open_max_calls == 3

    def test_custom_values(self):
        """Test custom config values."""
        config = CircuitBreakerConfig(
            failure_threshold=10,
            success_threshold=3,
            timeout=60.0,
            half_open_max_calls=5,
        )

        assert config.failure_threshold == 10
        assert config.success_threshold == 3

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with defaults."""
        config = CircuitBreakerConfig.from_env()

        assert config.failure_threshold == 5

    @patch.dict(os.environ, {
        "CIRCUIT_FAILURE_THRESHOLD": "10",
        "CIRCUIT_SUCCESS_THRESHOLD": "3",
        "CIRCUIT_TIMEOUT": "60.0",
        "CIRCUIT_HALF_OPEN_MAX": "5",
    }, clear=True)
    def test_from_env_with_values(self):
        """Test from_env with environment variables."""
        config = CircuitBreakerConfig.from_env()

        assert config.failure_threshold == 10
        assert config.success_threshold == 3
        assert config.timeout == 60.0
        assert config.half_open_max_calls == 5


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_creation(self):
        """Test creating a CircuitBreaker."""
        breaker = CircuitBreaker("test")

        assert breaker.name == "test"
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker._success_count == 0

    def test_state_property(self):
        """Test state property."""
        breaker = CircuitBreaker("test")

        assert breaker.state == CircuitState.CLOSED

    def test_is_available_closed(self):
        """Test is_available when closed."""
        breaker = CircuitBreaker("test")

        assert breaker.is_available is True

    def test_is_available_open_timeout_not_passed(self):
        """Test is_available when open and timeout not passed."""
        breaker = CircuitBreaker("test")
        breaker._state = CircuitState.OPEN
        breaker._last_failure_time = time.time()  # Just now

        assert breaker.is_available is False

    def test_is_available_open_timeout_passed(self):
        """Test is_available when open but timeout passed."""
        breaker = CircuitBreaker("test")
        breaker._state = CircuitState.OPEN
        breaker._last_failure_time = time.time() - 31  # More than 30s ago

        assert breaker.is_available is True
        assert breaker.state == CircuitState.HALF_OPEN

    def test_is_available_half_open(self):
        """Test is_available in half-open state."""
        breaker = CircuitBreaker("test")
        breaker._state = CircuitState.HALF_OPEN
        breaker._half_open_calls = 2

        assert breaker.is_available is True

    def test_is_available_half_open_max_calls(self):
        """Test is_available when half-open max calls reached."""
        breaker = CircuitBreaker("test")
        breaker._state = CircuitState.HALF_OPEN
        breaker._half_open_calls = 3  # Already at max

        assert breaker.is_available is False

    @pytest.mark.asyncio
    async def test_call_success(self):
        """Test successful call through circuit breaker."""
        breaker = CircuitBreaker("test")

        async def success_func():
            return "success"

        result = await breaker.call(success_func)

        assert result == "success"
        assert breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_call_failure(self):
        """Test failing call through circuit breaker."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))

        async def fail_func():
            raise Exception("Test error")

        with pytest.raises(Exception):
            await breaker.call(fail_func)

        assert breaker._failure_count == 1

    @pytest.mark.asyncio
    async def test_call_opens_circuit(self):
        """Test that repeated failures open the circuit."""
        breaker = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=2))

        async def fail_func():
            raise Exception("Test error")

        # First failure
        with pytest.raises(Exception):
            await breaker.call(fail_func)

        # Second failure - should open circuit
        with pytest.raises(Exception):
            await breaker.call(fail_func)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_call_when_open(self):
        """Test calling when circuit is open."""
        breaker = CircuitBreaker("test")
        breaker._state = CircuitState.OPEN
        breaker._last_failure_time = time.time()

        with pytest.raises(CircuitBreakerOpen):
            await breaker.call(lambda: "test")

    def test_get_status(self):
        """Test getting status."""
        breaker = CircuitBreaker("test")
        breaker._failure_count = 3

        status = breaker.get_status()

        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["failure_count"] == 3

    def test_reset(self):
        """Test manual reset."""
        breaker = CircuitBreaker("test")
        breaker._state = CircuitState.OPEN
        breaker._failure_count = 10

        breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0


class TestCircuitBreakerManager:
    """Tests for CircuitBreakerManager class."""

    def test_creation(self):
        """Test creating a manager."""
        manager = CircuitBreakerManager()

        assert manager._config is not None
        assert manager._breakers == {}

    def test_get_breaker_creates_new(self):
        """Test that get_breaker creates new breaker."""
        manager = CircuitBreakerManager()

        breaker = manager.get_breaker("provider1")

        assert isinstance(breaker, CircuitBreaker)
        assert breaker.name == "provider1"

    def test_get_breaker_returns_same(self):
        """Test that get_breaker returns same instance."""
        manager = CircuitBreakerManager()

        breaker1 = manager.get_breaker("provider1")
        breaker2 = manager.get_breaker("provider1")

        assert breaker1 is breaker2

    def test_get_all_status(self):
        """Test getting all status."""
        manager = CircuitBreakerManager()

        manager.get_breaker("provider1")
        manager.get_breaker("provider2")

        status = manager.get_all_status()

        assert "provider1" in status
        assert "provider2" in status

    def test_reset_all(self):
        """Test resetting all breakers."""
        manager = CircuitBreakerManager()

        manager.get_breaker("provider1")
        manager.get_breaker("provider2")

        manager.reset_all()

        assert len(manager._breakers) == 0


class TestGetCircuitBreakerManager:
    """Tests for get_circuit_breaker_manager function."""

    def test_singleton(self):
        """Test that get_circuit_breaker_manager returns singleton."""
        # Reset global
        import scout.circuit_breaker as cb_module
        cb_module._circuit_breaker_manager = None

        manager1 = get_circuit_breaker_manager()
        manager2 = get_circuit_breaker_manager()

        assert manager1 is manager2
