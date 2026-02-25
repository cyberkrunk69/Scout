"""Tests for Circuit Breaker (unified implementation).

These tests verify the circuit breaker functionality after consolidation.
The LLM circuit breaker now re-exports from the unified scout.circuit_breaker module.
"""

import pytest
import time

from scout.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerConfig,
    get_breaker,
    is_provider_available,
    record_success,
    record_failure,
    CircuitOpenError,
)


def test_circuit_breaker_initial_state():
    """Test circuit breaker starts in CLOSED state."""
    breaker = CircuitBreaker("test_provider")

    assert breaker.name == "test_provider"
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0
    assert breaker.is_available is True


def test_circuit_breaker_record_success():
    """Test recording success resets failure count."""
    # Create breaker with provider config (for longer cooldowns)
    breaker = CircuitBreaker("test", CircuitBreakerConfig.for_provider())

    # Record some failures
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_failure()

    assert breaker.failure_count == 3

    breaker.record_success()

    assert breaker.failure_count == 0
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_record_failure():
    """Test recording failures increments count."""
    breaker = CircuitBreaker("test")

    breaker.record_failure()
    assert breaker.failure_count == 1

    breaker.record_failure()
    assert breaker.failure_count == 2


def test_circuit_breaker_opens_after_threshold():
    """Test circuit opens after failure threshold."""
    # Use a config with low threshold for testing
    config = CircuitBreakerConfig(failure_threshold=3)
    breaker = CircuitBreaker("test", config)

    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED

    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.is_available is False


def test_circuit_breaker_permanent_failure():
    """Test permanent failure immediately opens circuit."""
    breaker = CircuitBreaker("test")

    breaker.record_failure(permanent=True)

    assert breaker.state == CircuitState.OPEN
    assert breaker.is_available is False


def test_circuit_breaker_half_open_transition():
    """Test circuit transitions to HALF_OPEN after timeout."""
    # Use a config with very short timeout for testing
    config = CircuitBreakerConfig(failure_threshold=1, timeout=0.01)
    breaker = CircuitBreaker("test", config)

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    # Wait for timeout
    time.sleep(0.05)

    # Should transition to HALF_OPEN
    is_available = breaker.is_available
    assert breaker.state == CircuitState.HALF_OPEN
    assert is_available is True  # First request allowed in HALF_OPEN


def test_circuit_breaker_recovery():
    """Test circuit closes after successful request in HALF_OPEN."""
    # Use a config with very short timeout for testing
    config = CircuitBreakerConfig(failure_threshold=1, timeout=0.01, success_threshold=1)
    breaker = CircuitBreaker("test", config)

    breaker.record_failure()
    time.sleep(0.05)
    _ = breaker.is_available  # Transition to HALF_OPEN

    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.failure_count == 0


def test_circuit_breaker_failure_in_half_open():
    """Test circuit reopens after failure in HALF_OPEN."""
    # Use a config with very short timeout for testing
    config = CircuitBreakerConfig(failure_threshold=1, timeout=0.01)
    breaker = CircuitBreaker("test", config)

    breaker.record_failure()
    time.sleep(0.05)
    _ = breaker.is_available  # Transition to HALF_OPEN

    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN


def test_get_breaker_singleton():
    """Test get_breaker returns same instance."""
    # Import the module-level registry
    import scout.circuit_breaker as cb_module

    # Clear the registry first
    cb_module._circuits.clear()

    breaker1 = get_breaker("test_provider")
    breaker2 = get_breaker("test_provider")

    assert breaker1 is breaker2


def test_is_provider_available():
    """Test is_provider_available helper."""
    # Import the module-level registry
    import scout.circuit_breaker as cb_module

    # Clear the registry first
    cb_module._circuits.clear()

    # Should return True for new provider (not yet in registry)
    result = is_provider_available("nonexistent")
    assert result is True


def test_record_success_failure_helpers():
    """Test record_success and record_failure helpers."""
    # Import the module-level registry
    import scout.circuit_breaker as cb_module

    # Clear the registry first
    cb_module._circuits.clear()

    record_success("test_provider")
    record_failure("test_provider")

    breaker = get_breaker("test_provider")
    assert breaker.failure_count == 1


def test_circuit_state_enum():
    """Test CircuitState enum values."""
    assert CircuitState.CLOSED.value == "closed"
    assert CircuitState.OPEN.value == "open"
    assert CircuitState.HALF_OPEN.value == "half_open"


def test_provider_config_has_permanent_failure():
    """Test that provider config enables permanent failure detection."""
    config = CircuitBreakerConfig.for_provider()

    assert config.permanent_failure_threshold is not None
    assert config.permanent_failure_threshold == 10
    assert config.timeout == 300  # Provider config uses longer timeout


def test_operations_config_no_permanent_failure():
    """Test that operations config doesn't have permanent failure detection."""
    config = CircuitBreakerConfig.for_operations()

    assert config.permanent_failure_threshold is None
    assert config.timeout == 30.0  # Operations use shorter timeout
