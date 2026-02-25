"""Circuit breaker pattern for provider failures.

Implements a circuit breaker to prevent repeated calls to failing providers.
This is the unified implementation that supports both general operations and
LLM provider-level failures.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

from scout.config.defaults import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
    CIRCUIT_BREAKER_TIMEOUT_SECONDS,
    CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS,
    CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS,
)

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing - reject calls
    HALF_OPEN = "half_open"  # Testing if recovery


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Number of failures before opening the circuit.
        success_threshold: Number of successes in half-open to close the circuit.
        timeout: Seconds to wait in OPEN state before transitioning to HALF_OPEN.
        half_open_max_calls: Maximum calls allowed in HALF_OPEN state.
        cooldown_seconds: Time to wait before allowing test requests (for provider-level CB).
        permanent_failure_threshold: If set, failures beyond this threshold mark
            the circuit as permanently failed (requires manual reset).
    """

    failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD
    success_threshold: int = CIRCUIT_BREAKER_SUCCESS_THRESHOLD
    timeout: float = CIRCUIT_BREAKER_TIMEOUT_SECONDS
    half_open_max_calls: int = CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS
    cooldown_seconds: float = CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS
    permanent_failure_threshold: Optional[int] = None

    @classmethod
    def from_env(cls) -> "CircuitBreakerConfig":
        """Create config from environment variables with config defaults as fallbacks."""
        return cls(
            failure_threshold=int(os.environ.get(
                "CIRCUIT_FAILURE_THRESHOLD", str(CIRCUIT_BREAKER_FAILURE_THRESHOLD))),
            success_threshold=int(os.environ.get(
                "CIRCUIT_SUCCESS_THRESHOLD", str(CIRCUIT_BREAKER_SUCCESS_THRESHOLD))),
            timeout=float(os.environ.get(
                "CIRCUIT_TIMEOUT", str(CIRCUIT_BREAKER_TIMEOUT_SECONDS))),
            half_open_max_calls=int(os.environ.get(
                "CIRCUIT_HALF_OPEN_MAX", str(CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS))),
            cooldown_seconds=float(os.environ.get(
                "CIRCUIT_COOLDOWN_SECONDS", str(CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS))),
            permanent_failure_threshold=int(os.environ["CIRCUIT_PERMANENT_FAILURE_THRESHOLD"])
                if "CIRCUIT_PERMANENT_FAILURE_THRESHOLD" in os.environ else None,
        )

    @classmethod
    def for_provider(cls) -> "CircuitBreakerConfig":
        """Create config optimized for LLM provider-level failures.

        Uses longer cooldown since provider failures are more expensive
        and typically longer-lasting than operation-level failures.
        """
        return cls(
            failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            success_threshold=CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
            timeout=CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS,  # Use longer timeout for providers
            half_open_max_calls=1,  # Only one test request for providers
            cooldown_seconds=CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS,
            permanent_failure_threshold=10,  # Mark as permanently failed after 10 failures
        )

    @classmethod
    def for_operations(cls) -> "CircuitBreakerConfig":
        """Create config optimized for general operations.

        Uses shorter timeout for faster recovery in operational contexts.
        """
        return cls(
            failure_threshold=CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            success_threshold=CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
            timeout=CIRCUIT_BREAKER_TIMEOUT_SECONDS,
            half_open_max_calls=CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS,
            cooldown_seconds=CIRCUIT_BREAKER_TIMEOUT_SECONDS,
            permanent_failure_threshold=None,  # No permanent failure for operations
        )


class CircuitBreaker:
    """Circuit breaker for provider calls.

    Supports both operation-level and provider-level circuit breakers with
    configurable behavior via CircuitBreakerConfig.

    For provider-level failures (expensive API calls), use CircuitBreakerConfig.for_provider()
    which has longer cooldowns and permanent failure detection.

    For general operations, use CircuitBreakerConfig.for_operations() or the defaults.
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0
        self._opened_at: Optional[float] = None
        self._permanently_failed = False

        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_permanently_failed(self) -> bool:
        """Check if circuit is permanently failed (requires manual reset)."""
        return self._permanently_failed

    @property
    def is_available(self) -> bool:
        """Check if the circuit allows calls."""
        # Permanently failed circuits never allow calls
        if self._permanently_failed:
            return False

        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            # Check if timeout period has passed
            # Use timeout for standard transition; cooldown_seconds is for provider-level
            check_timeout = self.config.timeout
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= check_timeout:
                    self._transition_to_half_open()
                    return True
            return False

        # HALF_OPEN state
        return self._half_open_calls < self.config.half_open_max_calls

    def _transition_to_half_open(self) -> None:
        """Transition from OPEN to HALF_OPEN state."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        logger.info(
            "circuit_breaker_state_change",
            extra={
                "event": "circuit_breaker_state_change",
                "provider": self.name,
                "from_state": "OPEN",
                "to_state": "HALF_OPEN",
                "cooldown_seconds": self.config.cooldown_seconds,
            }
        )

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function through the circuit breaker."""
        async with self._lock:
            if not self.is_available:
                raise CircuitBreakerOpen(
                    f"Circuit '{self.name}' is OPEN. Provider unavailable."
                )

            # Track half-open calls
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            await self._on_success()
            return result

        except Exception as e:
            await self._on_failure()
            raise

    async def _on_success(self) -> None:
        """Handle successful call."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                logger.debug(
                    f"Circuit '{self.name}': success {self._success_count}/"
                    f"{self.config.success_threshold}"
                )

                if self._success_count >= self.config.success_threshold:
                    logger.info(f"Circuit '{self.name}': CLOSING circuit")
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._permanently_failed = False
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
                self._permanently_failed = False

    async def _on_failure(self, permanent: bool = False) -> None:
        """Handle failed call.

        Args:
            permanent: If True, marks the circuit as permanently failed,
                       requiring manual reset. Useful for auth errors,
                       quota exceeded, or geo-blocking.
        """
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                logger.warning(
                    "circuit_breaker_state_change",
                    extra={
                        "event": "circuit_breaker_state_change",
                        "provider": self.name,
                        "from_state": "HALF_OPEN",
                        "to_state": "OPEN",
                        "reason": "test_request_failed",
                    }
                )
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._half_open_calls = 0
                self._opened_at = self._last_failure_time

            elif self._state == CircuitState.CLOSED:
                # Check for permanent failure
                if permanent or (
                    self.config.permanent_failure_threshold is not None
                    and self._failure_count >= self.config.permanent_failure_threshold
                ):
                    self._permanently_failed = True
                    self._state = CircuitState.OPEN
                    self._opened_at = self._last_failure_time
                    logger.warning(
                        "circuit_breaker_state_change",
                        extra={
                            "event": "circuit_breaker_state_change",
                            "provider": self.name,
                            "from_state": "CLOSED",
                            "to_state": "OPEN",
                            "reason": "permanent_failure",
                            "failure_count": self.failure_count,
                            "permanent": True,
                        }
                    )
                elif self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        "circuit_breaker_state_change",
                        extra={
                            "event": "circuit_breaker_state_change",
                            "provider": self.name,
                            "from_state": "CLOSED",
                            "to_state": "OPEN",
                            "failure_count": self.failure_count,
                            "threshold": self.config.failure_threshold,
                            "cooldown_seconds": self.config.cooldown_seconds,
                        }
                    )
                    self._state = CircuitState.OPEN
                    self._opened_at = self._last_failure_time

    # Synchronous methods for direct usage (non-async)

    def record_success(self) -> None:
        """Record a successful call (synchronous version).

        Use this method when calling from synchronous code.
        For async code, prefer using the call() method.
        """
        # Create a simple success event
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                logger.info(
                    "circuit_breaker_state_change",
                    extra={
                        "event": "circuit_breaker_state_change",
                        "provider": self.name,
                        "from_state": "HALF_OPEN",
                        "to_state": "CLOSED",
                        "reason": "recovery_success",
                    }
                )
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                self._permanently_failed = False
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0
            self._permanently_failed = False

    def record_failure(self, permanent: bool = False) -> None:
        """Record a failed call (synchronous version).

        Args:
            permanent: If True, marks the circuit as permanently failed.

        Use this method when calling from synchronous code.
        For async code, prefer using the call() method.
        """
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitState.HALF_OPEN:
            logger.warning(
                "circuit_breaker_state_change",
                extra={
                    "event": "circuit_breaker_state_change",
                    "provider": self.name,
                    "from_state": "HALF_OPEN",
                    "to_state": "OPEN",
                    "reason": "test_request_failed",
                }
            )
            self._state = CircuitState.OPEN
            self._success_count = 0
            self._half_open_calls = 0
            self._opened_at = self._last_failure_time

        elif self._state == CircuitState.CLOSED:
            if permanent or (
                self.config.permanent_failure_threshold is not None
                and self._failure_count >= self.config.permanent_failure_threshold
            ):
                self._permanently_failed = True
                self._state = CircuitState.OPEN
                self._opened_at = self._last_failure_time
                logger.warning(
                    "circuit_breaker_state_change",
                    extra={
                        "event": "circuit_breaker_state_change",
                        "provider": self.name,
                        "from_state": "CLOSED",
                        "to_state": "OPEN",
                        "reason": "permanent_failure" if permanent else "threshold_exceeded",
                        "failure_count": self.failure_count,
                        "permanent": permanent,
                    }
                )
            elif self._failure_count >= self.config.failure_threshold:
                logger.warning(
                    "circuit_breaker_state_change",
                    extra={
                        "event": "circuit_breaker_state_change",
                        "provider": self.name,
                        "from_state": "CLOSED",
                        "to_state": "OPEN",
                        "failure_count": self.failure_count,
                        "threshold": self.config.failure_threshold,
                        "cooldown_seconds": self.config.cooldown_seconds,
                    }
                )
                self._state = CircuitState.OPEN
                self._opened_at = self._last_failure_time

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "half_open_calls": self._half_open_calls,
            "last_failure": self._last_failure_time,
            "opened_at": self._opened_at,
            "is_available": self.is_available,
            "is_permanently_failed": self._permanently_failed,
        }

    def reset(self) -> None:
        """Manually reset the circuit breaker.

        Clears all state including permanent failure flag.
        """
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0
        self._opened_at = None
        self._permanently_failed = False


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different providers.

    Supports creating breakers with different configurations:
    - Use get_breaker() for operation-level breakers (default config)
    - Use get_provider_breaker() for LLM provider-level breakers (longer cooldown)
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._config = config or CircuitBreakerConfig.from_env()

    def get_breaker(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker for a name with optional custom config."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, config or self._config)
        return self._breakers[name]

    def get_provider_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker configured for LLM provider use.

        Uses longer cooldown and permanent failure detection appropriate
        for expensive provider API calls.
        """
        return self.get_breaker(name, CircuitBreakerConfig.for_provider())

    def get_operation_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker configured for general operations.

        Uses shorter timeout for faster recovery in operational contexts.
        """
        return self.get_breaker(name, CircuitBreakerConfig.for_operations())

    def get_all_status(self) -> dict:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()
        self._breakers.clear()


# Global instance
_circuit_breaker_manager: Optional[CircuitBreakerManager] = None
_provider_circuit_breaker_manager: Optional[CircuitBreakerManager] = None


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager for general operations."""
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager(CircuitBreakerConfig.for_operations())
    return _circuit_breaker_manager


def get_provider_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager for LLM providers.

    Uses provider-level configuration with longer cooldowns and
    permanent failure detection.
    """
    global _provider_circuit_breaker_manager
    if _provider_circuit_breaker_manager is None:
        _provider_circuit_breaker_manager = CircuitBreakerManager(
            CircuitBreakerConfig.for_provider()
        )
    return _provider_circuit_breaker_manager


# =============================================================================
# Module-level convenience functions (similar to LLM circuit breaker API)
# =============================================================================

# For backward compatibility, maintain a simple registry
_circuits: dict[str, CircuitBreaker] = {}


def get_breaker(provider: str) -> CircuitBreaker:
    """Get or create circuit breaker for a provider.

    This is a convenience function that uses the provider-level configuration
    (longer cooldown, permanent failure detection).

    For general operations, use get_circuit_breaker_manager().get_breaker() instead.
    """
    global _circuits
    if provider not in _circuits:
        _circuits[provider] = CircuitBreaker(
            provider,
            CircuitBreakerConfig.for_provider()
        )
    return _circuits[provider]


def is_provider_available(provider: str) -> bool:
    """Check if provider is available (circuit not open)."""
    return get_breaker(provider).is_available


def record_success(provider: str):
    """Record a successful request for a provider."""
    get_breaker(provider).record_success()


def record_failure(provider: str, permanent: bool = False):
    """Record a failed request for a provider.

    Args:
        provider: Provider name
        permanent: If True, marks the circuit as permanently failed
                   (for auth errors, quota exceeded, geo-blocking)
    """
    get_breaker(provider).record_failure(permanent=permanent)


# =============================================================================
# Exceptions
# =============================================================================


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and requests are rejected."""
    pass
