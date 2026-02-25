"""Circuit breaker pattern for LLM provider failures.

Implements a circuit breaker to prevent repeated calls to failing providers.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

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
    """Configuration for circuit breaker."""
    
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close
    timeout: float = 30.0        # Seconds before half-open
    half_open_max_calls: int = 3  # Max calls in half-open
    
    @classmethod
    def from_env(cls) -> "CircuitBreakerConfig":
        return cls(
            failure_threshold=int(os.environ.get("CIRCUIT_FAILURE_THRESHOLD", "5")),
            success_threshold=int(os.environ.get("CIRCUIT_SUCCESS_THRESHOLD", "2")),
            timeout=float(os.environ.get("CIRCUIT_TIMEOUT", "30.0")),
            half_open_max_calls=int(os.environ.get("CIRCUIT_HALF_OPEN_MAX", "3")),
        )


class CircuitBreaker:
    """Circuit breaker for provider calls."""
    
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
        
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def is_available(self) -> bool:
        """Check if the circuit allows calls."""
        if self._state == CircuitState.CLOSED:
            return True
        
        if self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if self._last_failure_time:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout:
                    logger.info(f"Circuit '{self.name}': timeout passed, entering half-open")
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    return True
            return False
        
        # HALF_OPEN state
        return self._half_open_calls < self.config.half_open_max_calls
    
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
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0
    
    async def _on_failure(self) -> None:
        """Handle failed call."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                logger.warning(f"Circuit '{self.name}': failure in half-open, REOPENING")
                self._state = CircuitState.OPEN
                self._success_count = 0
                self._half_open_calls = 0
                
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning(
                        f"Circuit '{self.name}': failure threshold reached, OPENING circuit"
                    )
                    self._state = CircuitState.OPEN
    
    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "half_open_calls": self._half_open_calls,
            "last_failure": self._last_failure_time,
            "is_available": self.is_available,
        }
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different providers."""
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._config = CircuitBreakerConfig.from_env()
    
    def get_breaker(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker for a provider."""
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(name, self._config)
        return self._breakers[name]
    
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


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get the global circuit breaker manager."""
    global _circuit_breaker_manager
    if _circuit_breaker_manager is None:
        _circuit_breaker_manager = CircuitBreakerManager()
    return _circuit_breaker_manager
