"""Shared circuit breaker for LLM providers.

Provides a clean interface for circuit breaker functionality that can be used
by both router.py and dispatch.py.
"""

from dataclasses import dataclass
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Circuit Breaker State
# =============================================================================

@dataclass
class CircuitState:
    """Current state of a circuit breaker."""
    provider: str
    state: str  # "CLOSED", "OPEN", "HALF_OPEN"
    failure_count: int
    last_failure_time: float
    opened_at: float


# =============================================================================
# Circuit Breaker Implementation
# =============================================================================

class CircuitBreaker:
    """Circuit breaker for provider-level failures.
    
    States:
    - CLOSED: Normal operation, requests allowed
    - OPEN: Provider failing, requests blocked
    - HALF_OPEN: Testing if provider recovered
    """
    
    STATES = ["CLOSED", "OPEN", "HALF_OPEN"]
    FAILURE_THRESHOLD = 5
    COOLDOWN_SECONDS = 300
    
    def __init__(self, name: str):
        self.name = name
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.opened_at = 0.0
        self._test_request_allowed = False
    
    def is_available(self) -> bool:
        """Check if provider is available (circuit not open)."""
        current_time = time.time()
        
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            if current_time - self.opened_at >= self.COOLDOWN_SECONDS:
                self._transition_to_half_open()
                return self._test_request_allowed
            return False
        
        if self.state == "HALF_OPEN":
            if self._test_request_allowed:
                self._test_request_allowed = False
                return True
            return False
        
        return True
    
    def _transition_to_half_open(self):
        self.state = "HALF_OPEN"
        self._test_request_allowed = True
        logger.info(
            "circuit_breaker_state_change",
            extra={
                "event": "circuit_breaker_state_change",
                "provider": self.name,
                "from_state": "OPEN",
                "to_state": "HALF_OPEN",
                "cooldown_seconds": self.COOLDOWN_SECONDS,
            }
        )
    
    def record_success(self):
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            self._test_request_allowed = False
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
        elif self.state == "CLOSED":
            self.failure_count = 0
    
    def record_failure(self, permanent: bool = False):
        current_time = time.time()
        self.last_failure_time = current_time
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
            self.opened_at = current_time
            self._test_request_allowed = False
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
        elif self.state == "CLOSED":
            if permanent:
                self.failure_count = self.FAILURE_THRESHOLD
                self._open()
            else:
                self.failure_count += 1
                if self.failure_count >= self.FAILURE_THRESHOLD:
                    self._open()
    
    def _open(self):
        self.state = "OPEN"
        self.opened_at = time.time()
        logger.warning(
            "circuit_breaker_state_change",
            extra={
                "event": "circuit_breaker_state_change",
                "provider": self.name,
                "from_state": "CLOSED",
                "to_state": "OPEN",
                "failure_count": self.failure_count,
                "threshold": self.FAILURE_THRESHOLD,
                "cooldown_seconds": self.COOLDOWN_SECONDS,
            }
        )
    
    def get_state(self) -> CircuitState:
        return CircuitState(
            provider=self.name,
            state=self.state,
            failure_count=self.failure_count,
            last_failure_time=self.last_failure_time,
            opened_at=self.opened_at,
        )


# =============================================================================
# Global Registry
# =============================================================================

_circuits: dict[str, CircuitBreaker] = {}


def get_breaker(provider: str) -> CircuitBreaker:
    """Get or create circuit breaker for a provider."""
    if provider not in _circuits:
        _circuits[provider] = CircuitBreaker(provider)
    return _circuits[provider]


def is_provider_available(provider: str) -> bool:
    """Check if provider is available (circuit not open)."""
    return get_breaker(provider).is_available()


def record_success(provider: str):
    """Record a successful request for a provider."""
    get_breaker(provider).record_success()


def record_failure(provider: str, permanent: bool = False):
    """Record a failed request for a provider."""
    get_breaker(provider).record_failure(permanent=permanent)


# =============================================================================
# Exceptions
# =============================================================================

class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass
