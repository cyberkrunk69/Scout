"""
Tool Circuit Breaker Registry - Per-tool failure tracking with error categorization.

Provides:
- ToolCircuitBreakerRegistry: Manages circuit breakers per tool name
- Error categorization: Fatal vs non-fatal errors

Phase 3, Task 3 of Unified Tool Framework.

Note: ProviderCircuitBreaker is a stub that wraps the basic CircuitBreaker.
"""

from __future__ import annotations

from typing import Any, Optional

from scout.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

# Error categories that trip the circuit breaker
FATAL_ERROR_CATEGORIES = {"execution", "timeout", "llm_error", "budget_exceeded"}

# Error categories that should NOT trip the circuit breaker
NON_FATAL_CATEGORIES = {"validation", "gate_rejected", "user_cancelled"}


class ProviderCircuitBreaker:
    """Wrapper around CircuitBreaker to provide tool-specific interface."""
    
    def __init__(self, tool_name: str, config: Optional[CircuitBreakerConfig] = None):
        self.tool_name = tool_name
        self._breaker = CircuitBreaker(tool_name, config)
        self.permanently_failed = False
    
    @property
    def state(self) -> str:
        return self._breaker.state.value
    
    @state.setter
    def state(self, value: str) -> None:
        from scout.circuit_breaker import CircuitState
        self._breaker._state = CircuitState(value)
    
    @property
    def failure_count(self) -> int:
        return self._breaker._failure_count
    
    @failure_count.setter
    def failure_count(self, value: int) -> None:
        self._breaker._failure_count = value
    
    def is_available(self) -> bool:
        return self._breaker.state.value == "closed"
    
    def record_success(self) -> None:
        self._breaker.record_success()
        self.permanently_failed = False
    
    def record_failure(self, permanent: bool = False) -> None:
        self._breaker.record_failure()
        if permanent:
            self.permanently_failed = True
    
    def get_state(self) -> dict:
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "permanently_failed": self.permanently_failed,
        }


class ToolCircuitBreakerRegistry:
    """Registry of circuit breakers per tool with error categorization.

    Manages a ProviderCircuitBreaker instance for each tool name.
    Only fatal error categories trip the circuit breaker.
    """

    _breakers: dict[str, ProviderCircuitBreaker] = {}

    @classmethod
    def get_breaker(cls, tool_name: str) -> ProviderCircuitBreaker:
        """Get or create circuit breaker for a tool."""
        if tool_name not in cls._breakers:
            cls._breakers[tool_name] = ProviderCircuitBreaker(tool_name)
        return cls._breakers[tool_name]

    @classmethod
    def is_available(cls, tool_name: str) -> bool:
        """Check if tool is available (circuit not open)."""
        return cls.get_breaker(tool_name).is_available()

    @classmethod
    def record_success(cls, tool_name: str) -> None:
        """Record successful execution."""
        cls.get_breaker(tool_name).record_success()

    @classmethod
    def record_failure(cls, tool_name: str, error_category: str = "execution") -> None:
        """Record failed execution. Only fatal categories trip the breaker.

        Args:
            tool_name: Name of the tool that failed
            error_category: Category of the error (execution, timeout, validation, etc.)
        """
        is_fatal = error_category in FATAL_ERROR_CATEGORIES
        cls.get_breaker(tool_name).record_failure(permanent=is_fatal)

    @classmethod
    def get_state(cls, tool_name: str) -> dict:
        """Get circuit breaker state for a tool."""
        return cls.get_breaker(tool_name).get_state()

    @classmethod
    def reset(cls, tool_name: str) -> None:
        """Reset circuit breaker for a tool."""
        if tool_name in cls._breakers:
            cls._breakers[tool_name].state = "CLOSED"
            cls._breakers[tool_name].failure_count = 0
            cls._breakers[tool_name].permanently_failed = False

    @classmethod
    def clear_all(cls) -> None:
        """Clear all circuit breakers (for testing)."""
        cls._breakers.clear()
