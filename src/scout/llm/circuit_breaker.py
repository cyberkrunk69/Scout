"""Shared circuit breaker for LLM providers.

DEPRECATED: This module is maintained for backward compatibility.
The circuit breaker functionality has been consolidated into:
    src/scout/circuit_breaker.py

Please update imports to use the unified implementation:
    from scout.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerConfig,
        CircuitBreakerManager,
        CircuitState,
        CircuitOpenError,
        get_breaker,
        is_provider_available,
        record_success,
        record_failure,
    )

This module will be removed in a future version.
"""

import warnings

# Issue deprecation warning when this module is imported
warnings.warn(
    "scout.llm.circuit_breaker is deprecated. "
    "Import from scout.circuit_breaker instead.",
    DeprecationWarning,
    stacklevel=2,
)

from scout.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitOpenError,
    CircuitState,
    get_breaker,
    get_circuit_breaker_manager,
    is_provider_available,
    record_failure,
    record_success,
)

# Backward compatibility - re-export everything
__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerManager",
    "CircuitOpenError",
    "CircuitState",
    "get_breaker",
    "get_circuit_breaker_manager",
    "is_provider_available",
    "record_failure",
    "record_success",
]
