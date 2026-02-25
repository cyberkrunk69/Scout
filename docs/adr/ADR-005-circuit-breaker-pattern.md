# ADR-005: Circuit Breaker Pattern

**Date:** 2025-02-25
**Status:** Accepted
**Deciders:** Scout Architecture Team
**Related:** ADR-001, ADR-008, ADR-010
**Last Updated:** 2026-02-25 (Consolidation complete)

## Test Status

| Test Suite | Tests | Status |
|------------|-------|--------|
| `tests/scout/test_circuit_breaker.py` | 26 | ✅ PASS |
| `tests/scout/llm/test_circuit_breaker.py` | 14 | ✅ PASS |

**Total:** 40 tests, all passing.

---

## Context

Scout relies on external LLM providers (Anthropic, Google, Groq, MiniMax, etc.) for executing AI-powered tasks. These providers can experience:

- **Transient failures**: Temporary network issues, rate limits, or server errors
- **Systemic outages**: Provider-wide problems affecting all requests
- **Regional issues**: Geo-blocking or regional availability problems

Without protection, a failing provider can cause:
- Cascading failures across the system
- Wasted retries on obviously unavailable services
- Poor user experience with long wait times before failure
- Potential cost implications from repeated failed calls

The Provider Registry (ADR-001) handles per-key health, but we also need provider-level protection to handle systemic failures.

## Decision

### Unified Circuit Breaker (`src/scout/circuit_breaker.py`)

The circuit breaker implementations have been **consolidated into a single implementation** in `src/scout/circuit_breaker.py`. This unified version supports both general operations and LLM provider-level failures through configurable presets.

```python
from scout.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    get_breaker,
    record_success,
    record_failure,
)

# For LLM providers (longer cooldown, permanent failure detection)
config = CircuitBreakerConfig.for_provider()

# For general operations (faster recovery)
config = CircuitBreakerConfig.for_operations()
```

### Configuration Presets

```python
@dataclass
class CircuitBreakerConfig:
    # Common settings
    failure_threshold: int          # Failures before opening (default: 5)
    success_threshold: int         # Successes to close (default: 2)
    timeout: float                 # Seconds before half-open
    half_open_max_calls: int       # Max calls in half-open
    cooldown_seconds: float        # Cooldown time for provider-level
    permanent_failure_threshold: Optional[int]  # For permanent failures
```

**Provider Config** (`CircuitBreakerConfig.for_provider()`):
- `timeout`: 300 seconds (5 minutes) - longer for expensive API calls
- `half_open_max_calls`: 1 - single test request
- `permanent_failure_threshold`: 10 - auto-detect permanent failures

**Operations Config** (`CircuitBreakerConfig.for_operations()`):
- `timeout`: 30 seconds - faster recovery
- `half_open_max_calls`: 3 - multiple test requests
- `permanent_failure_threshold`: None - no auto permanent failure

### Configuration via Environment

All values are now configurable via environment variables or the config module:

```bash
# General settings
CIRCUIT_FAILURE_THRESHOLD=5
CIRCUIT_SUCCESS_THRESHOLD=2
CIRCUIT_TIMEOUT=30.0
CIRCUIT_HALF_OPEN_MAX=3

# Provider-level settings
CIRCUIT_COOLDOWN_SECONDS=300
CIRCUIT_PERMANENT_FAILURE_THRESHOLD=10
```

Or via the config module:

```python
from scout.config import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD,
    CIRCUIT_BREAKER_TIMEOUT_SECONDS,
    CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS,
    CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS,
)
```

## Consequences

### Positive

- **Single implementation**: No more duplication or inconsistent behavior
- **Fast failure**: Open circuit rejects requests immediately (milliseconds vs seconds of timeout)
- **Automatic recovery**: Half-open state tests if provider recovered
- **Observability**: State changes logged with structured events
- **Resilience**: Prevents thundering herd on provider recovery
- **Cost savings**: Reduces wasted API calls to failing providers
- **Permanent failure detection**: Automatically detects auth errors, quota exceeded, geo-blocking
- **Flexible configuration**: Presets for different use cases (providers vs operations)

### Mitigations

- Conservative defaults work for most cases
- Half-open state prevents permanent blocking
- Status endpoint exposes circuit state for monitoring
- Manual reset available via `reset_all()`

### Integration Points

- **Provider Registry (ADR-001)**: Circuit breaker checked before key rotation
- **Batch Pipeline (ADR-008)**: Integrates with batch execution
- **Retry Mechanisms (ADR-010)**: Coordinates with retry logic

## Implementation Notes

### Code References

- Unified CB: `src/scout/circuit_breaker.py`
- Deprecated alias (backward compatibility): `src/scout/llm/circuit_breaker.py`
- Tests: `tests/scout/test_circuit_breaker.py`, `tests/scout/llm/test_circuit_breaker.py`
- Config defaults: `src/scout/config/defaults.py`

### Usage Example

```python
# For LLM providers (recommended)
from scout.circuit_breaker import get_breaker, record_success, record_failure

breaker = get_breaker("anthropic")

if breaker.is_available:
    try:
        result = await call_provider()
        record_success("anthropic")
    except Exception as e:
        record_failure("anthropic", permanent=is_permanent_error(e))
else:
    raise CircuitOpenError("Provider unavailable")

# For async operations
from scout.circuit_breaker import get_circuit_breaker_manager

manager = get_circuit_breaker_manager()
breaker = manager.get_breaker("operation_name")

result = await breaker.call(some_async_function)
```

### Monitoring

Circuit breaker status exposed via:
- `get_breaker(name).get_status()` - Individual breaker status
- `get_circuit_breaker_manager().get_all_status()` - All breakers aggregated

Status includes: state, failure_count, success_count, half_open_calls, last_failure_time, opened_at, is_permanently_failed

### Backward Compatibility

The `scout.llm.circuit_breaker` module is maintained for backward compatibility but now re-exports from the unified implementation:

```python
# Old import (deprecated, shows warning)
from scout.llm.circuit_breaker import get_breaker

# New import (recommended)
from scout.circuit_breaker import get_breaker
```

## Related ADRs

- [ADR-001](./ADR-001-provider-registry.md) - Key-level health tracking (complementary)
- [ADR-008](./ADR-008-batch-pipeline.md) - Batch pipeline integration
- [ADR-010](./ADR-010-retry-mechanisms.md) - Coordinates with retry logic

## Notes

### Technical Debt Resolution

As of 2026-02-25, the following technical debt items have been resolved:

| Ticket | Description | Status |
|--------|-------------|--------|
| #TECH-DEBT-001 | Consolidate circuit breaker implementations | ✅ RESOLVED |
| #TECH-DEBT-002 | Move magic numbers to config | ✅ RESOLVED |

### Configuration Values (All Externalized)

All magic numbers are now in `src/scout/config/defaults.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | 5 | Failures before opening |
| `CIRCUIT_BREAKER_SUCCESS_THRESHOLD` | 2 | Successes to close in half-open |
| `CIRCUIT_BREAKER_TIMEOUT_SECONDS` | 30.0 | Seconds before half-open (operations) |
| `CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS` | 3 | Max calls in half-open state |
| `CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS` | 300 | Seconds before half-open (providers) |

### Future Considerations

- Persist circuit state across restarts (currently in-memory only)
- Add metrics/alerting on circuit state changes
- Consider provider-specific configurations (some providers are more reliable)
- Add circuit breaker for tool-level failures (e.g., git operations)
