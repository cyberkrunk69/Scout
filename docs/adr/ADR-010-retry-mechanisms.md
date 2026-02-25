# ADR-010: Retry Mechanisms

**Date:** 2025-02-25
**Status:** Accepted
**Deciders:** Scout Architecture Team
**Related:** ADR-003, ADR-005, ADR-008

## Test Status

| Test Suite | Tests | Status |
|------------|-------|--------|
| `tests/scout/test_retry.py` | 24 | ✅ PASS |
| `tests/scout/llm/test_retry.py` | 7 | ✅ PASS |

**Total:** 31 tests, all passing.

---

## Context

Scout relies on external services that can fail transiently:
- LLM API calls (network issues, rate limits, server errors)
- Git operations (network timeouts, conflicts)
- File system operations (concurrent modifications)

Without retry logic:
- Single transient failure causes complete task failure
- Poor user experience
- Wasted development time

With naive retries:
- Don't retry permanent errors (auth failures, quota exceeded)
- Retry too aggressively (magnify provider issues)
- No backoff (can cause rate limit hits)

## Decision

Implemented a **multi-layered retry system** with:

### 1. Generic Retry Framework (`src/scout/retry.py`)

For batch operations and general use:

```python
@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay_ms: float = 1000.0
    max_delay_ms: float = 30000.0
    backoff_multiplier: float = 2.0
    jitter: float = 0.1  # 10% jitter
```

**Key Features:**
- **Exponential backoff**: Delay doubles each attempt
- **Jitter**: Random variation prevents thundering herd
- **Progress reporting**: Emits retry events
- **Configurable**: Per-operation tuning

**Retryable Exceptions:**

```python
retryable_exceptions = (
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
    OSError,
)

retryable_status_codes = (429, 500, 502, 503, 504)
```

### 2. LLM-specific Retry (`src/scout/llm/retry.py`)

Specialized for LLM calls with budget integration:

```python
async def call_with_retries(
    provider_call,
    context: LLMCallContext,  # Budget integration
    estimated_cost: float,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
) -> LLMResult:
```

**Key Features:**
- **Budget awareness**: Checks budget before each retry
- **Cost tracking**: Commits actual cost after success
- **Reservation rollback**: Rolls back reservation on failure
- **Audit logging**: Records all retry attempts

### 3. Backoff Calculation

```python
def calculate_backoff(attempt: int, config: RetryConfig) -> float:
    delay = config.base_delay_ms * (config.backoff_multiplier ** attempt)
    delay = min(delay, config.max_delay_ms)
    
    # Add jitter
    jitter_range = delay * config.jitter
    delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)

# Example timeline:
# Attempt 0: 1s + jitter
# Attempt 1: 2s + jitter
# Attempt 2: 4s + jitter
# Attempt 3: 8s + jitter (capped at max_delay)
```

### 4. Retry Decision Logic

```python
def is_retryable(error: Exception, config: RetryConfig) -> bool:
    # Direct exception types
    if isinstance(error, config.retryable_exceptions):
        return True
    
    # HTTP status codes
    if hasattr(error, 'status_code'):
        return error.status_code in config.retryable_status_codes
    
    # Keyword detection for string errors
    error_str = str(error).lower()
    retryable_keywords = ['timeout', 'connection', 'network', 'temporarily']
    return any(kw in error_str for kw in retryable_keywords)
```

### 5. Decorator API

Simple decorator for automatic retry:

```python
@retry_async(task_id="my_task", config=RetryConfig(max_retries=3))
async def unreliable_operation() -> Result:
    return await may_fail()
```

## Integration with Other Systems

### Circuit Breaker (ADR-005)

Retries + circuit breaker coordination:
- Circuit breaker handles provider-level failures
- Retry handles individual call failures
-两者协同: If circuit opens, retry stops quickly

### Budget Service (ADR-003)

LLM retries integrated with budget:
```python
# Before each retry attempt
if not budget_service.check(estimated_cost, operation):
    budget_service.rollback(reservation_id)
    raise BudgetExhaustedError()

# After success
budget_service.commit(reservation_id, actual_cost)
```

### Batch Pipeline (ADR-008)

Pipeline uses generic retry for each task:
```python
result = await self._run_task_with_retry(task, index)
```

## Consequences

### Positive

- **Resilience**: Transient failures automatically recovered
- **Cost control**: Budget integration prevents runaway costs
- **Observability**: Progress events for UI/debugging
- **Flexibility**: Configurable per-operation

### Negative

- **Complexity**: Multiple retry systems to understand
- **Latency**: Backoff adds time to failures
- **Debugging**: Harder to reproduce issues

#### ⚠️ Similar to Circuit Breaker: Implementation Duplication

**There are TWO retry implementations with DIFFERENT purposes:**

| Aspect | Generic Retry | LLM Retry |
|--------|--------------|-----------|
| File | `src/scout/retry.py` | `src/scout/llm/retry.py` |
| Budget | Optional | Required |
| Use case | Batch tasks | LLM calls |

**Assessment:** Unlike circuit breaker, this duplication is more justified - generic retry is for general tasks while LLM retry has budget integration. However, could be unified via shared base class.

**Recommendation:** Document the two use cases clearly, consider if they can share a base.

### Mitigations

- Clear logging at each retry stage
- Circuit breaker prevents excessive retries
- Jitter prevents synchronized retries
- Budget caps total retries

## Implementation Notes

### Code References

- Generic retry: `src/scout/retry.py`
- LLM retry: `src/scout/llm/retry.py`
- Tests: `tests/scout/test_retry.py`, `tests/scout/llm/test_retry.py`

### Usage Examples

```python
# Generic async retry
result = await with_retry_async(
    func,
    task_id="fetch_data",
    config=RetryConfig(max_retries=3, base_delay_ms=1000)
)

# LLM retry with budget
result = await call_with_retries(
    anthropic.call,
    context=LLMCallContext(
        budget_service=budget,
        reservation_id=reservation_id,
        model="claude-3"
    ),
    estimated_cost=0.01,
    max_retries=3
)

# Decorator
@retry_async(config=RetryConfig(max_retries=5))
async def fetch_url(url: str) -> Response:
    return await http.get(url)
```

### Configuration Tuning

| Scenario | max_retries | base_delay | max_delay |
|----------|-------------|-------------|-----------|
| LLM calls | 3 | 1s | 30s |
| Git operations | 5 | 0.5s | 10s |
| File operations | 3 | 0.1s | 1s |

## Related ADRs

- [ADR-003](./ADR-003-budget-service-reservation-semantics.md) - Budget integration
- [ADR-005](./ADR-005-circuit-breaker-pattern.md) - Circuit breaker coordination
- [ADR-008](./ADR-008-batch-pipeline.md) - Batch pipeline retry

## Notes

### Magic Number Audit

| File | Line | Value | Recommendation |
|------|------|-------|----------------|
| `retry.py` | 25 | `max_retries: int = 3` | Good default |
| `retry.py` | 26 | `base_delay_ms: float = 1000.0` | Good default |
| `retry.py` | 27 | `max_delay_ms: float = 30000.0` | 30 seconds cap |
| `retry.py` | 28 | `backoff_multiplier: float = 2.0` | Standard exponential |
| `retry.py` | 29 | `jitter: float = 0.1` | 10% - good |
| `llm/retry.py` | 76 | `max_retries: int = 3` | Matches generic |
| `llm/retry.py` | 77 | `base_delay: float = 1.0` | Matches generic (in seconds) |
| `llm/retry.py` | 78 | `max_delay: float = 30.0` | Matches generic |

**Positive:** LLM retry uses same values as generic (just in different units).

### Stub Implementations

None identified - retry mechanisms appear fully implemented.

### Future Considerations

- Adaptive retry (increase delays based on error patterns)
- Retry budgets (total retries per hour)
- Circuit breaker integration with LLM retry
- Retry across different providers (fallback)
