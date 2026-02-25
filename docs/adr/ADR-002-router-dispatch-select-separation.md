# ADR-002: Separation of Router, Dispatch, and Select

**Date:** 2025-02-24
**Status:** Accepted
**Deciders:** Scout Architecture Team
**Related:** ADR-001, ADR-003, ADR-005, ADR-006

## Context

Scout makes LLM calls for various tasks (simple queries, complex reasoning, code generation). Different tasks require different models:
- **Fast/cheap models**: For simple tasks, first attempts
- **Large/expensive models**: For complex reasoning, fallback from failures

Early implementations mixed concerns:
- `router.py` selected models, made calls, handled retries, estimated costs
- No clear separation between "which model to use" vs "actually make the call"

This led to:
- Difficult to test model selection logic in isolation
- Hard to add new providers without modifying multiple files
- Cost estimation buried in routing logic

## Decision

We separated into three distinct responsibilities:

### 1. Select (`scout/llm/select.py`)

**Responsibility**: Choose the best model given task type, iteration, tier, and mode.

```python
def select_model(task_type: str, iteration: int = 0, current_tier: Optional[str] = None, mode: Optional[str] = None) -> str:
    # Filter by tier (fast/medium/large)
    # Filter by mode (free/paid/auto)
    # Filter by provider availability
    # Return candidate_models[iteration % len(candidate_models)]
```

- **Inputs**: Task type, iteration count, current tier, LLM mode
- **Outputs**: Model name (e.g., "llama-3.1-8b-instant", "MiniMax-M2.5")
- **Pure function**: No side effects, easy to test

### 2. Router (`scout/llm/router.py`)

**Responsibility**: Orchestrate the full LLM call lifecycle.

- Select model via `select_model()`
- Get provider via `get_provider_for_model()`
- Check circuit breaker (provider health)
- Iterate through healthy keys, handle failures
- Implement fallback chains (provider → fallback provider → tier escalation)
- Return `LLMResult` with full metadata

### 3. Dispatch (`scout/llm/dispatch.py`)

**Responsibility**: Route to specific provider implementation.

- Maps model names to provider implementations (pattern-based)
- Example: "llama*" → Groq, "gemini*" → Google, "claude*" → Anthropic
- Provides unified interface (`call_llm_async`)
- Legacy compatibility layer (being phased out in favor of router)

## Consequences

### Positive

- **Testability**: Each component testable in isolation
  - `select.py`: Unit test model selection with mock cost tables
  - `router.py`: Integration test with mock providers
  - `dispatch.py`: Can test provider implementations independently
- **Extensibility**: Add new providers by updating dispatch map
- **Single Responsibility**: Each module has one reason to change
- **Debugging**: Clear which component failed (select vs route vs dispatch)

### Negative

- **Indirection**: Additional function call overhead
- **Initial complexity**: More files to understand
- **Coordination**: Ensuring router and dispatch stay synchronized

### Interactions

```
User Request
    ↓
router.call_llm()
    ↓
select.select_model()  ← Determine model
    ↓
dispatch.get_provider_for_model()  ← Map to provider
    ↓
registry.get(key)  ← Get healthy key (ADR-001)
    ↓
provider.call()  ← Actual LLM call
```

### Migration Path

The system is transitioning:
- New code: Use `router.call_llm()` directly
- Legacy: `dispatch.call_llm_async()` still works for backward compatibility
- Eventually: Dispatch becomes pure provider implementations, router handles orchestration

## Related ADRs

- [ADR-001](./ADR-001-provider-registry.md) - Provider registry and key health
- [ADR-003](./ADR-003-budget-service-reservation-semantics.md) - Budget service
- [ADR-005](./ADR-005-circuit-breaker-pattern.md) - Circuit breaker
- [ADR-006](./ADR-006-execution-framework.md) - Execution framework
