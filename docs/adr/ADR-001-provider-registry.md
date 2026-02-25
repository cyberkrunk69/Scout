# ADR-001: Use of Provider Registry with Multi-Key Support

**Date:** 2025-02-24
**Status:** Accepted
**Deciders:** Scout Architecture Team
**Related:** ADR-002, ADR-003, ADR-005

## Context

Scout relies on multiple LLM providers (Anthropic, Google Gemini, Groq, MiniMax, OpenRouter, DeepSeek) for executing AI-powered software engineering tasks. Each provider may have multiple API keys, either from the same account or different accounts. The system must handle:

- **Key rotation**: When one key fails, automatically try another
- **Health tracking**: Monitor individual key health (failures, cooldowns, permanent failures)
- **Rate limiting**: Respect per-key and per-provider rate limits
- **Permanent errors**: Detect auth failures, quota exceeded, geo-blocking that should not trigger key rotation
- **Circuit breaking**: Stop using providers that are experiencing systemic failures

Early implementations used simple per-provider API key environment variables (`ANTHROPIC_API_KEY`, `GROQ_API_KEY`, etc.) with no key rotation or health tracking.

## Decision

We implemented a **ProviderRegistry** system with the following components:

1. **ProviderClient**: Holds a provider's callable and manages a pool of `KeyState` objects
2. **KeyState**: Tracks individual API key health:
   - Failure count and last failure timestamp
   - Cooldown timer with exponential backoff (max 1 hour)
   - Permanent failure flag for unrecoverable errors
3. **ProviderRegistry**: Central registry mapping provider names to ProviderClient instances
4. **Permanent error detection**: Pattern matching + exception type checking to identify errors that should not retry

### Key Rotation Algorithm

```python
def get_working_key():
    healthy = [k for k in keys if k.is_healthy()]  # Not in cooldown, not permanently failed
    return min(healthy, key=lambda k: k.failures).key  # Prefer lowest failures
```

### Environment Variable Precedence

- `*_API_KEYS` (comma-separated) takes priority over `*_API_KEY` (single key)
- Example: `GROQ_API_KEYS=key1,key2,key3` or `ANTHROPIC_API_KEY=sk-...`

### Circuit Breaker Integration

Each provider has a circuit breaker that tracks failures at the provider level. When the circuit opens:
1. Fallback providers are consulted (e.g., Groq → MiniMax → OpenRouter)
2. If no fallbacks available, error propagates

## Related ADRs

- [ADR-002](./ADR-002-router-dispatch-select-separation.md) - Router/dispatch separation
- [ADR-003](./ADR-003-budget-service-reservation-semantics.md) - Budget service
- [ADR-005](./ADR-005-circuit-breaker-pattern.md) - Circuit breaker (provider-level protection)
- **Cost optimization**: Can use free-tier keys alongside paid keys
- **Observability**: Key health is visible via `get_router_status()`
- **Graceful degradation**: Circuit breakers prevent hammering failing providers
- **Developer experience**: Simple env var configuration (`*_API_KEYS`)

### Negative

- **Complexity**: Additional abstraction layer requires understanding
- **Memory**: Each key state object uses minimal but non-zero memory
- **Latency**: Key rotation adds a few milliseconds per LLM call

### Mitigations

- Circuit breaker state is persisted to avoid "thundering herd" on restart
- Maximum key attempts capped (default: 3) to prevent infinite loops
- Comprehensive logging at INFO level for routing decisions

## Related ADRs

- [ADR-002](./ADR-002-router-dispatch-select-separation.md) - Router/dispatch separation
- [ADR-003](./ADR-003-budget-service-reservation-semantics.md) - Budget service
- [ADR-005](./ADR-005-circuit-breaker-pattern.md) - Circuit breaker (provider-level protection)
- [ADR-010](./ADR-010-retry-mechanisms.md) - Retry mechanisms integration
