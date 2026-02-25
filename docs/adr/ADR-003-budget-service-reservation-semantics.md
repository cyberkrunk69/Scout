# ADR-003: Budget Service with Reservation Semantics

**Date:** 2025-02-24
**Status:** Accepted
**Deciders:** Scout Architecture Team
**Related:** ADR-001, ADR-002, ADR-009, ADR-010

## Context

Scout makes multiple LLM calls during a single task execution. Without budget controls, a runaway task could:
- Exhaust hourly quota in minutes
- Cause significant unexpected costs
- Leave no budget for subsequent tasks

Early implementations checked budget before each call but:
- Did not account for in-flight calls
- No way to "reserve" budget for multi-call operations
- Race conditions between concurrent calls
- No audit trail of reservations vs actuals

## Decision

Implemented **BudgetService** with reservation semantics inspired by database transaction patterns:

### Core Concepts

1. **Reservation**: Pre-allocate budget before operation
2. **Commit**: Record actual cost after completion
3. **Rollback**: Release reserved budget on failure

### API Design

```python
# Check if operation is possible (no reservation)
service.check(estimated_cost: float, operation: str) -> bool

# Reserve budget before operation
reservation = service.reserve(estimated_cost: float, operation: str)

# Use context manager for automatic commit/rollback
with service.reserve(0.01, "analyze_file") as reservation:
    result = call_llm(...)
    service.commit(reservation, actual_cost)
# OR on exception: automatic rollback
```

### Reservation Lifecycle

```
1. RESERVE          → Check limits, create Reservation object
2. (Operation runs) → LLM call in progress
3. COMMIT           → Deduct actual_cost from hourly budget
   OR
   ROLLBACK         → Release reserved amount (on exception)
```

### Budget Limits

- **Hourly budget**: Configurable limit (default: $1.00/hour)
- **Hard safety cap**: Absolute maximum (default: $10.00/hour)
- **Per-event limit**: Maximum cost per single operation
- **Allow overage %**: Permit slight over-reservation (default: 10%)

### Integration with Router (ADR-002)

The router checks budget before making LLM calls:
1. Router calls `budget.reserve(estimated, "llm_call")`
2. If reservation succeeds, proceed with LLM call
3. On success: `budget.commit(reservation, actual_cost)`
4. On exception: automatic rollback

## Consequences

### Positive

- **No race conditions**: Reservations tracked in memory, serialized per-instance
- **Predictable limits**: Hard cap prevents runaway costs
- **Audit trail**: Every reservation/commit/rollback logged
- **Overage handling**: Configurable tolerance prevents spurious failures
- **Integration**: Works with tier escalation (ADR-002) - each escalation is a new reservation

### Negative

- **In-memory only**: No persistence across restarts (reservations lost)
- **Per-instance**: Multiple Scout instances don't share budget state
- **Estimation error**: If estimate is wrong, budget may be under/over-utilized

### Mitigations

- Hourly budget resets on hour boundary (sliding window)
- Overage logging alerts operators to estimation issues
- Audit log provides post-hoc reconciliation capability
- Per-instance limits prevent single instance from consuming all budget

### Future Considerations

- Redis-backed budget service for multi-instance deployments
- Reservation timeout to auto-release stuck reservations
- Predictive budget allocation based on task complexity

## Related ADRs

- [ADR-001](./ADR-001-provider-registry.md) - Provider registry
- [ADR-002](./ADR-002-router-dispatch-select-separation.md) - Router and dispatch
- [ADR-009](./ADR-009-caching-strategy.md) - Caching (reduces budget usage)
- [ADR-010](./ADR-010-retry-mechanisms.md) - Retry with budget integration
