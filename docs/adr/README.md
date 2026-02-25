# Architecture Decision Records (ADR) Index

This directory contains Architecture Decision Records for the Scout system. ADRs document significant architectural decisions, their context, and consequences.

## What is an ADR?

An Architecture Decision Record is a document that captures an important architectural decision along with its context and consequences. See [MADR](https://adr.github.io/madr/) for the template format used here.

## ADR List

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](./ADR-001-provider-registry.md) | Use of Provider Registry with Multi-Key Support | Accepted | 2025-02-24 |
| [ADR-002](./ADR-002-router-dispatch-select-separation.md) | Separation of Router, Dispatch, and Select | Accepted | 2025-02-24 |
| [ADR-003](./ADR-003-budget-service-reservation-semantics.md) | Budget Service with Reservation Semantics | Accepted | 2025-02-24 |
| [ADR-004](./ADR-004-trust-system-design.md) | Trust System Design | Accepted | 2025-02-24 |
| [ADR-005](./ADR-005-circuit-breaker-pattern.md) | Circuit Breaker Pattern | Accepted | 2025-02-25 |
| [ADR-006](./ADR-006-execution-framework.md) | Execution Framework | Accepted | 2025-02-25 |
| [ADR-007](./ADR-007-plan-executor-state-machine.md) | Plan Executor and State Machine | Accepted | 2025-02-25 |
| [ADR-008](./ADR-008-batch-pipeline.md) | Batch Pipeline Design | Accepted | 2025-02-25 |
| [ADR-009](./ADR-009-caching-strategy.md) | Caching Strategy | Accepted | 2025-02-25 |
| [ADR-010](./ADR-010-retry-mechanisms.md) | Retry Mechanisms | Accepted | 2025-02-25 |

## ADR Statuses

- **Draft**: Under review, not yet accepted
- **Accepted**: Approved and implemented
- **Deprecated**: No longer recommended
- **Superseded**: Replaced by another ADR

## Cross-Reference Map

```
ADR-001 (Provider Registry)
  ├─ Related: ADR-002, ADR-003, ADR-005, ADR-010
  └─ References: circuit_breaker.py, router.py

ADR-002 (Router/Dispatch/Select)
  ├─ Related: ADR-001, ADR-003, ADR-005, ADR-006
  └─ References: select.py, router.py, dispatch.py

ADR-003 (Budget Service)
  ├─ Related: ADR-001, ADR-002, ADR-009, ADR-010
  └─ References: budget.py

ADR-004 (Trust System)
  ├─ Related: ADR-006, ADR-009
  └─ References: trust/verifier.py, trust/store.py

ADR-005 (Circuit Breaker)
  ├─ Related: ADR-001, ADR-008, ADR-010
  └─ References: circuit_breaker.py, llm/circuit_breaker.py

ADR-006 (Execution Framework)
  ├─ Related: ADR-007, ADR-010
  └─ References: execution/executor.py, execution/actions.py

ADR-007 (Plan Executor)
  ├─ Related: ADR-006, ADR-008
  └─ References: execution/executor.py, plan_state.py

ADR-008 (Batch Pipeline)
  ├─ Related: ADR-005, ADR-007, ADR-010
  └─ References: batch_pipeline.py

ADR-009 (Caching Strategy)
  ├─ Related: ADR-003
  └─ References: cache.py, cache_deps.py

ADR-010 (Retry Mechanisms)
  ├─ Related: ADR-003, ADR-005, ADR-008
  └─ References: retry.py, llm/retry.py
```

## Creating a New ADR

1. Copy `TEMPLATE.md` to a new file (e.g., `ADR-011-new-decision.md`)
2. Fill in all sections following the MADR format
3. Update this index with the new ADR
4. Add cross-references to related ADRs
5. Submit for review

## Maintenance

- Review ADRs annually for relevance
- Mark ADRs as deprecated/superseded when appropriate
- Keep cross-references up to date
