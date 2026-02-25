# ADR-004: Trust System Design

**Date:** 2025-02-24
**Status:** Accepted
**Deciders:** Scout Architecture Team

## Context

Scout uses AI to generate code and documentation. Generated outputs must be trusted before execution. The challenge:

- **Stale embeddings**: Documentation embeddings may not reflect current source code
- **Gap detection**: Need to detect when generated content diverges from source
- **Confidence scoring**: Not all source-documentation pairs have equal reliability
- **Learning capability**: System should improve trust decisions over time

Early approaches:
- Simple "trust all" - too permissive, risky for production
- Manual verification - doesn't scale
- Hash comparison only - too coarse, can't detect semantic drift

## Decision

Implemented a **multi-component Trust System** with the following architecture:

### Components

```
┌─────────────────────────────────────────────────────────┐
│                   TrustOrchestrator                      │
│  (Coordinates verification → penalty → result)          │
└─────────────────────────────────────────────────────────┘
         ↓              ↓              ↓           ↓
┌──────────────┐ ┌──────────────┐ ┌─────────────┐ ┌────────────┐
│TrustVerifier │ │ TrustPenalizer│ │TrustLearner │ │TrustStore  │
│(Check source │ │ (Apply trust │ │(Bayesian    │ │(Persist    │
│ vs docs)     │ │  penalties)  │ │ learning)   │ │ records)   │
└──────────────┘ └──────────────┘ └─────────────┘ └────────────┘
```

### TrustVerifier

Verifies source against embedded documentation:

```python
def verify(source_path: Path, doc_path: Path) -> VerificationResult:
    # Extract symbols from source
    # Extract embedded checksum from doc
    # Compare current checksum vs embedded
    # Identify stale vs fresh symbols
    # Return verification result
```

- **Checksum comparison**: SHA-based for exact match detection
- **Symbol extraction**: Parse source for functions, classes, constants
- **Gap message**: Human-readable explanation of trust level

### TrustPenalizer

Applies penalties based on trust level:

- **Low trust**: High penalty (reduce confidence, require human review)
- **Medium trust**: Moderate penalty (warn but allow)
- **High trust**: No penalty (proceed with confidence)

### TrustLearner

Bayesian learning from outcomes:

```python
# Track success/failure per source-documentation pair
record.success_count += 1  # or failure_count
# Compute posterior probability of trust
confidence = P(trust | historical_outcomes)
```

### TrustStore

Persistent storage of trust records:

- Source path → Documentation path mapping
- Trust level, penalty, query counts
- Checksums for staleness detection
- Last validated timestamps

### Trust Levels

| Level | Description | Penalty |
|-------|-------------|--------|
| **high** | Fresh, matching checksums, proven track record | 0.0 |
| **medium** | Some staleness or new pair | 0.25 |
| **low** | Stale embeddings or no history | 0.75 |
| **none** | Never validated or failed | 1.0 |

### Decision Flow

```
1. Receive source + doc pair
2. Lookup in TrustStore
3. TrustVerifier.verify() → VerificationResult
4. TrustPenalizer.apply() → PenaltyResult
5. TrustLearner.update() → Update success/failure counts
6. Return TrustResult (level, penalty, confidence, gap_message)
```

## Consequences

### Positive

- **Scalable**: Automatic verification at scale
- **Auditable**: Every decision logged with gap explanation
- **Self-improving**: Bayesian learning from real outcomes
- **Granular**: Symbol-level (not just file-level) detection
- **Configurable**: Penalty thresholds adjustable per deployment

### Negative

- **Complexity**: Multiple components require understanding
- **Overhead**: Verification adds latency before code execution
- **Storage**: Trust records grow over time (pruning needed)
- **Cold start**: New source-doc pairs have no history

### Mitigations

- Caching: Trust results cached, only recompute on source change
- Incremental: Only verify changed symbols
- Defaults: New pairs default to "medium" trust (cautious)
- Pruning: TrustStore periodically removes stale records

## Related ADRs

- [ADR-009](./ADR-009-caching-strategy.md) - Trust caching uses caching strategy
- [ADR-006](./ADR-006-execution-framework.md) - Trust verification integrated with execution
