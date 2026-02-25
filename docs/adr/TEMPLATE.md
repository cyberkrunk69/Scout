# ADR-XXX: Title of the Decision

**Date:** YYYY-MM-DD
**Status:** Draft | Accepted | Deprecated | Superseded
**Deciders:** Scout Architecture Team
**Related:** ADR-XXX, ADR-YYY

## Context

Describe the issue motivating this decision. What problem are we solving? Why is this decision needed now?

Include:
- Background on the system state before this decision
- Forces or constraints that influenced the decision
- Alternatives that were considered

## Decision

Describe the change we're proposing and/or implementing. Use present tense.

Include:
- What was decided
- How it will work
- Key components or modules affected
- Configuration options if applicable

## Consequences

Describe the resulting context after the decision is implemented.

### Positive

- List the benefits of this decision
- What problems does it solve?

### Negative

- List the drawbacks, trade-offs, or costs
- What new problems might arise?

### Mitigations

- How are negatives being addressed?
- What workarounds exist?

## Implementation Notes

Optional section for technical details helpful during implementation.

### Code References

- File: `src/scout/module/file.py`
- Function: `function_name()`

### Configuration

```python
# Configuration options
CONFIG_OPTION = "default_value"  # Description
```

### Testing

- Unit tests: `tests/scout/test_module.py`
- Integration tests: `tests/scout/integration/`

## Related ADRs

- [ADR-001](./ADR-001-provider-registry.md) - Related context
- [ADR-002](./ADR-002-router-dispatch-select-separation.md) - Related context

## Notes

Optional section for additional notes, questions, or future considerations.

### Open Questions

- Any unresolved questions
- Items for future discussion

### Future Considerations

- Potential future changes
- Scalability concerns
- Alternative approaches to revisit
