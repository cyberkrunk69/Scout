# ADR-007: Plan Executor and State Machine

**Date:** 2025-02-25
**Status:** Accepted
**Deciders:** Scout Architecture Team
**Related:** ADR-006, ADR-008

## Test Status

| Test Suite | Tests | Status |
|------------|-------|--------|
| `tests/scout/plan_state/test_plan_state.py` | 8 | ✅ PASS |

**Total:** 8 tests, all passing.

---

## Context

Scout performs dynamic re-synthesis planning where:
- Plans can span multiple sessions
- Users may pause and resume work
- Multiple plans can be active simultaneously
- Plans need to survive process restarts

Without persistent state:
- Progress lost on restart
- No way to resume interrupted plans
- Difficult to debug failed executions
- No audit trail of plan history

## Decision

Implemented a **Plan State Machine** with disk-based persistence:

### State Transitions

```
┌─────────┐    execute     ┌─────────┐   complete    ┌───────────┐
│ ACTIVE  │ ─────────────→ │ RUNNING │ ─────────────→ │ COMPLETED │
└─────────┘                └─────────┘                └───────────┘
       ↑                         │
       │    interrupt/fail       │
       └─────────────────────────┘
                ↓
          ┌───────────┐
          │ ARCHIVED  │
          └───────────┘
```

### Directory Structure

```
.scout/
├── locks/
│   └── {plan_id}.lock/     # Atomic lock files
└── plans/
    ├── active/              # Plans in progress
    │   └── {plan_id}.json
    ├── completed/           # Successfully finished
    │   └── {plan_id}.json
    └── archived/            # Failed or cancelled
        └── {plan_id}.json
```

### Lock Mechanism

Uses atomic `mkdir` for POSIX-compatible locking:

```python
def acquire_lock(plan_id: str, timeout: int = 30) -> bool:
    """Acquire lock using atomic mkdir."""
    lock_path = LOCKS_DIR / f"{plan_id}.lock"
    lock_path.mkdir(parents=True, exist_ok=False)  # Atomic!
```

**Key Features:**
- **Atomic**: Uses mkdir (atomic on POSIX)
- **Stale detection**: Configurable timeout (default: 1 hour)
- **Recovery**: Automatic stale lock cleanup
- **PID tracking**: Records process ID for debugging

### PlanStateManager

Manages plan lifecycle:

```python
class PlanStateManager:
    async def save_plan_state(self, context) -> str
    async def load_plan_state(self, plan_id: str)
    async def transition_plan(self, plan_id: str, to_state: str) -> bool
    async def recover_stale_locks(self) -> int
```

### Persistence Format

```json
{
  "plan_id": "fix_bug_abc123",
  "request": "Fix the authentication issue",
  "depth": 2,
  "max_depth": 3,
  "summary": "Implemented token refresh",
  "discoveries": [...],
  "is_pivoting": false,
  "created_at": "2025-02-25T10:00:00Z",
  "updated_at": "2025-02-25T10:30:00Z"
}
```

### Configuration

```bash
SCOUT_PLAN_LOCK_TIMEOUT=30        # Lock acquisition timeout (seconds)
SCOUT_PLAN_STALE_LOCK_HOURS=1    # Stale lock threshold (hours)
SCOUT_PLAN_ARCHIVE_DAYS=7        # Archive completed after N days
SCOUT_PLAN_DELETE_DAYS=30        # Delete archived after N days
```

## Consequences

### Positive

- **Persistence**: Plans survive restarts
- **Recovery**: Stale locks automatically cleaned
- **Audit**: Full history of plan executions
- **Concurrency**: Multiple plans can run safely
- **Debuggability**: Can inspect any plan state

### Negative

- **Disk I/O**: State writes add latency
- **Cleanup needed**: Old plans must be pruned
- **Complexity**: Lock mechanism adds code

### Mitigations

- Atomic writes (temp file + rename) prevent corruption
- Async operations don't block execution
- Automatic cleanup via `cleanup_old_plans()`

## Implementation Notes

### Code References

- State management: `src/scout/plan_state.py`
- Tests: `tests/scout/plan_state/test_plan_state.py`

### Atomic Write Pattern

```python
# Safe atomic write
temp_path = active_dir / f".{plan_id}.tmp"
temp_path.write_text(json.dumps(data))
temp_path.rename(final_path)  # Atomic on same filesystem
```

### Lock Acquisition Flow

```
1. Try mkdir (atomic)
2. If fails: check if stale (mtime > threshold)
3. If stale: remove and retry
4. If timeout: return False
5. On success: write pid/timestamp files
```

## Related ADRs

- [ADR-006](./ADR-006-execution-framework.md) - Execution framework integration
- [ADR-008](./ADR-008-batch-pipeline.md) - Batch pipeline uses plan state

## Notes

### Magic Number Audit

| File | Line | Value | Recommendation |
|------|------|-------|----------------|
| `plan_state.py` | 40 | `timeout: int = 30` | Already env configurable ✅ |
| `plan_state.py` | 41 | `stale_hours: int = 1` | Already env configurable ✅ |
| `plan_state.py` | 331 | `archive_days = 7` | Already env configurable ✅ |
| `plan_state.py` | 332 | `delete_days = 30` | Already env configurable ✅ |

**Positive:** Plan state management has GOOD configuration coverage via environment variables.

### Stub Implementations

None identified - the plan state system appears fully implemented.

### Future Considerations

- Add lock heartbeat to detect hung processes
- Implement plan merging for collaborative editing
- Add state compression for large plans
- Consider database backend for better querying
