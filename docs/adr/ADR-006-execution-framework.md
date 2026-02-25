# ADR-006: Execution Framework

**Date:** 2025-02-25
**Status:** Accepted
**Deciders:** Scout Architecture Team
**Related:** ADR-007, ADR-010

## Test Status

| Test Suite | Tests | Status |
|------------|-------|--------|
| `tests/scout/execution/test_executor.py` | 18 | ✅ PASS |

**Total:** 18 tests, all passing.

---

## Context

Scout executes complex multi-step tasks that involve:
- File operations (create, modify, delete)
- Command execution (run tests, lint, build)
- User interaction (get input, confirmations)
- Web automation (browser actions)

Early implementations:
- Mixed execution logic with planning/parsing
- No clear dependency resolution between steps
- Limited support for parallel execution
- No built-in rollback capability

This led to:
- Difficult to test individual operations
- No way to recover from failures gracefully
- Hard to optimize execution order
- Complex error handling scattered throughout

## Decision

Implemented a structured **Execution Framework** with clear separation of concerns:

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      PlanExecutor                                 │
│  (Orchestrates execution, manages dependencies, budgets)        │
└─────────────────────────────────────────────────────────────────┘
         ↓                    ↓                      ↓
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
│ExecutionRegistry│  │  RollbackManager │  │      BudgetGuard        │
│ (Tool mapping)   │  │ (Change tracking)│  │ (Cost enforcement)      │
└─────────────────┘  └─────────────────┘  └─────────────────────────┘
```

### 1. Action System (`src/scout/execution/actions.py`)

Defines the structured representation of executable steps:

```python
class ActionType(Enum):
    CREATE_FILE = "create_file"
    MODIFY_FILE = "modify_file"
    DELETE_FILE = "delete_file"
    RUN_COMMAND = "run_command"
    READ_FILE = "read_file"
    GET_USER_INPUT = "get_user_input"
    BROWSER_ACT = "browser_act"

@dataclass
class StructuredStep:
    action_type: ActionType
    description: str
    step_id: int
    depends_on: List[int]          # Dependency graph
    rollback_on_fail: bool         # Auto-rollback flag
    max_retries: int = 2
    timeout_seconds: int = 300
```

### 2. PlanExecutor (`src/scout/execution/executor.py`)

The main orchestrator that:

- **Builds execution batches**: Topological sort (Kahn's algorithm) for dependency resolution
- **Manages parallel execution**: Steps without dependencies run concurrently
- **Enforces budget**: BudgetGuard tracks and limits costs
- **Tracks changes**: RollbackManager records operations for potential rollback

```python
class PlanExecutor:
    async def execute(self, plan: StructuredPlan) -> ExecutionResult:
        # 1. Build DAG and get execution order
        batches = self._build_execution_batches(plan.steps)
        
        # 2. Execute each batch (parallel within batch)
        for batch in batches:
            tasks = [self._execute_step(step, results) for step in batch]
            await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. Execution Registry

Maps action types to actual tool implementations:

```python
class ExecutionToolRegistry:
    def get_tool_name(self, action_type: ActionType) -> str
    def get_adapter(self, action_type: ActionType) -> Optional[Callable]
```

### 4. RollbackManager

Tracks changes for self-healing:

```python
class RollbackManager:
    def record(self, step_id, change_type, undo_command, metadata)
    async def rollback_to(self, step_id) -> List[Dict]
```

### 5. BudgetGuard

Prevents runaway costs during execution:

```python
class BudgetGuard:
    async def check(self, step, estimated_cost) -> bool
    async def update(self, actual_cost) -> None
```

## Consequences

### Positive

- **Testability**: Each component testable in isolation
- **Dependency resolution**: Automatically identifies parallelization opportunities
- **Budget enforcement**: Prevents runaway costs during execution
- **Rollback capability**: Can undo changes on failure
- **Extensibility**: Easy to add new action types

### Negative

- **Complexity**: Multiple components to understand
- **Overhead**: Dependency analysis adds latency
- **State management**: Must track execution state carefully

### Mitigations

- Clear module boundaries and documentation
- Execution batching optimizes for common cases
- State persisted for debugging (see ADR-007)

## Implementation Notes

### Code References

- Actions: `src/scout/execution/actions.py`
- Executor: `src/scout/execution/executor.py`
- Registry: `src/scout/execution/registry.py`
- Tests: `tests/scout/execution/test_executor.py`

### Execution Flow

```
1. Parse LLM response → StructuredPlan (steps + dependencies)
2. PlanExecutor.execute() called
3. _build_execution_batches() → topological sort
4. For each batch:
   a. Filter steps where dependencies satisfied
   b. Execute in parallel via asyncio.gather()
   c. Update results for dependent steps
5. Return ExecutionResult with stats
```

### Configuration

```python
# Default configuration
PlanExecutor(
    registry=ExecutionToolRegistry(),
    max_budget=0.10,  # $0.10 default
)
```

## Related ADRs

- [ADR-007](./ADR-007-plan-executor-state-machine.md) - Plan state persistence
- [ADR-010](./ADR-010-retry-mechanisms.md) - Step-level retry integration

## Notes

### Stub Implementations

**⚠️ The following methods are STUBS - they contain `pass` and don't actually implement functionality:**

1. **`RollbackManager.generate_undo_plan()`** in `src/scout/execution/executor.py:63-66`:
   ```python
   def generate_undo_plan(self, from_step: int) -> StructuredPlan:
       """Generate a plan to undo changes."""
       # Future enhancement: generate structured undo plan
       pass
   ```
   
   **Impact:** Users cannot currently undo failed plan executions.
   
   **Recommendation:** Either complete implementation or remove the stub.
   
   **Ticket created:** `#TECH-DEBT-003` - Implement generate_undo_plan or remove stub

2. **Non-browser steps skipped** in `src/scout/plan_executor.py:138-140`:
   ```python
   else:
       # Non-browser step - skip for now or log warning
       logger.warning(f"Skipping non-browser step: {step_dict.get('command')}")
       continue
   ```
   
   **Impact:** Only web browser actions are executed; file operations, git, etc. are silently skipped.
   
   **Recommendation:** Implement full action type support.

### Magic Number Audit

| File | Line | Value | Recommendation |
|------|------|-------|----------------|
| `executor.py` | 98 | `max_budget=0.10` | Already configurable via constructor |
| `executor.py` | 235 | `estimated_cost = 0.001` | Should use actual estimation |
| `actions.py` | 37 | `max_retries: int = 2` | Should use RetryConfig |
| `actions.py` | 38 | `timeout_seconds: int = 300` | Should be configurable |

### Future Considerations

- Add more sophisticated rollback strategies
- Implement checkpointing for long-running executions
- Add execution visualization/debugging tools
- Support for conditional branching in execution
