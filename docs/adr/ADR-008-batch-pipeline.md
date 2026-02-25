# ADR-008: Batch Pipeline

**Date:** 2025-02-25
**Status:** Accepted
**Deciders:** Scout Architecture Team
**Related:** ADR-005, ADR-006, ADR-010

## Test Status

| Test Suite | Tests | Status |
|------------|-------|--------|
| `tests/scout/batch/test_batch_pipeline.py` | 19 | ✅ PASS |

**Total:** 19 tests, all passing.

**Note:** Batch pipeline tests are primarily import/existence tests. Integration tests for actual pipeline execution would strengthen confidence.

---

## Context

Scout needs to execute multiple tasks in sequence or parallel:
- Running multiple CLI commands (lint, test, build)
- Processing multiple files
- Executing workflows across repositories

Early approaches:
- Simple loop with no error handling
- No way to skip tasks conditionally
- No variable interpolation between tasks
- No progress reporting

This led to:
- Brittle batch executions
- Difficult to debug failures
- No way to handle early exit conditions
- Poor user experience

## Decision

Implemented a **self-healing batch pipeline** with comprehensive features:

### Architecture

```
┌─────────────────┐
│ PipelineExecutor│
│  (Orchestrator) │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │         ┌──────────────┐
    ▼         ▼         │CircuitBreaker│
┌───────┐ ┌───────┐    │(ADR-005)     │
│Batch  │ │Retry  │────▶│              │
│Context│ │Config │    └──────────────┘
└───────┘ └───────┘
         │              ┌──────────────┐
         │              │ Progress    │
         └─────────────▶│ Reporter    │
                        └──────────────┘
```

### Core Features

#### 1. Execution Modes

```python
async def run(self, tasks: list[dict], mode: str = "sequential") -> list[dict]:
    # "sequential" - one task at a time (default)
    # "parallel" - all tasks run concurrently
```

#### 2. Conditional Execution

```yaml
- command: run
  args: {cmd: "npm test"}
  if: ${tests_exist}  # Skip if condition false

- command: build
  skip_if: ${skip_build}  # Skip if condition true

- command: deploy
  stop_if: ${deploy_failed}  # Stop pipeline if true
```

#### 3. Variable Interpolation

```yaml
variables:
  version: "1.0.0"

- command: run
  args: {cmd: "npm publish ${version}"}
  store_as: publish_result

- command: run
  args: {cmd: "echo ${publish_result.version}"}
```

#### 4. Auto-JSON Flag Injection

Commands that need structured output automatically get `--json` flags:

```python
COMMANDS_NEED_JSON = {
    "git_status": "json",
    "git_branch": "json",
    "plan": "json_output",
    "lint": "json_output",
    # ... more commands
}
```

#### 5. Sub-batch Execution

Extract steps from plan output and execute as sub-batch:

```yaml
- command: plan
  args: {prompt: "Add tests for auth"}
  extract_steps: true
  subbatch_mode: sequential
```

### Error Handling

#### Circuit Breaker Integration

Pipeline stops if circuit opens (see ADR-005):

```python
if self.circuit_breaker and not self.circuit_breaker.can_execute():
    await self.reporter.emit(ProgressEvent(
        status=Status.CIRCUIT_OPEN,
        message="Circuit breaker open - stopping execution"
    ))
    break
```

#### Retry Logic

Each task can be retried with exponential backoff (see ADR-010):

```python
async def _run_task_with_retry(self, task: dict, index: int) -> dict:
    retry_ctx = RetryContext(task_id=f"task:{index}", config=self.retry_config)
    
    while retry_ctx.can_retry:
        result = await self.task_runner(task, ...)
        
        if result.get("status") == "success":
            return result
        
        if not is_retryable(error, self.retry_config):
            return result
        
        await retry_ctx.wait_before_retry()
```

### Progress Reporting

Emits structured progress events:

```python
await self.reporter.emit_async(ProgressEvent(
    task_id="batch:start",
    status=Status.RUNNING,
    message="Starting 5 tasks in sequential mode",
    metadata={"total_tasks": 5, "mode": "sequential"}
))
```

Statuses: `RUNNING`, `SUCCESS`, `FAILURE`, `SKIPPED`, `RETRY`, `CIRCUIT_OPEN`, `COMPLETE`

## Consequences

### Positive

- **Resilience**: Circuit breaker and retries prevent cascade failures
- **Flexibility**: Conditionals and variables enable complex workflows
- **Observability**: Rich progress events for UI integration
- **Automation**: Auto-JSON reduces user boilerplate
- **Composability**: Sub-batches enable hierarchical execution

### Negative

- **Complexity**: Many features to understand
- **Overhead**: Conditional evaluation adds latency
- **Debugging**: Variable interpolation can be hard to trace

### Mitigations

- Clear YAML schema documentation
- Progress events for debugging
- Dry-run mode for testing

## Implementation Notes

### Code References

- Main pipeline: `src/scout/batch_pipeline.py`
- Context: `src/scout/batch_context.py`
- Expression evaluator: `src/scout/batch_expression.py`
- Tests: `tests/scout/batch/test_batch_pipeline.py`

### Task Schema

```yaml
- command: string           # Required: command name
  args: object              # Optional: command arguments
  mode: sequential|parallel # Optional: execution mode
  variables: object         # Optional: set variables
  if: expression            # Optional: run if true
  skip_if: expression      # Optional: skip if true
  stop_if: expression      # Optional: stop if true
  store_as: string         # Optional: store result
  extract_steps: bool      # Optional: extract sub-batch
  subbatch_mode: string   # Optional: sub-batch mode
```

## Related ADRs

- [ADR-005](./ADR-005-circuit-breaker-pattern.md) - Circuit breaker integration
- [ADR-006](./ADR-006-execution-framework.md) - Execution framework
- [ADR-010](./ADR-010-retry-mechanisms.md) - Retry mechanisms

## Notes

### Magic Number Audit

| File | Line | Value | Recommendation |
|------|------|-------|----------------|
| `batch_pipeline.py` | 35-53 | Multiple commands in `COMMANDS_NEED_JSON` | Could be config file |

### Stub Implementations

None explicitly identified - batch pipeline appears functional.

### Future Considerations

- Add more expression operators (regex, contains, etc.)
- Implement task timeouts per-command
- Add support for distributed batch execution
- Add webhook notifications for completion
