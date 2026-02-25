# Batch Pipeline Guide

The batch pipeline enables parallel and sequential execution of multiple tasks with built-in support for conditionals, variables, retry logic, and circuit breakers.

## What is the Batch Pipeline For?

The batch pipeline is designed for:

- **Parallel processing of tasks**: Run multiple commands concurrently
- **Sequential workflows**: Execute dependent tasks in order
- **Conditional execution**: Skip or stop based on conditions
- **Variable interpolation**: Pass data between tasks
- **Self-healing**: Automatic retries and circuit breaker protection

## Core Concepts

### PipelineExecutor

The `PipelineExecutor` is the main orchestrator for batch operations:

```python
from scout.batch_pipeline import PipelineExecutor
from scout.batch_context import BatchContext

# Create context with initial variables
context = BatchContext(initial_vars={"env": "prod"})

# Create executor
executor = PipelineExecutor(
    context=context,
    task_runner=my_task_runner,
)

# Run tasks
results = await executor.run(tasks, mode="sequential")
```

### BatchContext

The `BatchContext` provides thread-safe state management across task execution:

```python
from scout.batch_context import BatchContext

context = BatchContext()

# Set variables
context.set_var("version", "1.0.0")
context.set_var("build_id", 123)

# Access variables with dot notation
version = context.get_var("version")
build_id = context.get_var("build_id")
```

### Execution Modes

Tasks can run in two modes:

- **`sequential`** (default): Tasks run one at a time
- **`parallel`**: All tasks run concurrently

```python
# Sequential execution
await executor.run(tasks, mode="sequential")

# Parallel execution
await executor.run(tasks, mode="parallel")
```

## Defining Batch Tasks

Each task is a dictionary with the following structure:

```python
task = {
    "command": "lint",           # Required: command name
    "args": {"paths": ["."]},    # Optional: command arguments
    "mode": "sequential",        # Optional: execution mode
    "variables": {},             # Optional: set variables for this task
    "if": "${var_exists}",        # Optional: run if condition is true
    "skip_if": "${skip_flag}",   # Optional: skip if condition is true
    "stop_if": "${failed}",      # Optional: stop pipeline if true
    "store_as": "result_var",    # Optional: store result in variable
    "extract_steps": False,      # Optional: extract sub-batch from plan
    "subbatch_mode": "sequential" # Optional: sub-batch mode
}
```

### Task Fields

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | The command to execute (required) |
| `args` | object | Arguments passed to the command |
| `mode` | string | "sequential" or "parallel" (overrides global mode) |
| `variables` | object | Variables to set before execution |
| `if` | expression | Run task only if condition is true |
| `skip_if` | expression | Skip task if condition is true |
| `stop_if` | expression | Stop pipeline if condition is true |
| `store_as` | string | Store result in context variable |
| `extract_steps` | bool | Extract plan steps and execute as sub-batch |
| `subbatch_mode` | string | Mode for sub-batch execution |

## Conditional Execution

### Using `if`

Run a task only when a condition is true:

```python
tasks = [
    {
        "command": "git_status",
        "args": {},
        "store_as": "status"
    },
    {
        "command": "run",
        "args": {"cmd": "npm test"},
        "if": "${status.clean}"  # Only run if repo is clean
    }
]
```

### Using `skip_if`

Skip a task when a condition is true:

```python
tasks = [
    {
        "command": "build",
        "args": {},
        "skip_if": "${skip_build}"  # Skip if skip_build is true
    }
]
```

### Using `stop_if`

Stop the entire pipeline when a condition is true:

```python
tasks = [
    {
        "command": "deploy",
        "args": {},
        "store_as": "deploy_result"
    },
    {
        "command": "notify",
        "args": {"message": "Deployment complete"},
        "stop_if": "${deploy_result.failed}"  # Stop if deploy failed
    }
]
```

## Variable Interpolation

### Setting Variables

Variables can be set in the task definition:

```python
tasks = [
    {
        "command": "run",
        "args": {"cmd": "echo Hello ${name}"},
        "variables": {"name": "World"}
    }
]
```

Or stored from task results:

```python
tasks = [
    {
        "command": "git_status",
        "args": {},
        "store_as": "git_status"
    },
    {
        "command": "run",
        "args": {"cmd": "echo ${git_status.branch}"}
    }
]
```

### Accessing Variables

Use `${variable_name}` syntax in:

- Task arguments
- Conditional expressions
- Command strings

Dot notation accesses nested properties:

```python
context.set_var("result", {"status": "success", "output": "data"})
value = context.get_var("result.output")  # Returns "data"
```

## Using `batch_process` and Related Utilities

### Basic Usage

```python
import asyncio
from scout.batch_pipeline import PipelineExecutor
from scout.batch_context import BatchContext

async def run_batch():
    context = BatchContext()
    
    async def task_runner(task, semaphore, index):
        # Execute the task
        command = task.get("command")
        args = task.get("args", {})
        # ... execute command ...
        return {"status": "success", "output": "done"}
    
    executor = PipelineExecutor(
        context=context,
        task_runner=task_runner,
    )
    
    tasks = [
        {"command": "lint", "args": {"paths": ["."]}},
        {"command": "test", "args": {}},
    ]
    
    results = await executor.run(tasks, mode="sequential")
    return results

results = asyncio.run(run_batch())
```

### With Retry Configuration

```python
from scout.retry import RetryConfig

retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    jitter_factor=0.1
)

executor = PipelineExecutor(
    context=context,
    task_runner=task_runner,
    retry_config=retry_config
)
```

### With Circuit Breaker

```python
from scout.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

cb_config = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=2,
    timeout_seconds=30.0
)
circuit_breaker = CircuitBreaker(config=cb_config)

executor = PipelineExecutor(
    context=context,
    task_runner=task_runner,
    circuit_breaker=circuit_breaker
)
```

## Examples

### Example 1: Sequential Build Pipeline

```python
tasks = [
    {
        "command": "run",
        "args": {"cmd": "npm install"},
        "store_as": "install_result"
    },
    {
        "command": "run",
        "args": {"cmd": "npm run lint"},
        "skip_if": "${install_result.failed}"
    },
    {
        "command": "run",
        "args": {"cmd": "npm run test"},
        "skip_if": "${install_result.failed}"
    },
    {
        "command": "run",
        "args": {"cmd": "npm run build"},
        "store_as": "build_result"
    }
]

results = await executor.run(tasks, mode="sequential")
```

### Example 2: Parallel File Processing

```python
tasks = [
    {"command": "lint", "args": {"paths": ["file1.py"]}},
    {"command": "lint", "args": {"paths": ["file2.py"]}},
    {"command": "lint", "args": {"paths": ["file3.py"]}},
]

results = await executor.run(tasks, mode="parallel")
```

### Example 3: CI Pipeline with Early Exit

```python
tasks = [
    {
        "command": "run",
        "args": {"cmd": "npm run lint"},
        "store_as": "lint_result"
    },
    {
        "command": "run",
        "args": {"cmd": "npm run test"},
        "store_as": "test_result",
        "stop_if": "${lint_result.failed}"
    },
    {
        "command": "run",
        "args": {"cmd": "npm run build"},
        "store_as": "build_result",
        "stop_if": "${test_result.failed}"
    },
    {
        "command": "run",
        "args": {"cmd": "npm run deploy"},
        "stop_if": "${build_result.failed}"
    }
]
```

### Example 4: Sub-batch Execution

Extract steps from a plan and execute them as a sub-batch:

```python
tasks = [
    {
        "command": "plan",
        "args": {"prompt": "Add authentication to the app"},
        "extract_steps": True,
        "subbatch_mode": "sequential"
    }
]
```

## Progress Reporting

The pipeline emits progress events that you can capture:

```python
from scout.progress import ProgressReporter, Status

reporter = ProgressReporter()

@reporter.on(ProgressEvent)
async def handle_event(event):
    print(f"{event.status}: {event.message}")
    if event.status == Status.COMPLETE:
        print(f"Metadata: {event.metadata}")

executor.set_reporter(reporter)
```

### Status Values

- `RUNNING`: Task is executing
- `SUCCESS`: Task completed successfully
- `FAILURE`: Task failed
- `SKIPPED`: Task was skipped (conditional)
- `RETRY`: Task is being retried
- `CIRCUIT_OPEN`: Circuit breaker is open
- `COMPLETE`: Pipeline finished

## Auto-JSON Flag Injection

Commands that require structured output automatically receive `--json` flags:

```python
# These commands automatically get JSON flags:
COMMANDS_NEED_JSON = {
    "git_status": "json",
    "git_branch": "json",
    "git_diff": "json",
    "plan": "json_output",
    "lint": "json_output",
    "audit": "json_output",
    "run": "json_output",
    # ... more commands
}
```

This enables transparent variable interpolation without manual flag specification.

## Related Documentation

- [ADR-008: Batch Pipeline](../adr/ADR-008-batch-pipeline.md) - Design decisions
- [ADR-005: Circuit Breaker](../adr/ADR-005-circuit-breaker-pattern.md) - Failure handling
- [ADR-010: Retry Mechanisms](../adr/ADR-010-retry-mechanisms.md) - Retry configuration
