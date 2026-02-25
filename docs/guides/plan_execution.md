# Plan Execution Guide

The plan execution framework provides structured execution of multi-step tasks with dependency resolution, budget enforcement, and rollback capabilities.

## Overview

The execution framework transforms high-level plans into executable steps with:

- **Dependency resolution**: Automatically identifies which steps can run in parallel
- **Budget enforcement**: Prevents runaway costs during execution
- **Rollback capability**: Can undo changes if a step fails
- **State management**: Tracks execution progress and results

## Core Concepts

### StructuredPlan and StructuredStep

Plans are represented as structured data:

```python
from scout.execution.actions import (
    StructuredPlan,
    StructuredStep,
    ActionType,
)

# Define individual steps
steps = [
    StructuredStep(
        action_type=ActionType.CREATE_FILE,
        description="Create config file",
        file_path="config.yaml",
        content="version: 1.0",
        step_id=0,
        depends_on=[],
    ),
    StructuredStep(
        action_type=ActionType.RUN_COMMAND,
        description="Initialize project",
        command="npm install",
        step_id=1,
        depends_on=[0],  # Depends on step 0
    ),
]

# Create the plan
plan = StructuredPlan(
    steps=steps,
    raw_plan="Create config and initialize project",
    summary="Setup project with config and dependencies"
)
```

### Action Types

The framework supports various action types:

```python
class ActionType(Enum):
    CREATE_FILE = "create_file"      # Create a new file
    MODIFY_FILE = "modify_file"      # Modify an existing file
    DELETE_FILE = "delete_file"      # Delete a file
    RUN_COMMAND = "run_command"      # Execute a shell command
    READ_FILE = "read_file"         # Read file contents
    GET_USER_INPUT = "get_user_input" # Request user input
    BROWSER_ACT = "browser_act"     # Browser automation
    UNKNOWN = "unknown"             # Unknown action
```

### StructuredStep Fields

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | ActionType | Type of action to perform |
| `description` | str | Human-readable description |
| `file_path` | str | Path to file (for file operations) |
| `content` | str | File content (for create/modify) |
| `command` | str | Command to execute (for RUN_COMMAND) |
| `args` | List[str] | Command arguments |
| `step_id` | int | Unique identifier for this step |
| `depends_on` | List[int] | List of step_ids this depends on |
| `success_conditions` | dict | Conditions for step success |
| `max_retries` | int | Maximum retry attempts |
| `timeout_seconds` | int | Execution timeout |
| `rollback_on_fail` | bool | Whether to rollback on failure |

## Creating a Plan

### Basic Plan

```python
from scout.execution.actions import StructuredPlan, StructuredStep, ActionType

# Create a simple plan with sequential steps
steps = [
    StructuredStep(
        action_type=ActionType.CREATE_FILE,
        description="Create README",
        file_path="README.md",
        content="# My Project\n\nHello World",
        step_id=0,
        depends_on=[],
    ),
    StructuredStep(
        action_type=ActionType.RUN_COMMAND,
        description="Initialize git",
        command="git init",
        step_id=1,
        depends_on=[0],
    ),
]

plan = StructuredPlan(
    steps=steps,
    raw_plan="Create project and initialize git",
    summary="Setup new project"
)
```

### Plan with Dependencies

Steps that don't depend on each other can run in parallel:

```python
# These steps can run in parallel (no dependencies between them)
steps = [
    StructuredStep(
        action_type=ActionType.CREATE_FILE,
        description="Create app.py",
        file_path="app.py",
        content="print('hello')",
        step_id=0,
        depends_on=[],
    ),
    StructuredStep(
        action_type=ActionType.CREATE_FILE,
        description="Create requirements.txt",
        file_path="requirements.txt",
        content="requests>=2.28",
        step_id=1,
        depends_on=[],
    ),
    StructuredStep(
        action_type=ActionType.RUN_COMMAND,
        description="Install dependencies",
        command="pip install -r requirements.txt",
        step_id=2,
        depends_on=[1],  # Depends on requirements.txt being created
    ),
]
```

### Plan with Rollback

Mark steps for automatic rollback on failure:

```python
steps = [
    StructuredStep(
        action_type=ActionType.CREATE_FILE,
        description="Create backup",
        file_path="backup.txt",
        content="original data",
        step_id=0,
        depends_on=[],
        rollback_on_fail=True,  # Delete on failure
    ),
    StructuredStep(
        action_type=ActionType.MODIFY_FILE,
        description="Modify config",
        file_path="config.yaml",
        content="new config",
        step_id=1,
        depends_on=[0],
        rollback_on_fail=True,  # Restore original on failure
    ),
]
```

## Executing a Plan

### Using PlanExecutor

```python
import asyncio
from scout.execution.executor import PlanExecutor
from scout.execution.registry import ExecutionToolRegistry

async def execute_plan(plan):
    # Create registry (maps action types to tools)
    registry = ExecutionToolRegistry()
    
    # Create executor with budget limit
    executor = PlanExecutor(
        registry=registry,
        max_budget=0.10,  # $0.10 max cost
    )
    
    # Execute the plan
    result = await executor.execute(plan)
    
    return result

# Run the executor
result = asyncio.run(execute_plan(plan))
```

### Understanding ExecutionResult

After execution, you receive an `ExecutionResult`:

```python
@dataclass
class ExecutionResult:
    steps_completed: int      # Number of successful steps
    steps_failed: int         # Number of failed steps
    total_cost: float         # Total execution cost
    total_duration: float     # Duration in milliseconds
    discoveries: List[Dict]   # Any discoveries made
    rollback_commands: List[str]  # Commands to rollback changes
```

### Full Example

```python
import asyncio
from scout.execution.executor import PlanExecutor
from scout.execution.registry import ExecutionToolRegistry
from scout.execution.actions import (
    StructuredPlan,
    StructuredStep,
    ActionType,
)

async def main():
    # Define the steps
    steps = [
        StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create hello.py",
            file_path="hello.py",
            content="def greet():\n    return 'Hello, World!'",
            step_id=0,
            depends_on=[],
        ),
        StructuredStep(
            action_type=ActionType.RUN_COMMAND,
            description="Run the script",
            command="python hello.py",
            step_id=1,
            depends_on=[0],
        ),
    ]
    
    plan = StructuredPlan(
        steps=steps,
        raw_plan="Create and run hello.py",
        summary="Simple hello world example"
    )
    
    # Create executor
    registry = ExecutionToolRegistry()
    executor = PlanExecutor(registry=registry, max_budget=0.10)
    
    # Execute
    result = await executor.execute(plan)
    
    print(f"Completed: {result.steps_completed}/{len(steps)}")
    print(f"Failed: {result.steps_failed}")
    print(f"Cost: ${result.total_cost:.4f}")
    print(f"Duration: {result.total_duration}ms")

asyncio.run(main())
```

## Error Handling

### BudgetGuard

The budget guard prevents runaway costs:

```python
from scout.execution.executor import BudgetGuard

# Initialize with budget limit
budget_guard = BudgetGuard(max_budget=0.10)

# Check before execution
can_proceed = await budget_guard.check(step, estimated_cost)

# Update after execution
await budget_guard.update(actual_cost)
```

### RollbackManager

The rollback manager tracks changes for potential rollback:

```python
from scout.execution.executor import RollbackManager

rollback_manager = RollbackManager()

# Record a change
rollback_manager.record(
    step_id=0,
    change_type="create_file",
    undo_command="rm file.txt",
    metadata={"path": "file.txt"}
)

# Rollback to a specific step
results = await rollback_manager.rollback_to(step_id=0)

# Generate undo plan
undo_plan = rollback_manager.generate_undo_plan(from_step=0)
```

### Handling Step Failures

```python
async def _execute_step(self, step, results):
    try:
        # Execute the step
        output = await execute_step(step)
        return StepResult(
            step_id=step.step_id,
            success=True,
            output=output
        )
    except Exception as e:
        # Check if rollback is needed
        if step.rollback_on_fail:
            self.rollback_manager.record(
                step.step_id,
                str(step.action_type),
                f"undo_{step.step_id}",
                {}
            )
        
        return StepResult(
            step_id=step.step_id,
            success=False,
            error=str(e)
        )
```

## Dependency Resolution

The executor uses topological sorting (Kahn's algorithm) to determine execution order:

```python
# Example: Dependency graph
# Step 0: no dependencies
# Step 1: depends on [0]
# Step 2: depends on [0]
# Step 3: depends on [1, 2]

# Execution batches:
# Batch 1: [Step 0]      # No dependencies
# Batch 2: [Step 1, 2]  # Both depend on 0, can run in parallel
# Batch 3: [Step 3]      # Depends on both 1 and 2
```

This automatic batching ensures:
- All dependencies are satisfied before execution
- Maximum parallelization of independent steps

## Configuration Options

### PlanExecutor Configuration

```python
executor = PlanExecutor(
    registry=registry,
    max_budget=0.10,           # Maximum cost allowed
    discovery_callback=None,  # Callback for discoveries
)
```

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_budget` | $0.10 | Maximum execution cost |
| `timeout` | 300s | Default step timeout |
| `max_retries` | 2 | Default retry attempts |

## Best Practices

### 1. Define Clear Dependencies

Always specify dependencies explicitly:

```python
# Good: clear dependency
StructuredStep(
    step_id=1,
    depends_on=[0],  # Clearly depends on step 0
)

# Avoid: implicit assumptions about order
StructuredStep(step_id=1, depends_on=[])
```

### 2. Set Appropriate Budgets

```python
# Conservative for expensive operations
executor = PlanExecutor(max_budget=0.05)

# Generous for complex workflows
executor = PlanExecutor(max_budget=1.00)
```

### 3. Use Rollback for Risky Operations

```python
StructuredStep(
    action_type=ActionType.DELETE_FILE,
    file_path="important.txt",
    rollback_on_fail=True,  # Can restore if needed
)
```

### 4. Handle User Input Gracefully

```python
# Steps that need user input
StructuredStep(
    action_type=ActionType.GET_USER_INPUT,
    input_prompt="Enter version number:",
    input_schema={"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
)
```

## Related Documentation

- [ADR-006: Execution Framework](../adr/ADR-006-execution-framework.md) - Design decisions
- [ADR-007: Plan Executor State Machine](../adr/ADR-007-plan-executor-state-machine.md) - State persistence
- [ADR-010: Retry Mechanisms](../adr/ADR-010-retry-mechanisms.md) - Retry configuration
