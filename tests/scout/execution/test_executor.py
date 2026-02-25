"""Tests for execution module."""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from scout.execution import (
    ActionType,
    StructuredStep,
    StructuredPlan,
    StepResult,
    ExecutionResult,
    PlanExecutor,
    ExecutionToolRegistry,
    SafetyGuard,
)


class TestActionTypes:
    """Test action type definitions."""

    def test_action_type_values(self):
        """Test that ActionType enum has expected values."""
        assert ActionType.CREATE_FILE.value == "create_file"
        assert ActionType.MODIFY_FILE.value == "modify_file"
        assert ActionType.DELETE_FILE.value == "delete_file"
        assert ActionType.RUN_COMMAND.value == "run_command"
        assert ActionType.READ_FILE.value == "read_file"

    def test_action_type_is_enum(self):
        """Test that ActionType is an Enum."""
        assert isinstance(ActionType.CREATE_FILE, ActionType)


class TestStructuredStep:
    """Test StructuredStep class."""

    def test_structured_step_creation(self):
        """Test creating a structured step."""
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create a test file",
            file_path="/tmp/test.txt",
            content="Hello, World!"
        )
        assert step.action_type == ActionType.CREATE_FILE
        assert step.description == "Create a test file"
        assert step.file_path == "/tmp/test.txt"
        assert step.content == "Hello, World!"


class TestStructuredPlan:
    """Test StructuredPlan class."""

    def test_structured_plan_creation(self):
        """Test creating a structured plan."""
        steps = [
            StructuredStep(
                action_type=ActionType.CREATE_FILE,
                description="Create file",
                file_path="/tmp/test.txt",
                content="content"
            ),
        ]
        plan = StructuredPlan(
            steps=steps,
            raw_plan="Create a test file"
        )
        assert len(plan.steps) == 1
        assert plan.raw_plan == "Create a test file"


class TestStepResult:
    """Test StepResult class (web step result)."""

    def test_step_result_success(self):
        """Test successful step result."""
        result = StepResult(
            step_index=1,
            action="navigate",
            success=True,
            output={"data": "test"}
        )
        assert result.success is True
        assert result.action == "navigate"
        assert result.output["data"] == "test"

    def test_step_result_failure(self):
        """Test failed step result."""
        result = StepResult(
            step_index=1,
            action="click",
            success=False,
            error="Element not found"
        )
        assert result.success is False
        assert result.error == "Element not found"


class TestExecutionToolRegistry:
    """Test ExecutionToolRegistry class."""

    def test_registry_creation(self):
        """Test creating a registry."""
        registry = ExecutionToolRegistry()
        assert registry is not None


class TestSafetyGuard:
    """Test SafetyGuard class."""

    def test_safety_guard_creation(self):
        """Test creating a safety guard."""
        workspace = Path("/tmp/test_workspace")
        guard = SafetyGuard(workspace_root=workspace)
        assert guard is not None
        assert guard.workspace_root == workspace.resolve()

    def test_safety_guard_validate_path(self):
        """Test path validation."""
        workspace = Path("/tmp/test_workspace")
        guard = SafetyGuard(workspace_root=workspace)
        # Valid path within workspace
        valid_path = guard.validate_path("test.txt")
        assert valid_path is not None


class TestPlanExecutor:
    """Test PlanExecutor class."""

    def test_plan_executor_creation(self):
        """Test creating a plan executor."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        assert executor is not None
        assert executor.registry == registry

    def test_plan_executor_default_budget(self):
        """Test default budget value."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        assert executor.max_budget == 0.10
