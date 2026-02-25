"""Tests for execution module."""

import asyncio
import json
import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from scout.execution import (
    ActionType,
    StructuredStep,
    StructuredPlan,
    StepResult,
    ExecutionResult,
    PlanExecutor,
    ExecutionToolRegistry,
    SafetyGuard,
    StepToToolMapper,
    LlmProseParser,
)
from scout.execution.actions import (
    WebStep,
    PlanContext,
    PlanResult,
    StepResult as WebStepResult,
)
from scout.execution.safety import (
    SafetyViolation,
    DEFAULT_COMMAND_WHITELIST,
    scout_mkdir,
    scout_remove,
    scout_copy,
    scout_move,
    scout_list,
    scout_read_file,
    scout_write_file,
    scout_command,
    scout_check_command,
    scout_wait,
    scout_condition,
)
from scout.execution.executor import (
    ChangeRecord,
    RollbackManager,
    BudgetGuard,
)
from scout.execution.registry import ToolContract


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

    def test_registry_default_mappings(self):
        """Test default tool mappings."""
        registry = ExecutionToolRegistry()
        assert registry.get_tool_name(ActionType.CREATE_FILE) == "scout_create_file"
        assert registry.get_tool_name(ActionType.MODIFY_FILE) == "scout_edit"
        assert registry.get_tool_name(ActionType.RUN_COMMAND) == "bash"
        assert registry.get_tool_name(ActionType.READ_FILE) == "scout_read_file"

    def test_registry_register(self):
        """Test registering a new tool."""
        registry = ExecutionToolRegistry()
        registry.register(ActionType.DELETE_FILE, "scout_delete_file")
        assert registry.get_tool_name(ActionType.DELETE_FILE) == "scout_delete_file"

    def test_registry_get_adapter(self):
        """Test getting adapter function."""
        registry = ExecutionToolRegistry()
        adapter = registry.get_adapter(ActionType.CREATE_FILE)
        assert adapter is None

    def test_registry_with_adapter(self):
        """Test registering with adapter function."""
        registry = ExecutionToolRegistry()
        mock_adapter = AsyncMock()
        registry.register(ActionType.CREATE_FILE, "scout_create_file", mock_adapter)
        adapter = registry.get_adapter(ActionType.CREATE_FILE)
        assert adapter is mock_adapter


class TestToolContract:
    """Test ToolContract class."""

    def test_tool_contract_creation(self):
        """Test creating a tool contract."""
        contract_data = {
            "id": "test_tool",
            "version": "1.0.0",
            "name": "Test Tool",
            "description": "A test tool",
            "input_schema": {"type": "object"},
            "output_schema": {"type": "object"},
        }
        contract = ToolContract(contract_data)
        assert contract.id == "test_tool"
        assert contract.version == "1.0.0"
        assert contract.name == "Test Tool"

    def test_tool_contract_is_compatible(self):
        """Test version compatibility check."""
        contract1 = ToolContract({"id": "test", "version": "1.0.0"})
        contract2 = ToolContract({"id": "test", "version": "1.1.0"})
        contract3 = ToolContract({"id": "test", "version": "2.0.0"})
        assert contract1.is_compatible_with(contract2) is True
        assert contract1.is_compatible_with(contract3) is False

    def test_tool_contract_estimate_cost(self):
        """Test cost estimation."""
        contract = ToolContract({
            "id": "test",
            "cost_estimate": {
                "type": "function",
                "params": {"input_size_kb": "input_size_kb:0.001"},
                "min": 0.001,
                "max": 1.0
            }
        })
        cost = contract.estimate_cost({"data": "x" * 1000})
        assert cost > 0


class TestSafetyGuard:
    """Test SafetyGuard class."""

    def test_safety_guard_creation(self):
        """Test creating a safety guard."""
        workspace = Path("/tmp/test_workspace")
        guard = SafetyGuard(workspace_root=workspace)
        assert guard is not None
        assert guard.workspace_root == workspace.resolve()

    def test_safety_guard_validate_path(self):
        """Test path validation prevents directory traversal."""
        workspace = Path("/tmp/test_workspace")
        guard = SafetyGuard(workspace_root=workspace)
        
        # Valid paths within workspace should pass
        valid_path = guard.validate_path("test.txt")
        assert valid_path is not None
        assert "test_workspace" in str(valid_path)
        
        valid_subdir = guard.validate_path("subdir/test.txt")
        assert valid_subdir is not None
        assert "subdir" in str(valid_subdir)
        
        # Path traversal attempts should be rejected
        with pytest.raises(SafetyViolation):
            guard.validate_path("../test.txt")
        
        with pytest.raises(SafetyViolation):
            guard.validate_path("..")
        
        with pytest.raises(SafetyViolation):
            guard.validate_path("subdir/../../etc/passwd")
        
        # Absolute paths outside workspace should be rejected
        with pytest.raises(SafetyViolation):
            guard.validate_path("/etc/passwd")
        
        with pytest.raises(SafetyViolation):
            guard.validate_path("/tmp/other_file.txt")

    def test_safety_guard_validate_command(self):
        """Test command validation."""
        workspace = Path("/tmp/test_workspace")
        guard = SafetyGuard(workspace_root=workspace)
        assert guard.validate_command("git", DEFAULT_COMMAND_WHITELIST) is True

    def test_safety_guard_validate_command_not_in_whitelist(self):
        """Test command validation rejects non-whitelisted commands."""
        workspace = Path("/tmp/test_workspace")
        guard = SafetyGuard(workspace_root=workspace)
        with pytest.raises(SafetyViolation):
            guard.validate_command("rm -rf /", DEFAULT_COMMAND_WHITELIST)

    def test_safety_guard_check_depth(self):
        """Test depth checking."""
        workspace = Path("/tmp/test_workspace")
        guard = SafetyGuard(workspace_root=workspace)
        assert guard.check_depth("a/b/c", max_depth=5) is True

    def test_safety_guard_check_depth_exceeds(self):
        """Test depth checking rejects deep paths."""
        workspace = Path("/tmp/test_workspace")
        guard = SafetyGuard(workspace_root=workspace)
        with pytest.raises(SafetyViolation):
            guard.check_depth("a/b/c/d/e/f", max_depth=5)

    def test_safety_guard_dry_run(self):
        """Test dry run mode."""
        workspace = Path("/tmp/test_workspace")
        guard = SafetyGuard(workspace_root=workspace, dry_run=True)
        assert guard.check_dry_run() is True


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

    def test_plan_executor_custom_budget(self):
        """Test custom budget value."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry, max_budget=0.50)
        assert executor.max_budget == 0.50

    @pytest.mark.asyncio
    async def test_execute_empty_plan(self):
        """Test executing an empty plan."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        plan = StructuredPlan(steps=[], raw_plan="Empty plan")
        result = await executor.execute(plan)
        assert result.steps_completed == 0
        assert result.steps_failed == 0

    @pytest.mark.asyncio
    async def test_execute_simple_step(self):
        """Test executing a simple step."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        # Create a mock adapter
        async def mock_adapter(step):
            return {"executed": True}
        
        registry.register(ActionType.CREATE_FILE, "scout_create_file", mock_adapter)
        
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create file",
            file_path="/tmp/test.txt",
            content="test"
        )
        plan = StructuredPlan(steps=[step], raw_plan="Create file")
        result = await executor.execute(plan)
        assert result.steps_completed == 1

    @pytest.mark.asyncio
    async def test_execute_with_dependencies(self):
        """Test executing steps with dependencies."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        async def mock_adapter(step):
            return {"executed": True}
        
        registry.register(ActionType.CREATE_FILE, "scout_create_file", mock_adapter)
        registry.register(ActionType.READ_FILE, "scout_read_file", mock_adapter)
        
        step1 = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create file",
            file_path="/tmp/test.txt",
            content="test",
            step_id=0
        )
        step2 = StructuredStep(
            action_type=ActionType.READ_FILE,
            description="Read file",
            file_path="/tmp/test.txt",
            step_id=1,
            depends_on=[0]
        )
        
        plan = StructuredPlan(steps=[step2, step1], raw_plan="Two steps")
        result = await executor.execute(plan)
        assert result.steps_completed == 2

    @pytest.mark.asyncio
    async def test_execute_fails_on_missing_tool(self):
        """Test execution fails when tool not found."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        step = StructuredStep(
            action_type=ActionType.DELETE_FILE,  # Not registered
            description="Delete file",
            file_path="/tmp/test.txt",
            step_id=0
        )
        
        plan = StructuredPlan(steps=[step], raw_plan="Delete file")
        result = await executor.execute(plan)
        assert result.steps_failed == 1

    @pytest.mark.asyncio
    async def test_execute_budget_exceeded(self):
        """Test execution stops when budget exceeded."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry, max_budget=0.0005)  # Very low budget
        
        async def mock_adapter(step):
            return {"executed": True}
        
        registry.register(ActionType.CREATE_FILE, "scout_create_file", mock_adapter)
        
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create file",
            file_path="/tmp/test.txt",
            content="test",
            step_id=0
        )
        plan = StructuredPlan(steps=[step], raw_plan="Create file")
        result = await executor.execute(plan)
        assert result.steps_failed == 1

    @pytest.mark.asyncio
    async def test_execute_with_exception(self):
        """Test execution handles exceptions."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        async def failing_adapter(step):
            raise RuntimeError("Test error")
        
        registry.register(ActionType.CREATE_FILE, "scout_create_file", failing_adapter)
        
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create file",
            file_path="/tmp/test.txt",
            content="test",
            step_id=0
        )
        plan = StructuredPlan(steps=[step], raw_plan="Create file")
        result = await executor.execute(plan)
        assert result.steps_failed == 1

    def test_build_execution_batches_no_dependencies(self):
        """Test building batches with no dependencies."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        step1 = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step 1", step_id=0)
        step2 = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step 2", step_id=1)
        
        batches = executor._build_execution_batches([step1, step2])
        assert len(batches) == 1
        assert len(batches[0]) == 2

    def test_build_execution_batches_with_dependencies(self):
        """Test building batches with dependencies."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        step1 = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step 1", step_id=0)
        step2 = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step 2", step_id=1, depends_on=[0])
        
        batches = executor._build_execution_batches([step1, step2])
        assert len(batches) == 2
        assert batches[0][0].step_id == 0
        assert batches[1][0].step_id == 1

    def test_build_execution_batches_complex_dependencies(self):
        """Test building batches with complex dependencies."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        # 0 -> 1 -> 2
        #      1 -> 3
        step0 = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step 0", step_id=0)
        step1 = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step 1", step_id=1, depends_on=[0])
        step2 = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step 2", step_id=2, depends_on=[1])
        step3 = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step 3", step_id=3, depends_on=[1])
        
        batches = executor._build_execution_batches([step0, step1, step2, step3])
        assert len(batches) == 3

    def test_can_execute_no_dependencies(self):
        """Test can_execute with satisfied dependencies."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        step = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step", step_id=1)
        results = {
            0: StepResult(step_id=0, success=True)
        }
        
        assert executor._can_execute(step, results) is True

    def test_can_execute_missing_dependency(self):
        """Test can_execute with missing dependency."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        step = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step", step_id=1, depends_on=[0])
        results = {}
        
        assert executor._can_execute(step, results) is False

    def test_can_execute_failed_dependency(self):
        """Test can_execute with failed dependency."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        step = StructuredStep(action_type=ActionType.CREATE_FILE, description="Step", step_id=1, depends_on=[0])
        results = {
            0: StepResult(step_id=0, success=False)
        }
        
        assert executor._can_execute(step, results) is False

    @pytest.mark.asyncio
    async def test_execute_user_input_step(self):
        """Test executing a GET_USER_INPUT step."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        step = StructuredStep(
            action_type=ActionType.GET_USER_INPUT,
            description="Get user input",
            step_id=0
        )
        
        plan = StructuredPlan(steps=[step], raw_plan="Get input")
        result = await executor.execute(plan)
        assert result.steps_completed == 1

    @pytest.mark.asyncio
    async def test_execute_discovers_data(self):
        """Test execution collects discoveries."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        async def mock_adapter(step):
            return {"discovery": True, "discovery_type": "test", "discovery_detail": "info"}
        
        registry.register(ActionType.CREATE_FILE, "scout_create_file", mock_adapter)
        
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create file",
            file_path="/tmp/test.txt",
            content="test",
            step_id=0
        )
        plan = StructuredPlan(steps=[step], raw_plan="Create file")
        result = await executor.execute(plan)
        assert len(result.discoveries) == 1

    @pytest.mark.asyncio
    async def test_execute_with_rollback(self):
        """Test execution with rollback recording."""
        registry = ExecutionToolRegistry()
        executor = PlanExecutor(registry=registry)
        
        async def mock_adapter(step):
            return {"executed": True}
        
        registry.register(ActionType.CREATE_FILE, "scout_create_file", mock_adapter)
        
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create file",
            file_path="/tmp/test.txt",
            content="test",
            step_id=0,
            rollback_on_fail=True
        )
        plan = StructuredPlan(steps=[step], raw_plan="Create file")
        result = await executor.execute(plan)
        
        # Check rollback manager recorded the change
        assert len(executor.rollback_manager._change_log) == 1


class TestRollbackManager:
    """Test RollbackManager class."""

    def test_rollback_manager_creation(self):
        """Test creating a rollback manager."""
        manager = RollbackManager()
        assert manager is not None
        assert len(manager._change_log) == 0

    def test_rollback_manager_record(self):
        """Test recording a change."""
        manager = RollbackManager()
        manager.record(1, "create", "rm file.txt", {"path": "file.txt"})
        assert len(manager._change_log) == 1

    @pytest.mark.asyncio
    async def test_rollback_manager_rollback_to(self):
        """Test rolling back to a specific step."""
        manager = RollbackManager()
        manager.record(1, "create", "rm file1.txt", {"path": "file1.txt"})
        manager.record(2, "modify", "git checkout", {"path": "file2.txt"})
        results = await manager.rollback_to(1)
        assert len(results) == 1

    def test_rollback_manager_generate_undo_plan(self):
        """Test generating an undo plan."""
        manager = RollbackManager()
        manager.record(1, "create", "rm file1.txt", {"path": "file1.txt"})
        manager.record(2, "modify", "git checkout", {"path": "file2.txt"})
        plan = manager.generate_undo_plan(0)
        assert len(plan.steps) == 2


class TestBudgetGuard:
    """Test BudgetGuard class."""

    def test_budget_guard_creation(self):
        """Test creating a budget guard."""
        guard = BudgetGuard(max_budget=1.0)
        assert guard.max_budget == 1.0
        assert guard.current_cost == 0.0

    @pytest.mark.asyncio
    async def test_budget_guard_check_passes(self):
        """Test budget check passes when under limit."""
        guard = BudgetGuard(max_budget=1.0)
        step = StructuredStep(action_type=ActionType.CREATE_FILE, description="test")
        result = await guard.check(step, 0.5)
        assert result is True

    @pytest.mark.asyncio
    async def test_budget_guard_check_fails(self):
        """Test budget check fails when over limit."""
        guard = BudgetGuard(max_budget=1.0)
        step = StructuredStep(action_type=ActionType.CREATE_FILE, description="test")
        result = await guard.check(step, 1.5)
        assert result is False

    @pytest.mark.asyncio
    async def test_budget_guard_update(self):
        """Test updating budget."""
        guard = BudgetGuard(max_budget=1.0)
        await guard.update(0.5)
        assert guard.current_cost == 0.5


class TestScoutFunctions:
    """Test scout_* utility functions."""

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace."""
        workspace = tempfile.mkdtemp()
        yield Path(workspace)
        shutil.rmtree(workspace, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_scout_mkdir(self, temp_workspace):
        """Test creating a directory."""
        result = await scout_mkdir(str(temp_workspace / "test_dir"))
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_scout_mkdir_already_exists(self, temp_workspace):
        """Test creating a directory that already exists."""
        (temp_workspace / "test_dir").mkdir()
        result = await scout_mkdir(str(temp_workspace / "test_dir"))
        assert result["success"] is True
        assert "already exists" in result.get("note", "")

    @pytest.mark.asyncio
    async def test_scout_remove_file(self, temp_workspace):
        """Test removing a file."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")
        result = await scout_remove(str(test_file))
        assert result["success"] is True
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_scout_remove_nonexistent(self, temp_workspace):
        """Test removing a nonexistent path."""
        result = await scout_remove(str(temp_workspace / "nonexistent.txt"))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_scout_copy_file(self, temp_workspace):
        """Test copying a file."""
        source = temp_workspace / "source.txt"
        source.write_text("content")
        dest = temp_workspace / "dest.txt"
        result = await scout_copy(str(source), str(dest))
        assert result["success"] is True
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_scout_copy_nonexistent_source(self, temp_workspace):
        """Test copying with nonexistent source."""
        result = await scout_copy(str(temp_workspace / "nonexistent.txt"), str(temp_workspace / "dest.txt"))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_scout_move_file(self, temp_workspace):
        """Test moving a file."""
        source = temp_workspace / "source.txt"
        source.write_text("content")
        dest = temp_workspace / "dest.txt"
        result = await scout_move(str(source), str(dest))
        assert result["success"] is True
        assert not source.exists()
        assert dest.exists()

    @pytest.mark.asyncio
    async def test_scout_list_directory(self, temp_workspace):
        """Test listing directory contents."""
        (temp_workspace / "file1.txt").write_text("content")
        (temp_workspace / "file2.txt").write_text("content")
        result = await scout_list(str(temp_workspace))
        assert result["success"] is True
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_scout_list_with_pattern(self, temp_workspace):
        """Test listing with pattern filter."""
        (temp_workspace / "file1.txt").write_text("content")
        (temp_workspace / "file2.py").write_text("content")
        result = await scout_list(str(temp_workspace), pattern="*.txt")
        assert result["success"] is True
        assert result["count"] == 1

    @pytest.mark.asyncio
    async def test_scout_read_file(self, temp_workspace):
        """Test reading a file."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Hello, World!")
        result = await scout_read_file(str(test_file))
        assert result["success"] is True
        assert result["content"] == "Hello, World!"

    @pytest.mark.asyncio
    async def test_scout_read_nonexistent_file(self, temp_workspace):
        """Test reading a nonexistent file."""
        result = await scout_read_file(str(temp_workspace / "nonexistent.txt"))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_scout_read_directory(self, temp_workspace):
        """Test reading a directory."""
        result = await scout_read_file(str(temp_workspace))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_scout_write_file(self, temp_workspace):
        """Test writing a file."""
        test_file = temp_workspace / "test.txt"
        result = await scout_write_file(str(test_file), "Hello, World!")
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_scout_write_file_append(self, temp_workspace):
        """Test appending to a file."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Hello")
        result = await scout_write_file(str(test_file), " World!", append=True)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_scout_command(self, temp_workspace):
        """Test running a command."""
        result = await scout_command("echo", ["hello"], cwd=str(temp_workspace))
        assert result["success"] is True
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_scout_command_timeout(self, temp_workspace):
        """Test command timeout."""
        result = await scout_command("sleep", ["5"], timeout=1, cwd=str(temp_workspace))
        assert result["success"] is False
        assert "timed out" in result["error"]

    @pytest.mark.asyncio
    async def test_scout_check_command_exists(self):
        """Test checking if command exists."""
        result = await scout_check_command("python3")
        assert result["success"] is True
        assert result["available"] is True

    @pytest.mark.asyncio
    async def test_scout_check_command_not_exists(self):
        """Test checking if command doesn't exist."""
        result = await scout_check_command("nonexistent_command_xyz")
        assert result["success"] is True
        assert result["available"] is False

    @pytest.mark.asyncio
    async def test_scout_wait_seconds(self):
        """Test waiting for fixed seconds."""
        result = await scout_wait(seconds=0.1)
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_scout_wait_condition_timeout(self):
        """Test waiting with condition that times out."""
        result = await scout_wait(
            condition="file_exists",
            condition_params={"path": "/nonexistent/file.txt"},
            max_wait=1
        )
        assert result["success"] is True
        assert result["condition_met"] is False

    @pytest.mark.asyncio
    async def test_scout_condition_file_exists(self, temp_workspace):
        """Test file_exists condition."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("content")
        result = await scout_condition("file_exists", {"path": str(test_file)})
        assert result["success"] is True
        assert result["result"] is True

    @pytest.mark.asyncio
    async def test_scout_condition_file_contains(self, temp_workspace):
        """Test file_contains condition."""
        test_file = temp_workspace / "test.txt"
        test_file.write_text("Hello, World!")
        result = await scout_condition("file_contains", {"path": str(test_file), "pattern": "World"})
        assert result["success"] is True
        assert result["result"] is True

    @pytest.mark.asyncio
    async def test_scout_condition_unknown(self):
        """Test unknown condition type."""
        result = await scout_condition("unknown_type", {})
        assert result["success"] is False


class TestWebStep:
    """Test WebStep dataclass."""

    def test_web_step_creation(self):
        """Test creating a WebStep."""
        step = WebStep(action="navigate", url="https://example.com")
        assert step.action == "navigate"
        assert step.url == "https://example.com"

    def test_web_step_defaults(self):
        """Test default values."""
        step = WebStep(action="click")
        assert step.max_retries == 1
        assert step.timeout_seconds == 30

    def test_web_step_empty_action(self):
        """Test WebStep with empty action defaults to click."""
        step = WebStep(action="")
        assert step.action == "click"

    def test_web_step_to_browser_act_params(self):
        """Test converting to browser_act parameters."""
        step = WebStep(
            action="click",
            target="Submit button",
            value="test value",
            url="https://example.com"
        )
        params = step.to_browser_act_params()
        assert params["action"] == "click"
        assert params["goal"] == "Submit button"
        assert params["value"] == "test value"
        assert params["url"] == "https://example.com"


class TestPlanContext:
    """Test PlanContext dataclass."""

    def test_plan_context_creation(self):
        """Test creating a PlanContext."""
        context = PlanContext()
        assert context.extracted_data == {}
        assert context.current_url is None


class TestPlanResult:
    """Test PlanResult dataclass."""

    def test_plan_result_creation(self):
        """Test creating a PlanResult."""
        result = PlanResult(
            success=True,
            plan_id="plan123",
            steps_executed=5,
            steps_failed=0,
            step_results=[]
        )
        assert result.success is True
        assert result.plan_id == "plan123"


class TestStepToToolMapper:
    """Test StepToToolMapper class."""

    def test_mapper_creation(self):
        """Test creating a mapper."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        assert mapper is not None

    def test_map_create_file(self):
        """Test mapping CREATE_FILE action."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create file",
            file_path="/tmp/test.txt",
            content="hello"
        )
        result = mapper.map(step)
        assert result is not None
        assert result["tool_name"] == "scout_create_file"
        assert result["arguments"]["path"] == "/tmp/test.txt"

    def test_map_modify_file(self):
        """Test mapping MODIFY_FILE action."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        step = StructuredStep(
            action_type=ActionType.MODIFY_FILE,
            description="Modify file",
            file_path="/tmp/test.txt",
            content="updated"
        )
        result = mapper.map(step)
        assert result["tool_name"] == "scout_edit"

    def test_map_delete_file(self):
        """Test mapping DELETE_FILE action."""
        registry = ExecutionToolRegistry()
        # Register DELETE_FILE mapping
        registry.register(ActionType.DELETE_FILE, "scout_delete_file")
        mapper = StepToToolMapper(registry)
        step = StructuredStep(
            action_type=ActionType.DELETE_FILE,
            description="Delete file",
            file_path="/tmp/test.txt"
        )
        result = mapper.map(step)
        assert result is not None
        assert result["tool_name"] == "scout_delete_file"

    def test_map_run_command(self):
        """Test mapping RUN_COMMAND action."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        step = StructuredStep(
            action_type=ActionType.RUN_COMMAND,
            description="Run command",
            command="echo hello"
        )
        result = mapper.map(step)
        assert result["tool_name"] == "bash"
        assert result["arguments"]["command"] == "echo hello"

    def test_map_read_file(self):
        """Test mapping READ_FILE action."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        step = StructuredStep(
            action_type=ActionType.READ_FILE,
            description="Read file",
            file_path="/tmp/test.txt"
        )
        result = mapper.map(step)
        assert result["tool_name"] == "scout_read_file"

    def test_map_unknown_action(self):
        """Test mapping UNKNOWN action."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        step = StructuredStep(
            action_type=ActionType.UNKNOWN,
            description="Unknown action"
        )
        result = mapper.map(step)
        assert result is None

    def test_map_missing_file_path(self):
        """Test mapping with missing file_path."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create file"
        )
        result = mapper.map(step)
        assert result is None

    def test_map_missing_command(self):
        """Test mapping RUN_COMMAND without command."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        step = StructuredStep(
            action_type=ActionType.RUN_COMMAND,
            description="Run command"
        )
        result = mapper.map(step)
        assert result is None


class TestLlmProseParser:
    """Test LlmProseParser class."""

    def test_parser_creation(self):
        """Test creating a parser."""
        mock_client = Mock()
        mock_client.complete = Mock(return_value={"content": "[]", "input_tokens": 10, "output_tokens": 5})
        parser = LlmProseParser(llm_client=mock_client)
        assert parser is not None
        assert parser.max_retries == 3

    def test_parser_with_custom_max_retries(self):
        """Test parser with custom max_retries."""
        mock_client = Mock()
        mock_client.complete = Mock(return_value={"content": "[]"})
        parser = LlmProseParser(llm_client=mock_client, max_retries=5)
        assert parser.max_retries == 5

    def test_parse_valid_json(self):
        """Test parsing valid JSON response."""
        mock_client = Mock()
        mock_client.complete = Mock(return_value={
            "content": '[{"action_type": "CREATE_FILE", "description": "Create file", "file_path": "test.txt", "content": "hello"}]',
            "input_tokens": 10,
            "output_tokens": 20
        })
        parser = LlmProseParser(llm_client=mock_client)
        steps = parser.parse("Create a test file")
        assert len(steps) == 1
        assert steps[0].action_type == ActionType.CREATE_FILE

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON."""
        mock_client = Mock()
        mock_client.complete = Mock(return_value={"content": "not valid json"})
        parser = LlmProseParser(llm_client=mock_client)
        with pytest.raises(Exception):
            parser.parse("Create a test file")

    def test_parse_with_markdown_code_block(self):
        """Test parsing JSON in markdown code block."""
        mock_client = Mock()
        mock_client.complete = Mock(return_value={
            "content": "```json\n[{\"action_type\": \"CREATE_FILE\", \"description\": \"Create file\", \"file_path\": \"test.txt\"}]\n```",
            "input_tokens": 10,
            "output_tokens": 20
        })
        parser = LlmProseParser(llm_client=mock_client)
        steps = parser.parse("Create a test file")
        assert len(steps) == 1

    def test_parse_missing_required_field(self):
        """Test parsing with missing required field."""
        mock_client = Mock()
        mock_client.complete = Mock(return_value={
            "content": '[{"description": "Create file"}]',
            "input_tokens": 10,
            "output_tokens": 20
        })
        parser = LlmProseParser(llm_client=mock_client)
        with pytest.raises(Exception):
            parser.parse("Create a test file")

    def test_parse_llm_error(self):
        """Test handling LLM errors."""
        mock_client = Mock()
        mock_client.complete = Mock(side_effect=TimeoutError("LLM timeout"))
        parser = LlmProseParser(llm_client=mock_client, max_retries=2)
        with pytest.raises(Exception):
            parser.parse("Create a test file")

    def test_parse_action_type_normalization(self):
        """Test action type normalization."""
        mock_client = Mock()
        mock_client.complete = Mock(return_value={
            "content": '[{"action_type": "create-file", "description": "Create file", "file_path": "test.txt"}]',
            "input_tokens": 10,
            "output_tokens": 20
        })
        parser = LlmProseParser(llm_client=mock_client)
        steps = parser.parse("Create a test file")
        assert steps[0].action_type == ActionType.CREATE_FILE

    def test_parse_unknown_action_type(self):
        """Test handling unknown action type."""
        mock_client = Mock()
        mock_client.complete = Mock(return_value={
            "content": '[{"action_type": "UNKNOWN_ACTION", "description": "Unknown", "file_path": "test.txt"}]',
            "input_tokens": 10,
            "output_tokens": 20
        })
        parser = LlmProseParser(llm_client=mock_client)
        steps = parser.parse("Create a test file")
        assert steps[0].action_type == ActionType.UNKNOWN

    def test_parse_file_action_without_path(self):
        """Test file action without file_path gets converted to UNKNOWN."""
        mock_client = Mock()
        mock_client.complete = Mock(return_value={
            "content": '[{"action_type": "CREATE_FILE", "description": "Create file"}]',
            "input_tokens": 10,
            "output_tokens": 20
        })
        parser = LlmProseParser(llm_client=mock_client)
        steps = parser.parse("Create a file")
        assert steps[0].action_type == ActionType.UNKNOWN
