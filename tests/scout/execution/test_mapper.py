"""Tests for execution/mapper module."""

import pytest
from unittest.mock import Mock, MagicMock

from scout.execution.mapper import StepToToolMapper
from scout.execution.actions import ActionType, StructuredStep
from scout.execution.registry import ExecutionToolRegistry


class TestStepToToolMapper:
    """Test StepToToolMapper class."""

    def test_map_create_file(self):
        """Test mapping CREATE_FILE action."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create a test file",
            file_path="/tmp/test.txt",
            content="Hello, World!"
        )
        
        result = mapper.map(step)
        
        assert result is not None
        assert result["tool_name"] == "scout_create_file"
        assert result["arguments"]["path"] == "/tmp/test.txt"
        assert result["arguments"]["content"] == "Hello, World!"

    def test_map_modify_file(self):
        """Test mapping MODIFY_FILE action."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.MODIFY_FILE,
            description="Modify a test file",
            file_path="/tmp/test.txt",
            content="Modified content"
        )
        
        result = mapper.map(step)
        
        assert result is not None
        assert result["tool_name"] == "scout_edit"
        assert result["arguments"]["path"] == "/tmp/test.txt"
        assert result["arguments"]["content"] == "Modified content"

    def test_map_delete_file(self):
        """Test mapping DELETE_FILE action - need to register first."""
        registry = ExecutionToolRegistry()
        # Register the DELETE_FILE action
        registry.register(ActionType.DELETE_FILE, "scout_remove")
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.DELETE_FILE,
            description="Delete a test file",
            file_path="/tmp/test.txt"
        )
        
        result = mapper.map(step)
        
        assert result is not None
        assert result["tool_name"] == "scout_remove"
        assert result["arguments"]["path"] == "/tmp/test.txt"

    def test_map_run_command(self):
        """Test mapping RUN_COMMAND action."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.RUN_COMMAND,
            description="Run a command",
            command="ls -la"
        )
        
        result = mapper.map(step)
        
        assert result is not None
        assert result["tool_name"] == "bash"
        assert result["arguments"]["command"] == "ls -la"

    def test_map_read_file(self):
        """Test mapping READ_FILE action."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.READ_FILE,
            description="Read a file",
            file_path="/tmp/test.txt"
        )
        
        result = mapper.map(step)
        
        assert result is not None
        assert result["tool_name"] == "scout_read_file"
        assert result["arguments"]["path"] == "/tmp/test.txt"

    def test_map_unknown_action(self):
        """Test mapping UNKNOWN action returns None."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.UNKNOWN,
            description="Unknown action"
        )
        
        result = mapper.map(step)
        
        assert result is None

    def test_map_create_file_no_path(self):
        """Test CREATE_FILE without file_path returns None."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create file without path"
            # No file_path
        )
        
        result = mapper.map(step)
        
        assert result is None

    def test_map_modify_file_no_path(self):
        """Test MODIFY_FILE without file_path returns None."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.MODIFY_FILE,
            description="Modify file without path"
            # No file_path
        )
        
        result = mapper.map(step)
        
        assert result is None

    def test_map_delete_file_no_path(self):
        """Test DELETE_FILE without file_path returns None."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.DELETE_FILE,
            description="Delete file without path"
            # No file_path
        )
        
        result = mapper.map(step)
        
        assert result is None

    def test_map_read_file_no_path(self):
        """Test READ_FILE without file_path returns None."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.READ_FILE,
            description="Read file without path"
            # No file_path
        )
        
        result = mapper.map(step)
        
        assert result is None

    def test_map_run_command_no_command(self):
        """Test RUN_COMMAND without command returns None."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.RUN_COMMAND,
            description="Run command without command"
            # No command
        )
        
        result = mapper.map(step)
        
        assert result is None

    def test_map_no_content(self):
        """Test that content defaults to empty string."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Create file without content",
            file_path="/tmp/test.txt"
            # No content
        )
        
        result = mapper.map(step)
        
        assert result is not None
        assert result["arguments"]["content"] == ""

    def test_map_unregistered_action(self):
        """Test mapping unregistered action returns None."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        # Create a step with an action type not in registry
        # Use browser_act which is not registered by default
        step = StructuredStep(
            action_type=ActionType.BROWSER_ACT,
            description="Browser action"
        )
        
        result = mapper.map(step)
        
        assert result is None


class TestBuildArguments:
    """Test _build_arguments method."""

    def test_build_arguments_create_file(self):
        """Test building arguments for CREATE_FILE."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Test",
            file_path="/path/to/file",
            content="test content"
        )
        
        args = mapper._build_arguments(step)
        
        assert args == {"path": "/path/to/file", "content": "test content"}

    def test_build_arguments_modify_file(self):
        """Test building arguments for MODIFY_FILE."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.MODIFY_FILE,
            description="Test",
            file_path="/path/to/file",
            content="new content"
        )
        
        args = mapper._build_arguments(step)
        
        assert args == {"path": "/path/to/file", "content": "new content"}

    def test_build_arguments_delete_file(self):
        """Test building arguments for DELETE_FILE."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.DELETE_FILE,
            description="Test",
            file_path="/path/to/file"
        )
        
        args = mapper._build_arguments(step)
        
        assert args == {"path": "/path/to/file"}

    def test_build_arguments_run_command(self):
        """Test building arguments for RUN_COMMAND."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.RUN_COMMAND,
            description="Test",
            command="echo hello"
        )
        
        args = mapper._build_arguments(step)
        
        assert args == {"command": "echo hello"}

    def test_build_arguments_read_file(self):
        """Test building arguments for READ_FILE."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.READ_FILE,
            description="Test",
            file_path="/path/to/file"
        )
        
        args = mapper._build_arguments(step)
        
        assert args == {"path": "/path/to/file"}

    def test_build_arguments_unknown_action(self):
        """Test building arguments for UNKNOWN action."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.UNKNOWN,
            description="Test"
        )
        
        args = mapper._build_arguments(step)
        
        assert args is None

    def test_build_arguments_create_file_no_path(self):
        """Test CREATE_FILE without path returns None."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.CREATE_FILE,
            description="Test"
            # No file_path
        )
        
        args = mapper._build_arguments(step)
        
        assert args is None

    def test_build_arguments_run_command_no_command(self):
        """Test RUN_COMMAND without command returns None."""
        registry = ExecutionToolRegistry()
        mapper = StepToToolMapper(registry)
        
        step = StructuredStep(
            action_type=ActionType.RUN_COMMAND,
            description="Test"
            # No command
        )
        
        args = mapper._build_arguments(step)
        
        assert args is None
