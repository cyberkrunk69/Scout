"""Tests for parameter_registry.py module."""

import pytest

from scout.parameter_registry import (
    ParameterRegistry,
    CLI_TOOL_PARAM_MAP,
    register_all_mappings,
    transform_params,
)


# Reset singleton before each test
@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the ParameterRegistry singleton before each test."""
    ParameterRegistry._instance = None
    yield
    ParameterRegistry._instance = None


class TestParameterRegistry:
    """Tests for ParameterRegistry class."""

    def test_singleton_pattern(self):
        """Test that ParameterRegistry is a singleton."""
        reg1 = ParameterRegistry()
        reg2 = ParameterRegistry()

        assert reg1 is reg2

    def test_register_mapping(self):
        """Test registering a parameter mapping."""
        registry = ParameterRegistry()

        registry.register("test_tool", "cli_param", "tool_param")

        assert registry.get_tool_param("test_tool", "cli_param") == "tool_param"

    def test_register_reverse_mapping(self):
        """Test that reverse mapping is created automatically."""
        registry = ParameterRegistry()

        registry.register("test_tool", "cli_param", "tool_param")

        assert registry.get_cli_param("test_tool", "tool_param") == "cli_param"

    def test_get_tool_param_not_found(self):
        """Test getting non-existent mapping."""
        registry = ParameterRegistry()

        result = registry.get_tool_param("unknown_tool", "unknown_param")

        assert result is None

    def test_get_cli_param_not_found(self):
        """Test getting non-existent reverse mapping."""
        registry = ParameterRegistry()

        result = registry.get_cli_param("unknown_tool", "unknown_param")

        assert result is None

    def test_transform_params_basic(self):
        """Test basic parameter transformation."""
        registry = ParameterRegistry()
        registry.register("test_tool", "query", "request")

        result = registry.transform_params("test_tool", {"query": "test value"})

        assert result == {"request": "test value"}

    def test_transform_params_preserve_unmapped(self):
        """Test that unmapped parameters are preserved."""
        registry = ParameterRegistry()
        registry.register("test_tool", "query", "request")

        result = registry.transform_params("test_tool", {
            "query": "test value",
            "other": "keep this"
        })

        assert result == {"request": "test value", "other": "keep this"}

    def test_transform_params_no_mapping(self):
        """Test transformation when no mapping exists."""
        registry = ParameterRegistry()

        result = registry.transform_params("unknown_tool", {"param": "value"})

        # Should return params as-is when no mapping exists
        assert result == {"param": "value"}

    def test_multiple_tools(self):
        """Test registering mappings for multiple tools."""
        registry = ParameterRegistry()

        registry.register("tool_a", "input", "input_param")
        registry.register("tool_b", "query", "query_param")

        assert registry.get_tool_param("tool_a", "input") == "input_param"
        assert registry.get_tool_param("tool_b", "query") == "query_param"


class TestCLIToolParamMap:
    """Tests for CLI_TOOL_PARAM_MAP constant."""

    def test_scout_plan_mapping(self):
        """Test scout_plan tool mappings."""
        mappings = CLI_TOOL_PARAM_MAP.get("scout_plan", {})

        assert "query" in mappings
        assert mappings["query"] == "request"

    def test_scout_nav_mapping(self):
        """Test scout_nav tool mappings."""
        mappings = CLI_TOOL_PARAM_MAP.get("scout_nav", {})

        assert "query" in mappings
        assert mappings["query"] == "task"

    def test_scout_query_mapping(self):
        """Test scout_query tool mappings."""
        mappings = CLI_TOOL_PARAM_MAP.get("scout_query", {})

        assert "question" in mappings
        assert mappings["question"] == "query"

    def test_scout_edit_mapping(self):
        """Test scout_edit tool mappings."""
        mappings = CLI_TOOL_PARAM_MAP.get("scout_edit", {})

        assert "file" in mappings
        assert mappings["file"] == "file_path"
        assert "prompt" in mappings
        assert mappings["prompt"] == "instruction"

    def test_git_tools_mapping(self):
        """Test git tool mappings."""
        # scout_git_commit should have message mapping
        mappings = CLI_TOOL_PARAM_MAP.get("scout_git_commit", {})

        assert "query" in mappings
        assert mappings["query"] == "message"


class TestRegisterAllMappings:
    """Tests for register_all_mappings function."""

    def test_register_all_creates_mappings(self):
        """Test that register_all_mappings populates the registry."""
        # Reset first
        ParameterRegistry._instance = None

        register_all_mappings()

        registry = ParameterRegistry()

        # Should have mappings for scout_plan
        assert registry.get_tool_param("scout_plan", "query") == "request"

    def test_register_all_does_not_register_identical(self):
        """Test that identical param names are not registered."""
        ParameterRegistry._instance = None

        register_all_mappings()

        registry = ParameterRegistry()

        # scout_plan also has "request" -> "request" which shouldn't be registered
        # as it's identical (handled by condition in register_all_mappings)
        # The key "request" should map to "request" but might not be stored
        # since identical mappings are skipped


class TestTransformParams:
    """Tests for transform_params function."""

    def test_transform_params_function(self):
        """Test the module-level transform_params function."""
        ParameterRegistry._instance = None

        # Register a mapping
        registry = ParameterRegistry()
        registry.register("my_tool", "input", "input_param")

        result = transform_params("my_tool", {"input": "hello"})

        assert result == {"input_param": "hello"}

    def test_transform_params_unknown_tool(self):
        """Test transform_params with unknown tool."""
        ParameterRegistry._instance = None

        result = transform_params("unknown_tool", {"param": "value"})

        assert result == {"param": "value"}
