"""Tests for validation tools."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scout.tools import validation


class TestValidationTools:
    """Test cases for validation tools."""

    def test_lint_is_callable(self):
        """Test that lint is callable."""
        assert callable(validation.scout_lint)

    def test_validate_module_is_callable(self):
        """Test that validate_module is callable."""
        assert callable(validation.scout_validate_module)

    def test_env_is_callable(self):
        """Test that env is callable."""
        assert callable(validation.scout_env)

    def test_function_info_is_callable(self):
        """Test that function_info is callable."""
        assert callable(validation.scout_function_info)


class TestValidationToolOutput:
    """Test that validation tools return ToolOutput objects."""

    @patch("subprocess.run")
    def test_lint_returns_tool_output(self, mock_run):
        """Test that lint returns a ToolOutput object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="test.py:1: warning: unused variable",
            stderr="",
        )

        result = validation.scout_lint(path="test.py")

        assert hasattr(result, "tool_name")
        assert hasattr(result, "content")
        assert hasattr(result, "cost_usd")
        assert result.tool_name == "lint"

    @patch("subprocess.run")
    def test_validate_module_returns_tool_output(self, mock_run):
        """Test that validate_module returns a ToolOutput object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="test_module.py: pass",
            stderr="",
        )

        result = validation.scout_validate_module(module_path="test_module")

        assert hasattr(result, "tool_name")
        assert result.tool_name == "validate_module"

    def test_env_returns_tool_output(self):
        """Test that env returns a ToolOutput object."""
        result = validation.scout_env()

        assert hasattr(result, "tool_name")
        assert hasattr(result, "content")
        assert result.tool_name == "env"

    @patch("subprocess.run")
    def test_function_info_returns_tool_output(self, mock_run):
        """Test that function_info returns a ToolOutput object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="def test_func():",
            stderr="",
        )

        result = validation.scout_function_info(
            module="test_module", function="test_func"
        )

        assert hasattr(result, "tool_name")
        assert result.tool_name == "function_info"
