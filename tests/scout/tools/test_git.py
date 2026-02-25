"""Tests for git tools."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from scout.tools import git
from scout.tool_output import ToolOutput


class TestGitTools:
    """Test cases for git operation tools."""

    def test_git_status_is_callable(self):
        """Test that git_status is callable."""
        assert callable(git.scout_git_status)

    def test_git_diff_is_callable(self):
        """Test that git_diff is callable."""
        assert callable(git.scout_git_diff)

    def test_git_log_is_callable(self):
        """Test that git_log is callable."""
        assert callable(git.scout_git_log)

    def test_git_branch_is_callable(self):
        """Test that git_branch is callable."""
        assert callable(git.scout_git_branch)

    def test_git_show_is_callable(self):
        """Test that git_show is callable."""
        assert callable(git.scout_git_show)

    def test_git_add_is_callable(self):
        """Test that git_add is callable."""
        assert callable(git.scout_git_add)

    def test_git_commit_is_callable(self):
        """Test that git_commit is callable."""
        assert callable(git.scout_git_commit)


class TestGitToolOutput:
    """Test that git tools return ToolOutput objects."""

    @patch("subprocess.run")
    def test_git_status_returns_tool_output(self, mock_run):
        """Test that git_status returns a ToolOutput object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="M  modified.txt",
            stderr="",
        )

        result = git.scout_git_status()

        assert hasattr(result, "tool_name")
        assert hasattr(result, "content")
        assert hasattr(result, "cost_usd")
        assert result.tool_name == "git_status"

    @patch("subprocess.run")
    def test_git_diff_returns_tool_output(self, mock_run):
        """Test that git_diff returns a ToolOutput object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="diff --git a/test.py b/test.py\n+new line",
            stderr="",
        )

        result = git.scout_git_diff()

        assert hasattr(result, "tool_name")
        assert result.tool_name == "git_diff"

    @patch("subprocess.run")
    def test_git_log_returns_tool_output(self, mock_run):
        """Test that git_log returns a ToolOutput object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="commit abc123",
            stderr="",
        )

        result = git.scout_git_log()

        assert hasattr(result, "tool_name")
        assert result.tool_name == "git_log"

    @patch("subprocess.run")
    def test_git_branch_returns_tool_output(self, mock_run):
        """Test that git_branch returns a ToolOutput object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="* main",
            stderr="",
        )

        result = git.scout_git_branch()

        assert hasattr(result, "tool_name")
        assert result.tool_name == "git_branch"

    @patch("subprocess.run")
    def test_git_show_returns_tool_output(self, mock_run):
        """Test that git_show returns a ToolOutput object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="commit abc123",
            stderr="",
        )

        result = git.scout_git_show()

        assert hasattr(result, "tool_name")
        assert result.tool_name == "git_show"

    @patch("subprocess.run")
    def test_git_add_returns_tool_output(self, mock_run):
        """Test that git_add returns a ToolOutput object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr="",
        )

        result = git.scout_git_add(paths=["test.py"])

        assert hasattr(result, "tool_name")
        assert result.tool_name == "git_add"

    @patch("subprocess.run")
    def test_git_commit_returns_tool_output(self, mock_run):
        """Test that git_commit returns a ToolOutput object."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="[main abc123] Test commit",
            stderr="",
        )

        result = git.scout_git_commit(message="Test commit")

        assert hasattr(result, "tool_name")
        assert result.tool_name == "git_commit"
