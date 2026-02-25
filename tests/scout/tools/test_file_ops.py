"""Tests for file operations tools."""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scout.tools import file_ops


class TestFileOpsTools:
    """Test cases for file operation tools."""

    def test_read_file_is_callable(self):
        """Test that read_file is callable."""
        assert callable(file_ops.scout_read_file)

    def test_write_with_review_is_callable(self):
        """Test that write_with_review is callable."""
        assert callable(file_ops.scout_write_with_review)

    def test_delete_with_review_is_callable(self):
        """Test that delete_with_review is callable."""
        assert callable(file_ops.scout_delete_with_review)

    def test_edit_is_callable(self):
        """Test that edit is callable."""
        assert callable(file_ops.scout_edit)

    def test_write_with_review_is_async(self):
        """Test that write_with_review is an async function."""
        assert asyncio.iscoroutinefunction(file_ops.scout_write_with_review)

    def test_delete_with_review_is_async(self):
        """Test that delete_with_review is an async function."""
        assert asyncio.iscoroutinefunction(file_ops.scout_delete_with_review)

    def test_edit_is_async(self):
        """Test that edit is an async function."""
        assert asyncio.iscoroutinefunction(file_ops.scout_edit)


class TestFileOpsToolOutput:
    """Test that file_ops tools return ToolOutput objects."""

    def test_read_file_returns_tool_output(self):
        """Test that read_file returns a ToolOutput object."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("test")
            temp_path = f.name

        try:
            result = file_ops.scout_read_file(file_path=temp_path)

            assert hasattr(result, "tool_name")
            assert hasattr(result, "content")
            assert hasattr(result, "cost_usd")
            assert result.tool_name == "read_file"
        finally:
            os.unlink(temp_path)
