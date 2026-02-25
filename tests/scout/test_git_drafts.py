"""Tests for git_drafts.py module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scout.git_drafts import (
    _stem_for_file,
    _find_package_root,
    _read_module_summary,
    assemble_pr_description,
    assemble_pr_description_from_docs,
    assemble_commit_message,
    _call_graph_summary_for_scope,
)


class TestStemForFile:
    """Tests for _stem_for_file function."""

    def test_basic_stem(self):
        """Test basic stem extraction."""
        root = Path("/repo")
        file_path = Path("/repo/src/module.py")

        result = _stem_for_file(file_path, root)

        assert result == "module"

    def test_nested_path(self):
        """Test with nested path."""
        root = Path("/repo")
        file_path = Path("/repo/src/scout/auth.py")

        result = _stem_for_file(file_path, root)

        assert result == "auth"

    def test_file_outside_root(self):
        """Test with file outside root."""
        root = Path("/repo")
        file_path = Path("/other/path.py")

        result = _stem_for_file(file_path, root)

        # Should return stem of the file itself
        assert result == "path"


class TestAssemblePRDescription:
    """Tests for assemble_pr_description function."""

    def test_no_staged_files(self):
        """Test with no staged files."""
        result = assemble_pr_description(Path("/repo"), [])

        assert "No staged doc files" in result

    def test_non_doc_files(self):
        """Test with non-documentation files."""
        staged = [Path("/repo/README.md"), Path("/repo/config.yaml")]

        result = assemble_pr_description(Path("/repo"), staged)

        assert "No staged doc files" in result


class TestAssembleCommitMessage:
    """Tests for assemble_commit_message function."""

    def test_no_staged_files(self):
        """Test with no staged files."""
        result = assemble_commit_message(Path("/repo"), [])

        assert "No staged doc files" in result

    def test_non_doc_files(self):
        """Test with non-documentation files."""
        staged = [Path("/repo/README.md")]

        result = assemble_commit_message(Path("/repo"), staged)

        assert "No staged doc files" in result


class TestCallGraphSummaryForScope:
    """Tests for _call_graph_summary_for_scope function."""

    @patch("pathlib.Path.exists")
    def test_no_call_graph_file(self, mock_exists):
        """Test when call graph file doesn't exist."""
        mock_exists.return_value = False

        result = _call_graph_summary_for_scope(
            Path("/repo"),
            "src",
            Path("/repo/call_graph.json"),
        )

        assert result == ""

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    def test_invalid_json(self, mock_read, mock_exists):
        """Test with invalid JSON."""
        mock_exists.return_value = True
        mock_read.return_value = "not valid json"

        result = _call_graph_summary_for_scope(
            Path("/repo"),
            "src",
            Path("/repo/call_graph.json"),
        )

        assert result == ""

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.read_text")
    def test_with_valid_call_graph(self, mock_read, mock_exists):
        """Test with valid call graph JSON."""
        mock_exists.return_value = True
        mock_read.return_value = """{
            "edges": [
                {"from": "src/a.py::func_a", "to": "src/b.py::func_b"},
                {"from": "src/a.py::func_a", "to": "src/c.py::func_c"}
            ]
        }"""

        result = _call_graph_summary_for_scope(
            Path("/repo"),
            "src",
            Path("/repo/call_graph.json"),
        )

        assert "Call graph" in result
        assert "src/a.py" in result
