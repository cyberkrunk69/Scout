"""Tests for git_analyzer module."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestGitAnalyzer:
    """Test git analysis functions."""

    @patch("subprocess.run")
    def test_get_files_in_last_commit(self, mock_run):
        """Test getting files from last commit."""
        from scout.git_analyzer import get_files_in_last_commit
        
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="file1.py\nfile2.py\n"
        )
        
        files = get_files_in_last_commit(Path("/tmp"))
        assert isinstance(files, list)

    @patch("subprocess.run")
    def test_get_changed_files(self, mock_run):
        """Test getting changed files."""
        from scout.git_analyzer import get_changed_files
        
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="file1.py\nfile2.py\n"
        )
        
        files = get_changed_files("HEAD~1", "HEAD", Path("/tmp"))
        assert isinstance(files, list)

    @patch("subprocess.run")
    def test_get_current_branch(self, mock_run):
        """Test getting current branch."""
        from scout.git_analyzer import get_current_branch
        
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="main\n"
        )
        
        branch = get_current_branch(Path("/tmp"))
        assert branch == "main"

    @patch("subprocess.run")
    def test_has_remote_origin(self, mock_run):
        """Test checking for remote origin."""
        from scout.git_analyzer import has_remote_origin
        
        mock_run.return_value = MagicMock(returncode=0)
        
        result = has_remote_origin(Path("/tmp"))
        assert isinstance(result, bool)

    @patch("subprocess.run")
    def test_get_git_commit_hash(self, mock_run):
        """Test getting git commit hash."""
        from scout.git_analyzer import get_git_commit_hash
        
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123def456\n"
        )
        
        hash_val = get_git_commit_hash(Path("/tmp"))
        assert hash_val == "abc123def456"

    @patch("subprocess.run")
    def test_get_git_version(self, mock_run):
        """Test getting git version."""
        from scout.git_analyzer import get_git_version
        
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="git version 2.40.0\n"
        )
        
        version = get_git_version(Path("/tmp"))
        assert "git" in version.lower()
