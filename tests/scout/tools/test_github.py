"""Tests for GitHub tools."""
from __future__ import annotations

import pytest

from scout.tools import github


class TestGitHubTools:
    """Test cases for GitHub API tools."""

    def test_pr_info_is_callable(self):
        """Test that pr_info is callable."""
        assert callable(github.scout_pr_info)

    def test_pr_is_callable(self):
        """Test that pr is callable."""
        assert callable(github.scout_pr)

    def test_pr_is_async(self):
        """Test that pr is an async function."""
        import asyncio
        assert asyncio.iscoroutinefunction(github.scout_pr)
