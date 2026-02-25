"""Tests for batch processing tools."""
from __future__ import annotations

import pytest

from scout.tools import batch


class TestBatchTools:
    """Test cases for batch processing tools."""

    def test_batch_is_callable(self):
        """Test that batch is callable."""
        assert callable(batch.scout_batch)

    def test_run_is_callable(self):
        """Test that run is callable."""
        assert callable(batch.scout_run)

    def test_batch_is_async(self):
        """Test that batch is an async function."""
        import asyncio
        assert asyncio.iscoroutinefunction(batch.scout_batch)

    def test_run_is_async(self):
        """Test that run is an async function."""
        import asyncio
        assert asyncio.iscoroutinefunction(batch.scout_run)
