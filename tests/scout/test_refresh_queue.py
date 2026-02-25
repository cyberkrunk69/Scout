"""Tests for refresh_queue.py module."""

import pytest
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from scout.refresh_queue import (
    RefreshTask,
    RefreshQueue,
    get_refresh_queue,
    get_refresh_queue_sync,
    queue_regeneration,
    run_refresh_processor,
)


class TestRefreshTask:
    """Tests for RefreshTask dataclass."""

    def test_creation(self):
        """Test creating a RefreshTask."""
        task = RefreshTask(
            symbol_ref="module::function",
            file_path=Path("/repo/src/module.py"),
            reason="cascade",
        )

        assert task.symbol_ref == "module::function"
        assert task.file_path == Path("/repo/src/module.py")
        assert task.reason == "cascade"
        assert task.queued_at is not None


class TestRefreshQueue:
    """Tests for RefreshQueue class."""

    def test_initialization(self):
        """Test RefreshQueue initialization."""
        queue = RefreshQueue()

        assert queue.pending == []
        assert queue.in_progress == set()
        assert queue.completed_today == []
        assert queue.debounce_seconds == 1.0
        assert queue.last_change_time is None
        assert queue._paused is False

    def test_add_task(self):
        """Test adding a task to the queue."""
        queue = RefreshQueue()

        queue.add(
            symbol_ref="module::func",
            file_path=Path("/repo/src/module.py"),
            reason="cascade",
        )

        assert len(queue.pending) == 1
        assert queue.pending[0].symbol_ref == "module::func"
        assert queue.last_change_time is not None

    def test_add_duplicate(self):
        """Test that duplicate tasks are ignored."""
        queue = RefreshQueue()

        queue.add("module::func", Path("/repo/src/module.py"))
        queue.add("module::func", Path("/repo/src/module.py"))

        assert len(queue.pending) == 1

    def test_add_already_in_progress(self):
        """Test that tasks already in progress are ignored."""
        queue = RefreshQueue()
        queue.in_progress.add("module::func")

        queue.add("module::func", Path("/repo/src/module.py"))

        assert len(queue.pending) == 0

    def test_should_process_not_paused(self):
        """Test should_process when not paused and debounce elapsed."""
        queue = RefreshQueue()
        queue.pending = [RefreshTask("a", Path("a.py"))]
        queue.last_change_time = datetime.now(timezone.utc)

        # Time hasn't elapsed yet
        assert queue.should_process() is False

    def test_should_process_paused(self):
        """Test should_process when paused."""
        queue = RefreshQueue()
        queue._paused = True
        queue.pending = [RefreshTask("a", Path("a.py"))]
        queue.last_change_time = datetime.now(timezone.utc)

        assert queue.should_process() is False

    def test_should_process_no_pending(self):
        """Test should_process with no pending tasks."""
        queue = RefreshQueue()
        queue.last_change_time = datetime.now(timezone.utc)

        assert queue.should_process() is False

    def test_should_process_no_last_change(self):
        """Test should_process with no last change time."""
        queue = RefreshQueue()
        queue.pending = [RefreshTask("a", Path("a.py"))]

        assert queue.should_process() is False

    def test_get_pending_count(self):
        """Test getting pending count."""
        queue = RefreshQueue()
        queue.pending = [
            RefreshTask("a", Path("a.py")),
            RefreshTask("b", Path("b.py")),
        ]

        assert queue.get_pending_count() == 2

    def test_get_in_progress_count(self):
        """Test getting in-progress count."""
        queue = RefreshQueue()
        queue.in_progress = {"a", "b", "c"}

        assert queue.get_in_progress_count() == 3

    def test_is_paused(self):
        """Test is_paused method."""
        queue = RefreshQueue()
        assert queue.is_paused() is False

        queue._paused = True
        assert queue.is_paused() is True

    def test_pause_resume(self):
        """Test pause and resume."""
        queue = RefreshQueue()

        queue.pause()
        assert queue._paused is True

        queue.resume()
        assert queue._paused is False

    def test_clear_pending(self):
        """Test clearing pending tasks."""
        queue = RefreshQueue()
        queue.pending = [
            RefreshTask("a", Path("a.py")),
            RefreshTask("b", Path("b.py")),
        ]

        count = queue.clear_pending()

        assert count == 2
        assert len(queue.pending) == 0

    @pytest.mark.asyncio
    async def test_process_pending_empty(self):
        """Test processing with no pending tasks."""
        queue = RefreshQueue()
        config = MagicMock()

        result = await queue.process_pending(config, Path("/repo"))

        assert result == 0

    @pytest.mark.asyncio
    async def test_process_pending_default(self):
        """Test processing with default regenerate function."""
        queue = RefreshQueue()
        queue.pending = [RefreshTask("module::func", Path("module.py"))]
        queue.last_change_time = datetime.now(timezone.utc)

        config = MagicMock()

        # Patch at the import location inside the function
        with patch("scout.router.check_budget_with_message", return_value=True):
            result = await queue.process_pending(config, Path("/repo"))

        assert result == 1
        assert len(queue.pending) == 0
        assert len(queue.completed_today) == 1

    @pytest.mark.asyncio
    async def test_process_pending_budget_deferred(self):
        """Test processing when budget is exceeded."""
        queue = RefreshQueue()
        queue.pending = [RefreshTask("module::func", Path("module.py"))]
        queue.last_change_time = datetime.now(timezone.utc)

        config = MagicMock()

        with patch("scout.router.check_budget_with_message", return_value=False):
            result = await queue.process_pending(config, Path("/repo"))

        assert result == 0
        # Task should still be pending
        assert len(queue.pending) == 1

    @pytest.mark.asyncio
    async def test_process_pending_with_custom_function(self):
        """Test processing with custom regenerate function."""
        queue = RefreshQueue()
        queue.pending = [RefreshTask("module::func", Path("module.py"))]
        queue.last_change_time = datetime.now(timezone.utc)

        config = MagicMock()
        mock_func = AsyncMock(return_value=True)

        with patch("scout.router.check_budget_with_message", return_value=True):
            result = await queue.process_pending(
                config, Path("/repo"), regenerate_func=mock_func
            )

        assert result == 1
        mock_func.assert_called_once()

    def test_get_status(self):
        """Test getting queue status."""
        queue = RefreshQueue()
        queue.pending = [
            RefreshTask("module::func1", Path("module1.py")),
            RefreshTask("module::func2", Path("module2.py")),
        ]
        queue.in_progress = {"module::func3"}
        queue._paused = True
        queue.last_change_time = datetime.now(timezone.utc)

        status = queue.get_status()

        assert status["pending"] == 2
        assert status["in_progress"] == 1
        assert status["paused"] is True
        assert status["debounce_seconds"] == 1.0
        assert status["last_change"] is not None


class TestGetRefreshQueue:
    """Tests for get_refresh_queue functions."""

    @pytest.mark.asyncio
    async def test_get_refresh_queue(self):
        """Test async get_refresh_queue."""
        # Reset global
        import scout.refresh_queue as rq_module
        rq_module._refresh_queue = None
        rq_module._queue_lock = asyncio.Lock()

        queue = await get_refresh_queue()

        assert isinstance(queue, RefreshQueue)

    def test_get_refresh_queue_sync(self):
        """Test sync get_refresh_queue."""
        # Reset global
        import scout.refresh_queue as rq_module
        rq_module._refresh_queue = None

        queue = get_refresh_queue_sync()

        assert isinstance(queue, RefreshQueue)


class TestQueueRegeneration:
    """Tests for queue_regeneration function."""

    @pytest.mark.asyncio
    async def test_queue_regeneration(self):
        """Test queue_regeneration."""
        # Reset global
        import scout.refresh_queue as rq_module
        rq_module._refresh_queue = None
        rq_module._queue_lock = asyncio.Lock()

        await queue_regeneration(
            symbol_refs=["module::func1", "module::func2"],
            file_paths=[Path("module1.py"), Path("module2.py")],
        )

        queue = await get_refresh_queue()
        assert queue.get_pending_count() == 2


class TestRunRefreshProcessor:
    """Tests for run_refresh_processor function."""

    @pytest.mark.asyncio
    async def test_processor_stops_on_shutdown(self):
        """Test that processor stops on shutdown event."""
        config = MagicMock()
        shutdown_event = asyncio.Event()

        # Set shutdown immediately
        shutdown_event.set()

        await run_refresh_processor(
            config=config,
            repo_root=Path("/repo"),
            shutdown_event=shutdown_event,
        )

        # Should complete without error
