"""
RefreshQueue - Budget-aware queue for documentation auto-refresh.

Manages pending regeneration tasks with:
- Debounced batch processing
- Budget enforcement before each refresh
- Idempotent operations (multiple changes -> single batch)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set

from scout.config import ScoutConfig

# Use standard logging only (structlog not compatible with keyword args in this context)
logger = logging.getLogger(__name__)


def _log_info(msg: str, **kwargs) -> None:
    """Log info message."""
    logger.info(f"{msg}: {kwargs}")


def _log_debug(msg: str, **kwargs) -> None:
    """Log debug message."""
    logger.debug(f"{msg}: {kwargs}")


def _log_warning(msg: str, **kwargs) -> None:
    """Log warning message."""
    logger.warning(f"{msg}: {kwargs}")


def _log_error(msg: str, **kwargs) -> None:
    """Log error message."""
    logger.error(f"{msg}: {kwargs}")


@dataclass
class RefreshTask:
    """Single refresh task for a symbol."""

    symbol_ref: str  # str representation of SymbolRef
    file_path: Path
    reason: str = "cascade"  # "cascade", "manual", "budget_deferred"
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RefreshQueue:
    """
    Manages pending refresh tasks with budget awareness.

    Flow:
    1. add() - Add symbols to pending queue (deduplicated)
    2. should_process() - Check if debounce period has elapsed
    3. process_pending() - Process within budget, batch multiple changes
    """

    pending: List[RefreshTask] = field(default_factory=list)
    in_progress: Set[str] = field(default_factory=set)
    completed_today: List[datetime] = field(default_factory=list)
    debounce_seconds: float = 1.0
    last_change_time: Optional[datetime] = None
    _paused: bool = False
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def add(self, symbol_ref: str, file_path: Path, reason: str = "cascade") -> None:
        """Add a symbol to the refresh queue (deduplicated by symbol_ref)."""
        # Check if already pending or in progress
        if symbol_ref in self.in_progress:
            _log_debug("refresh_skip_in_progress", symbol=symbol_ref)
            return

        # Check if already in pending (deduplicate)
        for task in self.pending:
            if task.symbol_ref == symbol_ref:
                _log_debug("refresh_skip_duplicate", symbol=symbol_ref)
                return

        task = RefreshTask(symbol_ref=symbol_ref, file_path=file_path, reason=reason)
        self.pending.append(task)
        self.last_change_time = datetime.now(timezone.utc)

        _log_info(
            "refresh_queued",
            symbol=symbol_ref,
            file=str(file_path),
            reason=reason,
            pending_count=len(self.pending),
        )

    def should_process(self) -> bool:
        """Check if debounce period has elapsed."""
        if self._paused:
            return False
        if not self.last_change_time:
            return False
        if not self.pending:
            return False

        elapsed = (datetime.now(timezone.utc) - self.last_change_time).total_seconds()
        return elapsed >= self.debounce_seconds

    def get_pending_count(self) -> int:
        """Return number of pending refresh tasks."""
        return len(self.pending)

    def get_in_progress_count(self) -> int:
        """Return number of in-progress refresh tasks."""
        return len(self.in_progress)

    def is_paused(self) -> bool:
        """Return whether the queue is paused."""
        return self._paused

    def pause(self) -> None:
        """Pause auto-refresh processing."""
        self._paused = True
        _log_info("refresh_queue_paused")

    def resume(self) -> None:
        """Resume auto-refresh processing."""
        self._paused = False
        _log_info("refresh_queue_resumed")

    def clear_pending(self) -> int:
        """Clear all pending tasks. Returns count of cleared tasks."""
        count = len(self.pending)
        self.pending.clear()
        _log_info("refresh_queue_cleared", count=count)
        return count

    async def process_pending(
        self,
        config: ScoutConfig,
        repo_root: Path,
        regenerate_func=None,
    ) -> int:
        """
        Process all pending refreshes within budget.

        Args:
            config: ScoutConfig for budget checking
            repo_root: Repository root path
            regenerate_func: Optional async function(symbol_ref, repo_root) -> bool

        Returns:
            Number of symbols processed
        """
        from scout.router import check_budget_with_message

        if not self.pending:
            return 0

        processed = 0

        async with self._lock:
            while self.pending:
                # Check budget before each refresh
                # Small cost estimate per symbol refresh
                estimated_cost = 0.01

                if not check_budget_with_message(config, estimated_cost=estimated_cost):
                    _log_info(
                        "refresh_deferred_budget",
                        pending_count=len(self.pending),
                        estimated_cost=estimated_cost,
                    )
                    break

                task = self.pending.pop(0)
                self.in_progress.add(task.symbol_ref)

                try:
                    if regenerate_func:
                        success = await regenerate_func(task.symbol_ref, repo_root)
                        if success:
                            processed += 1
                            self.completed_today.append(datetime.now(timezone.utc))
                            _log_info(
                                "refresh_completed",
                                symbol=task.symbol_ref,
                                reason=task.reason,
                            )
                        else:
                            _log_warning(
                                "refresh_failed",
                                symbol=task.symbol_ref,
                            )
                    else:
                        # Default: just log (actual regeneration handled elsewhere)
                        processed += 1
                        self.completed_today.append(datetime.now(timezone.utc))
                        _log_info(
                            "refresh_processed_default",
                            symbol=task.symbol_ref,
                        )

                except Exception as e:
                    _log_error(
                        "refresh_error",
                        symbol=task.symbol_ref,
                        error=str(e),
                    )
                finally:
                    self.in_progress.discard(task.symbol_ref)

        return processed

    def get_status(self) -> dict:
        """Get queue status for MCP tools."""
        return {
            "pending": len(self.pending),
            "in_progress": len(self.in_progress),
            "paused": self._paused,
            "debounce_seconds": self.debounce_seconds,
            "last_change": self.last_change_time.isoformat() if self.last_change_time else None,
            "sample_pending": [
                {"symbol": t.symbol_ref, "reason": t.reason}
                for t in self.pending[:5]
            ],
        }


# Global refresh queue instance
_refresh_queue: Optional[RefreshQueue] = None
_queue_lock = asyncio.Lock()


async def get_refresh_queue() -> RefreshQueue:
    """Get or create the global refresh queue."""
    global _refresh_queue
    async with _queue_lock:
        if _refresh_queue is None:
            _refresh_queue = RefreshQueue()
        return _refresh_queue


def get_refresh_queue_sync() -> RefreshQueue:
    """Get the global refresh queue synchronously (for non-async contexts)."""
    global _refresh_queue
    if _refresh_queue is None:
        _refresh_queue = RefreshQueue()
    return _refresh_queue


async def queue_regeneration(
    symbol_refs: List[str],
    file_paths: List[Path],
    reason: str = "cascade",
) -> None:
    """
    Queue symbols for regeneration.

    Args:
        symbol_refs: List of symbol references (str form)
        file_paths: Corresponding file paths
        reason: Reason for refresh (cascade, manual)
    """
    queue = await get_refresh_queue()

    for ref, path in zip(symbol_refs, file_paths):
        queue.add(ref, path, reason)


# Background task runner for continuous processing
async def run_refresh_processor(
    config: ScoutConfig,
    repo_root: Path,
    regenerate_func=None,
    poll_interval: float = 0.5,
    shutdown_event: Optional[asyncio.Event] = None,
) -> None:
    """
    Background task that continuously processes the refresh queue.

    Args:
        config: ScoutConfig for budget checking
        repo_root: Repository root path
        regenerate_func: Optional async function(symbol_ref, repo_root) -> bool
        poll_interval: How often to check the queue (seconds)
        shutdown_event: Optional event to signal shutdown
    """
    shutdown_event = shutdown_event or asyncio.Event()

    _log_info("refresh_processor_started", poll_interval=poll_interval)

    while not shutdown_event.is_set():
        try:
            queue = await get_refresh_queue()

            if queue.should_process():
                processed = await queue.process_pending(
                    config=config,
                    repo_root=repo_root,
                    regenerate_func=regenerate_func,
                )

                if processed > 0:
                    _log_info("refresh_batch_completed", count=processed)

            # Wait before next poll
            await asyncio.sleep(poll_interval)

        except asyncio.CancelledError:
            _log_info("refresh_processor_cancelled")
            break
        except Exception as e:
            _log_error("refresh_processor_error", error=str(e))
            await asyncio.sleep(1)  # Back off on error

    _log_info("refresh_processor_stopped")
