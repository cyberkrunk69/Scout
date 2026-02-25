"""
Cache Dependency Tracking for Auto-Stale Awareness

Provides file watching and dependency tracking for automatic cache invalidation
when source files change. This reduces token waste from stale cached data.

Key components:
- FileWatcher: Uses watchdog to monitor file changes
- dependency_map: Maps file paths to cache keys that depend on them
- register_dependency: Register cache keys with their dependency files
- invalidate_for_path: Invalidate cache entries when a file changes
"""

from __future__ import annotations

import asyncio
import glob as glob_module
import os
import threading
from pathlib import Path
from typing import Any, Callable

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:
    import logging

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)


def _log_info(msg: str, **kwargs) -> None:
    """Log info message, handling both structlog and standard logging."""
    if hasattr(logger, "bind"):
        logger.info(msg, **kwargs)
    else:
        logger.info(f"{msg}: {kwargs}")


def _log_debug(msg: str, **kwargs) -> None:
    """Log debug message, handling both structlog and standard logging."""
    if hasattr(logger, "bind"):
        logger.debug(msg, **kwargs)
    else:
        logger.debug(f"{msg}: {kwargs}")


def _log_warning(msg: str, **kwargs) -> None:
    """Log warning message, handling both structlog and standard logging."""
    if hasattr(logger, "bind"):
        logger.warning(msg, **kwargs)
    else:
        logger.warning(f"{msg}: {kwargs}")


def resolve_to_repo_relative(abs_path: str, repo_root: str) -> Path:
    """
    Convert absolute path to repo-relative Path for SymbolRef.

    Args:
        abs_path: Absolute file path
        repo_root: Repository root directory

    Returns:
        Path object relative to repo_root

    Raises:
        ValueError: If abs_path is not under repo_root
    """
    if not abs_path or not repo_root:
        raise ValueError("abs_path and repo_root must be non-empty")

    abs_path = os.path.realpath(abs_path)
    repo_root = os.path.realpath(repo_root)

    if not abs_path.startswith(repo_root + os.sep) and abs_path != repo_root:
        raise ValueError(f"Path {abs_path} is outside repo root {repo_root}")

    rel = os.path.relpath(abs_path, repo_root)
    return Path(rel)


async def on_file_changed(
    changed_file_path: str,
    repo_root: Path,
    refresh_queue=None,
) -> None:
    """
    Bridge file watcher to dependency graph - auto-refresh trigger.

    Flow:
    1. Normalize path (absolute to repo-relative)
    2. Call invalidate_for_path() for direct cache invalidation
    3. Call invalidate_cascade() on dependency graph for transitive invalidation
    4. Queue affected symbols for regeneration
    """
    from scout.deps import DependencyGraph, SymbolRef

    if not changed_file_path:
        _log_warning("empty_file_path")
        return

    abs_fp = os.path.abspath(changed_file_path)

    # 1. Direct cache invalidation
    invalidate_for_path(abs_fp)

    # 2. Convert to repo-relative for dependency graph
    try:
        rel_path = resolve_to_repo_relative(abs_fp, str(repo_root))
    except ValueError as e:
        _log_warning("file_outside_repo", path=abs_fp, error=str(e))
        return

    if rel_path.suffix != ".py":
        _log_debug("skip_non_python", path=str(rel_path))
        return

    # 3. Cascade invalidation via dependency graph
    graph = DependencyGraph(repo_root)
    affected = graph.invalidate_cascade(changed_files=[rel_path])

    if affected:
        _log_info(
            "cascade_invalidation",
            changed_file=str(rel_path),
            affected_count=len(affected),
        )

        # 4. Queue for regeneration
        queue = refresh_queue
        if queue is None:
            from scout.refresh_queue import get_refresh_queue
            queue = await get_refresh_queue()

        for ref in affected:
            file_path = repo_root / ref.path
            queue.add(symbol_ref=str(ref), file_path=file_path, reason="cascade")

        _log_info(
            "queued_for_regeneration",
            count=len(affected),
            pending_count=queue.get_pending_count(),
        )
    else:
        _log_debug("no_affected_symbols", file=str(rel_path))


# Global dependency map: file_path -> set of (wrapper, cache_key) tuples
_dependency_map: dict[str, set[tuple[Any, str]]] = {}
_map_lock = threading.RLock()

# Reference to the running event loop for async cache operations
_event_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Set the event loop reference for async cache invalidation."""
    global _event_loop
    _event_loop = loop


def register_dependency(cache_id: tuple[Any, str], file_paths: list[str]) -> None:
    """
    Register a cache entry's dependency on specific files.

    Args:
        cache_id: Tuple of (wrapper_function, cache_key)
        file_paths: List of absolute file paths this cache entry depends on
    """
    with _map_lock:
        for fp in file_paths:
            abs_fp = os.path.abspath(fp)
            # Check if already registered for this file - idempotency check
            existing = _dependency_map.get(abs_fp, set())
            if cache_id in existing:
                _log_debug(
                    "dependency_already_registered",
                    file=abs_fp,
                    key=cache_id[1][:8] if cache_id[1] else None,
                    func=(
                        cache_id[0].__name__
                        if hasattr(cache_id[0], "__name__")
                        else None
                    ),
                )
                continue
            _dependency_map.setdefault(abs_fp, set()).add(cache_id)
            _log_debug(
                "dependency_registered",
                file=abs_fp,
                key=cache_id[1][:8] if cache_id[1] else None,
                func=cache_id[0].__name__ if hasattr(cache_id[0], "__name__") else None,
            )


async def _do_invalidate(file_path: str) -> None:
    """
    Async function to perform cache invalidation.
    Must be called from the async event loop.
    """
    abs_fp = os.path.abspath(file_path)

    with _map_lock:
        items = _dependency_map.pop(abs_fp, set())

    if not items:
        return

    invalidated_count = 0
    for wrapper, key in items:
        if hasattr(wrapper, "_cache") and wrapper._cache:
            removed = wrapper._cache.pop(key, None)
            if removed is not None:
                invalidated_count += 1
                _log_debug(
                    "cache_invalidated",
                    file=abs_fp,
                    key=key[:8] if key else None,
                    func=wrapper.__name__ if hasattr(wrapper, "__name__") else None,
                )

    if invalidated_count > 0:
        _log_info(
            "cache_invalidation_complete",
            file=abs_fp,
            invalidated_count=invalidated_count,
        )


def invalidate_for_path(file_path: str) -> None:
    """
    Invalidate all cache entries that depend on the given file.

    This function is called from the file watcher thread. It schedules
    the actual invalidation on the async event loop to ensure thread safety.
    """
    abs_fp = os.path.abspath(file_path)

    # Check if we have any dependencies for this file
    with _map_lock:
        has_deps = abs_fp in _dependency_map and len(_dependency_map[abs_fp]) > 0

    if not has_deps:
        return

    # Schedule the async invalidation on the event loop
    if _event_loop is not None and not _event_loop.is_closed():
        asyncio.run_coroutine_threadsafe(_do_invalidate(abs_fp), _event_loop)
        _log_debug("invalidation_scheduled", file=abs_fp)
    else:
        # Fallback: try to run synchronously (may cause issues with async locks)
        _log_warning(
            "event_loop_unavailable_falling_back",
            file=abs_fp,
            has_loop=_event_loop is not None,
        )
        try:
            # Try to get the running loop if we're in async context
            loop = asyncio.get_running_loop()
            asyncio.ensure_future(_do_invalidate(abs_fp), loop=loop)
        except RuntimeError:
            # No running loop - skip invalidation for now
            _log_warning("no_event_loop_skip_invalidation", file=abs_fp)


class ChangeHandler(PatternMatchingEventHandler):
    """Handler for watchdog file system events with directory exclusion and
    debouncing."""

    # Patterns to ignore: .git, __pycache__, .scout index DB, and other cache dirs
    IGNORE_PATTERNS = [
        "*/.git/*",
        "*/.git/**",
        "*/__pycache__/*",
        "*/__pycache__/**",
        "*/.scout/*",
        "*/.scout/**",
    ]

    # Debounce delay in seconds - wait after last event before processing
    DEBOUNCE_DELAY = 0.1  # 100ms

    def __init__(self, callback: Callable[[str], None]):
        super().__init__(ignore_patterns=self.IGNORE_PATTERNS, ignore_directories=True)
        self.callback = callback
        # Debounce state: file_path -> timer
        self._pending_timers: dict[str, threading.Timer] = {}
        self._timer_lock = threading.Lock()

    def _schedule_debounced(self, file_path: str, event_type: str) -> None:
        """Schedule a debounced callback for the given file path."""
        with self._timer_lock:
            # Cancel any existing timer for this file
            existing_timer = self._pending_timers.get(file_path)
            if existing_timer is not None:
                existing_timer.cancel()

            # Create new debounced timer
            timer = threading.Timer(
                self.DEBOUNCE_DELAY,
                self._execute_callback,
                args=(file_path, event_type),
            )
            self._pending_timers[file_path] = timer
            timer.start()

    def _execute_callback(self, file_path: str, event_type: str) -> None:
        """Execute the callback after debounce delay."""
        with self._timer_lock:
            # Clean up timer reference
            self._pending_timers.pop(file_path, None)

        _log_debug("debounce_complete", path=file_path, event_type=event_type)
        self.callback(file_path)

    def on_modified(self, event) -> None:
        """Handle file modification events with debouncing to prevent event storms."""
        if not event.is_directory:
            _log_debug("file_modified_raw", path=event.src_path)
            self._schedule_debounced(event.src_path, "modified")

    def on_created(self, event) -> None:
        """Handle file creation events with debouncing to prevent event storms."""
        if not event.is_directory:
            _log_debug("file_created_raw", path=event.src_path)
            # Use debounce to coalesce rapid create events (e.g., git checkout)
            self._schedule_debounced(event.src_path, "created")


class FileWatcher:
    """
    File system watcher using watchdog.

    Watches specified paths for file changes and triggers callbacks
    when files are modified.
    """

    def __init__(
        self,
        paths_to_watch: list[str],
        callback: Callable[[str], None] | None = None,
        recursive: bool = True,
    ):
        """
        Initialize the file watcher.

        Args:
            paths_to_watch: List of directory paths to monitor
            callback: Function to call when files change
            recursive: Whether to watch subdirectories
        """
        self.observer = Observer()
        self.handler = ChangeHandler(callback or invalidate_for_path)
        self.paths = paths_to_watch
        self.recursive = recursive
        self._started = False

    def start(self) -> None:
        """Start watching for file changes."""
        if self._started:
            return

        for path in self.paths:
            if os.path.isdir(path):
                self.observer.schedule(self.handler, path, recursive=self.recursive)
                _log_info("watching_directory", path=path, recursive=self.recursive)
            else:
                _log_warning("path_not_directory_skipping", path=path)

        self.observer.start()
        self._started = True
        _log_info("file_watcher_started", paths=self.paths)

    def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._started:
            return

        self.observer.stop()
        self.observer.join(timeout=5)
        self._started = False
        _log_info("file_watcher_stopped")

    def __enter__(self) -> "FileWatcher":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()


def resolve_dependencies(
    patterns: list[str],
    base_dir: str | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[str]:
    """
    Resolve glob patterns to absolute file paths.

    Args:
        patterns: List of glob patterns (e.g., ["**/*.py", "src/*.js"])
        base_dir: Base directory for relative patterns (defaults to cwd)
        exclude_patterns: List of substrings to exclude from results
                         (e.g., ["tests/", "test_"] to skip test files)

    Returns:
        List of absolute file paths matching the patterns
    """
    if base_dir is None:
        base_dir = os.getcwd()

    # Default exclude patterns for production use
    if exclude_patterns is None:
        exclude_patterns = []

    resolved = []
    for pattern in patterns:
        # Handle relative patterns
        if not os.path.isabs(pattern):
            pattern = os.path.join(base_dir, pattern)

        # Glob the pattern
        matches = glob_module.glob(pattern, recursive=True)
        resolved.extend(matches)

    # Deduplicate while preserving order, and filter excluded paths
    seen = set()
    unique_resolved = []
    for fp in resolved:
        abs_fp = os.path.abspath(fp)
        if abs_fp in seen:
            continue

        # Filter out excluded patterns
        should_exclude = False
        for exclude in exclude_patterns:
            if exclude in abs_fp:
                should_exclude = True
                break
        if should_exclude:
            continue

        seen.add(abs_fp)
        unique_resolved.append(abs_fp)

    return unique_resolved


# Global watcher instance
_watcher: FileWatcher | None = None
_watcher_lock = threading.Lock()


def get_watcher() -> FileWatcher:
    """Get or create the global file watcher."""
    global _watcher
    with _watcher_lock:
        if _watcher is None:
            _watcher = FileWatcher(
                paths_to_watch=[os.getcwd()],
                callback=invalidate_for_path,
                recursive=True,
            )
        return _watcher


def ensure_watcher_started() -> None:
    """Ensure the global file watcher is started."""
    watcher = get_watcher()
    if not watcher._started:
        watcher.start()
