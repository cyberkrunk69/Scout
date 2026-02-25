"""Shared utilities for DataClaw parsers.

Provides reusable utilities for:
- Path normalization (home → ~, project root → relative)
- Temp file copy pattern for SQLite databases (avoid DB locks)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


def normalize_path(path: Path, project_root: Optional[Path] = None) -> str:
    """Normalize a path for consistent storage.

    Transforms:
    - Absolute paths under home directory → ~/relative
    - Paths under project_root → ./relative (if provided)

    Args:
        path: The path to normalize
        project_root: Optional project root for relative path conversion

    Returns:
        Normalized path string
    """
    path = path.expanduser().resolve()

    # Try to make relative to project root first
    if project_root:
        try:
            rel = path.relative_to(project_root.resolve())
            return f"./{rel}"
        except ValueError:
            pass

    # Try to make relative to home directory
    home = Path.home()
    try:
        rel = path.relative_to(home)
        return f"~/{rel}"
    except ValueError:
        pass

    # Return as-is if neither applies
    return str(path)


def find_project_root(start_path: Path) -> Optional[Path]:
    """Find the project root by looking for common markers.

    Args:
        start_path: Path to start searching from

    Returns:
        Project root path if found, None otherwise
    """
    markers = [".git", "pyproject.toml", "package.json", "Cargo.toml"]
    current = start_path.resolve()

    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent

    return None


@contextmanager
def temp_copy(src_path: Path, suffix: str = "") -> Iterator[Path]:
    """Create a temporary copy of a file (useful for SQLite databases).

    This pattern avoids database locks when the original is open by another
    process. The temporary file is cleaned up automatically when the context
    exits.

    Args:
        src_path: Path to file to copy
        suffix: Optional suffix for temp filename

    Yields:
        Path to the temporary copy

    Raises:
        FileNotFoundError: If source file doesn't exist
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")

    # Create temp file in same directory as source to avoid cross-device issues
    # with SQLite
    temp_dir = src_path.parent.resolve()
    fd, temp_path = tempfile.mkstemp(
        dir=temp_dir,
        prefix=f".dataclaw_tmp_{src_path.stem}_",
        suffix=suffix
    )
    os.close(fd)

    try:
        shutil.copy2(src_path, temp_path)
        yield Path(temp_path)
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except OSError as e:
            logger.warning("Failed to clean up temp file %s: %s", temp_path, e)


def parse_kv_store(
    conn: "sqlite3.Connection",
    key_prefix: str
) -> Dict[str, Any]:
    """Parse key-value store from SQLite cursorDiskKV table.

    Args:
        conn: SQLite connection
        key_prefix: Prefix to filter keys (e.g., "composerData", "bubbleId")

    Returns:
        Dictionary of key -> parsed JSON value
    """
    import json

    cursor = conn.execute(
        "SELECT key, value FROM cursorDiskKV WHERE key LIKE ?",
        (f"{key_prefix}%",)
    )

    result = {}
    for key, value in cursor:
        try:
            result[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            result[key] = value

    return result


def get_all_kv_keys(conn: "sqlite3.Connection") -> list[str]:
    """Get all keys from cursorDiskKV table.

    Args:
        conn: SQLite connection

    Returns:
        List of all keys
    """
    cursor = conn.execute("SELECT key FROM cursorDiskKV")
    return [row[0] for row in cursor]


def extract_timestamp(data: Dict[str, Any], *keys: str) -> Optional[str]:
    """Extract timestamp from nested dictionary, trying multiple keys.

    Args:
        data: Dictionary to search
        *keys: Keys to try in order (e.g., "timestamp", "created_at", "ts")

    Returns:
        ISO format timestamp string or None
    """
    for key in keys:
        value = data.get(key)
        if value:
            return str(value)
    return None
