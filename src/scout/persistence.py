"""
Atomic-write persistence layer for Scout plans.

Provides reliable file I/O with fsync to ensure plans are persisted
to disk even if the terminal session crashes unexpectedly.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional


# Scout plans directory - stored in repo root
SCOUT_DIR = Path(".scout")
SCOUT_PLANS_DIR = SCOUT_DIR / "plans"
LAST_PLAN_FILE = SCOUT_PLANS_DIR / "last_plan.md"


def ensure_scout_plans_dir() -> Path:
    """
    Ensure the .scout/plans directory exists.
    
    Returns:
        Path to the .scout/plans directory
    """
    SCOUT_PLANS_DIR.mkdir(parents=True, exist_ok=True)
    return SCOUT_PLANS_DIR


def atomic_write_with_fsync(path: Path, content: str) -> None:
    """
    Write content to a file atomically with fsync for durability.
    
    This function:
    1. Writes content to a temporary file in the same directory
    2. Flushes the file buffers
    3. Calls fsync to ensure data is written to disk
    4. Atomically renames the temp file to the target path
    5. Syncs the parent directory to ensure the rename persists
    
    Args:
        path: Target file path
        content: Content to write to the file
        
    Raises:
        OSError: If the write operation fails
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temp file in same directory (for atomic rename)
    fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp"
    )
    
    try:
        # Write content to temp file
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        
        # Sync directory to ensure rename persists
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        
        # Atomic rename
        os.replace(temp_path, path)
        
        # Sync directory again after rename
        dir_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
            
    except Exception:
        # Clean up temp file on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        raise


def save_last_plan(content: str) -> Path:
    """
    Save the latest synthesized plan to .scout/plans/last_plan.md.
    
    This function ensures the plan is persisted atomically with fsync,
    so it survives terminal crashes or unexpected shutdowns.
    
    Args:
        content: The plan content (markdown text)
        
    Returns:
        Path to the saved last_plan.md file
    """
    ensure_scout_plans_dir()
    atomic_write_with_fsync(LAST_PLAN_FILE, content)
    return LAST_PLAN_FILE


def read_last_plan() -> Optional[str]:
    """
    Read the last saved plan from .scout/plans/last_plan.md.
    
    Returns:
        The plan content, or None if no last plan exists
    """
    if not LAST_PLAN_FILE.exists():
        return None
    
    try:
        return LAST_PLAN_FILE.read_text(encoding='utf-8')
    except OSError:
        return None
