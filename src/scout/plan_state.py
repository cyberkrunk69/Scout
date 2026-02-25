"""Plan State Management with atomic locking.

Provides disk-based state persistence for the dynamic re-synthesis planning engine:
- active/: Plans currently being executed
- completed/: Successfully completed plans
- archived/: Failed or archived plans

Uses atomic mkdir for lock files (POSIX-compatible).

Configuration (via env vars or config file):
- SCOUT_PLAN_LOCK_TIMEOUT: Lock acquisition timeout in seconds (default: 30)
- SCOUT_PLAN_STALE_LOCK_HOURS: Stale lock threshold in hours (default: 1)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Constants
LOCKS_DIR = Path(".scout/locks")
PLAN_STATE_DIR = Path(".scout/plans")


def _get_lock_config() -> dict:
    """Get lock configuration from environment or defaults."""
    return {
        "timeout": int(os.environ.get("SCOUT_PLAN_LOCK_TIMEOUT", 30)),
        "stale_hours": int(os.environ.get("SCOUT_PLAN_STALE_LOCK_HOURS", 1)),
    }


def acquire_lock(plan_id: str, timeout: int = None) -> bool:
    """Acquire lock using atomic mkdir. Returns True if acquired.

    Uses atomic mkdir for POSIX-compatible locking.
    Configuration:
    - timeout: How long to wait for lock acquisition (default from config: 30s)
    - stale_hours: When to consider a lock stale (default from config: 1 hour)
    """
    config = _get_lock_config()
    lock_timeout = timeout if timeout is not None else config["timeout"]
    stale_seconds = config["stale_hours"] * 3600

    lock_path = LOCKS_DIR / f"{plan_id}.lock"
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    while time.time() - start_time < lock_timeout:
        try:
            # exist_ok=False is CRITICAL - this is what makes it atomic
            lock_path.mkdir(parents=True, exist_ok=False)
            # Write PID to separate file inside (keeps dir as atomic indicator)
            (lock_path / "pid").write_text(str(os.getpid()))
            (lock_path / "timestamp").write_text(str(time.time()))
            (lock_path / "config").write_text(json.dumps(config))
            logger.debug("lock_acquired", plan_id=plan_id, timeout=lock_timeout)
            return True
        except FileExistsError:
            # Check if stale (older than configured threshold)
            try:
                if not lock_path.exists():
                    continue  # Was removed by someone else, retry
                mtime = lock_path.stat().st_mtime
                if time.time() - mtime > stale_seconds:
                    # Stale - try to remove and retry
                    shutil.rmtree(lock_path)
                    # Re-verify lock_path doesn't exist after removal
                    if not lock_path.exists():
                        continue
            except Exception:
                pass
            time.sleep(0.1)  # Retry with backoff

    logger.warning("lock_acquire_timeout", plan_id=plan_id, timeout=lock_timeout)
    return False


def release_lock(plan_id: str) -> bool:
    """Release lock atomically."""
    lock_path = LOCKS_DIR / f"{plan_id}.lock"
    try:
        if lock_path.exists():
            shutil.rmtree(lock_path)
            logger.debug("lock_released", plan_id=plan_id)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.error("lock_release_error", plan_id=plan_id, error=str(e))
        return False


def is_locked(plan_id: str) -> bool:
    """Check if a plan is currently locked."""
    lock_path = LOCKS_DIR / f"{plan_id}.lock"
    return lock_path.exists()


def generate_plan_id(request: str) -> str:
    """Generate unique, deterministic plan ID.
    
    Uses UUID4 for uniqueness, with request prefix for readability.
    """
    uid = uuid.uuid4().hex[:8]
    # Sanitize request for prefix (alphanumeric only)
    prefix = "".join(c for c in request[:20] if c.isalnum()).lower()
    return f"{prefix}_{uid}"


def _active_dir(repo_root: Path) -> Path:
    p = repo_root / ".scout" / "plans" / "active"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _completed_dir(repo_root: Path) -> Path:
    p = repo_root / ".scout" / "plans" / "completed"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _archived_dir(repo_root: Path) -> Path:
    p = repo_root / ".scout" / "plans" / "archived"
    p.mkdir(parents=True, exist_ok=True)
    return p


class PlanStateManager:
    """Manages plan state with async locks and disk persistence."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._async_lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
    
    async def save_plan_state(self, context) -> str:
        """Persist PlanningContext to active/ directory.
        
        Uses atomic write: write to temp file, then rename.
        """
        # Import here to avoid circular imports
        from vivarium.scout.cli.plan import PlanningContext
        
        plan_id = generate_plan_id(context.request)
        context.plan_id = plan_id  # Store in context for reference
        
        # Ensure same directory for atomic rename
        active_dir = _active_dir(self.repo_root)
        temp_path = active_dir / f".{plan_id}.tmp"
        final_path = active_dir / f"{plan_id}.json"
        
        # Serialize context (exclude non-serializable locks)
        data = {
            "plan_id": plan_id,
            "request": context.request,
            "depth": context.depth,
            "max_depth": context.max_depth,
            "summary": context.summary,
            "parent_goals": context.parent_goals,
            "constraints": context.constraints,
            "discoveries": context.discoveries,
            "pivots_needed": context.pivots_needed,
            "sub_plan_outcomes": context.sub_plan_outcomes,
            "is_pivoting": context.is_pivoting,
            "pivot_reason": context.pivot_reason,
            "replan_required": context.replan_required,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Atomic write: temp file + rename
        async with self._sync_lock:
            temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            temp_path.rename(final_path)  # Atomic on same filesystem
        
        logger.info("plan_state_saved", plan_id=plan_id, depth=context.depth)
        return plan_id
    
    async def load_plan_state(self, plan_id: str):
        """Load PlanningContext from disk."""
        from vivarium.scout.cli.plan import PlanningContext
        
        active_path = self.repo_root / ".scout" / "plans" / "active" / f"{plan_id}.json"
        if not active_path.exists():
            logger.warning("plan_state_not_found", plan_id=plan_id)
            return None
        
        try:
            data = json.loads(active_path.read_text(encoding="utf-8"))
            context = PlanningContext(
                request=data["request"],
                depth=data.get("depth", 0),
                max_depth=data.get("max_depth", 3),
                summary=data.get("summary", ""),
                parent_goals=data.get("parent_goals", []),
                constraints=data.get("constraints", []),
            )
            context.plan_id = plan_id
            context.discoveries = data.get("discoveries", [])
            context.pivots_needed = data.get("pivots_needed", [])
            context.sub_plan_outcomes = data.get("sub_plan_outcomes", [])
            context.is_pivoting = data.get("is_pivoting", False)
            context.pivot_reason = data.get("pivot_reason", "")
            context.replan_required = data.get("replan_required", False)
            
            logger.info("plan_state_loaded", plan_id=plan_id)
            return context
        except (json.JSONDecodeError, KeyError) as e:
            # Corrupted - move to archived
            logger.error("plan_state_corrupt", plan_id=plan_id, error=str(e))
            await self._archive_plan(plan_id, error=str(e))
            return None
    
    async def transition_plan(self, plan_id: str, to_state: str) -> bool:
        """Move plan between active/completed/archived. Acquires lock before transitioning."""
        states = {"active", "completed", "archived"}
        if to_state not in states:
            logger.error("invalid_transition_state", to_state=to_state)
            return False
        
        # Acquire lock before transitioning to prevent race with writer
        if not acquire_lock(plan_id, timeout=5):
            logger.warning("transition_plan_lock_failed", plan_id=plan_id)
            return False
        
        try:
            active_dir = _active_dir(self.repo_root)
            completed_dir = _completed_dir(self.repo_root)
            archived_dir = _archived_dir(self.repo_root)
            
            source = active_dir / f"{plan_id}.json"
            if not source.exists():
                logger.warning("transition_source_missing", plan_id=plan_id)
                return False
            
            target_dir = {"completed": completed_dir, "archived": archived_dir}[to_state]
            target = target_dir / f"{plan_id}.json"
            
            # Atomic move
            source.rename(target)
            
            # Update timestamp if completing
            if to_state == "completed":
                data = json.loads(target.read_text())
                data["completed_at"] = datetime.now(timezone.utc).isoformat()
                target.write_text(json.dumps(data, indent=2))
            
            logger.info("plan_transitioned", plan_id=plan_id, to_state=to_state)
            return True
        except Exception as e:
            logger.error("transition_error", plan_id=plan_id, error=str(e))
            return False
        finally:
            release_lock(plan_id)
    
    async def _archive_plan(self, plan_id: str, error: str = None):
        """Archive a plan with error info."""
        archived_dir = _archived_dir(self.repo_root)
        target = archived_dir / f"{plan_id}.json"
        
        data = {
            "plan_id": plan_id,
            "error": error,
            "archived_at": datetime.now(timezone.utc).isoformat()
        }
        target.write_text(json.dumps(data, indent=2))
        logger.info("plan_archived", plan_id=plan_id, error=error)
    
    async def recover_stale_locks(self) -> int:
        """Scan for stale locks and recover. Returns count of recovered plans."""
        config = _get_lock_config()
        stale_seconds = config["stale_hours"] * 3600
        recovered = 0
        LOCKS_DIR.mkdir(parents=True, exist_ok=True)

        for lock_dir in LOCKS_DIR.glob("*.lock"):
            try:
                plan_id = lock_dir.stem
                mtime = lock_dir.stat().st_mtime

                if time.time() - mtime > stale_seconds:
                    # Stale lock - remove it
                    shutil.rmtree(lock_dir)
                    logger.info("stale_lock_recovered", plan_id=plan_id)
                    recovered += 1
            except Exception as e:
                logger.error("lock_recovery_error", lock=str(lock_dir), error=str(e))
        
        return recovered
    
    def list_active_plans(self) -> list[str]:
        """List all active plan IDs."""
        active_dir = _active_dir(self.repo_root)
        return [p.stem for p in active_dir.glob("*.json")]

    def list_completed_plans(self) -> list[str]:
        """List all completed plan IDs."""
        completed_dir = _completed_dir(self.repo_root)
        return [p.stem for p in completed_dir.glob("*.json")]

    def list_archived_plans(self) -> list[str]:
        """List all archived plan IDs."""
        archived_dir = _archived_dir(self.repo_root)
        return [p.stem for p in archived_dir.glob("*.json")]

    def cleanup_old_plans(self) -> dict:
        """Archive/delete plans based on retention policy.

        Configuration (via env vars):
        - SCOUT_PLAN_ARCHIVE_DAYS: Archive completed plans older than N days (default: 7)
        - SCOUT_PLAN_DELETE_DAYS: Delete archived plans older than N days (default: 30)

        Returns cleanup stats: {"archived": N, "deleted": N, "skipped": bool, "reason": str}
        """
        import time as time_module

        # Get configuration
        archive_days = int(os.environ.get("SCOUT_PLAN_ARCHIVE_DAYS", 7))
        delete_days = int(os.environ.get("SCOUT_PLAN_DELETE_DAYS", 30))

        # Skip if any plans are currently locked/active
        active_plans = self.list_active_plans()
        if active_plans:
            logger.info("cleanup_skipped_active_plans", active_count=len(active_plans))
            return {"skipped": True, "reason": "active_plans_present", "archived": 0, "deleted": 0}

        now = time_module.time()
        archive_threshold = now - (archive_days * 86400)  # days to seconds
        delete_threshold = now - (delete_days * 86400)

        stats = {"archived": 0, "deleted": 0, "skipped": False}

        completed_dir = _completed_dir(self.repo_root)
        archived_dir = _archived_dir(self.repo_root)

        # Archive completed plans older than archive_days
        for plan_file in completed_dir.glob("*.json"):
            try:
                mtime = plan_file.stat().st_mtime
                if mtime < archive_threshold:
                    # Move to archived
                    target = archived_dir / plan_file.name
                    plan_file.rename(target)
                    stats["archived"] += 1
                    logger.info("plan_archived_cleanup", plan_id=plan_file.stem, age_days=int((now - mtime) / 86400))
            except Exception as e:
                logger.error("cleanup_archive_error", plan_id=plan_file.stem, error=str(e))

        # Delete archived plans older than delete_days
        for plan_file in archived_dir.glob("*.json"):
            try:
                mtime = plan_file.stat().st_mtime
                if mtime < delete_threshold:
                    plan_file.unlink()
                    stats["deleted"] += 1
                    logger.info("plan_deleted_cleanup", plan_id=plan_file.stem, age_days=int((now - mtime) / 86400))
            except Exception as e:
                logger.error("cleanup_delete_error", plan_id=plan_file.stem, error=str(e))

        logger.info(f"cleanup_completed: {stats}")
        return stats


# Global instance
_plan_state_manager: Optional[PlanStateManager] = None


def get_plan_state_manager(repo_root: Optional[Path] = None) -> PlanStateManager:
    """Get or create the global PlanStateManager."""
    global _plan_state_manager
    if _plan_state_manager is None:
        if repo_root is None:
            repo_root = Path.cwd()
        _plan_state_manager = PlanStateManager(repo_root)
    return _plan_state_manager
