"""Plan store for caching web task plans.

This module provides the PlanStore class for caching successful web automation
plans. Plans are matched by goal text and optionally by URL pattern.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scout.trust.store import TrustStore

logger = logging.getLogger(__name__)


class PlanStore(TrustStore):
    """Cache for successful web task plans.
    
    Execution Traces:
    - Happy: Cache hit, return exact matching plan
    - Failure: Cache miss, caller generates new plan
    - Edge: Multiple plans match, return highest success_count
    
    The store uses exact-goal matching (normalized whitespace) for v1.
    URL patterns should be anchored regexes (e.g., "^https://github\\.com/.*$")
    to avoid unintended matches.
    """
    
    # Default cache duration in days
    DEFAULT_CACHE_DAYS = 30
    # Minimum success count to avoid deletion
    MIN_SUCCESS_COUNT = 1
    
    async def initialize(self) -> None:
        """Initialize the plan store schema.
        
        Execution Traces:
        - Happy: Schema created successfully
        - Failure: Database error, logged and raised
        - Edge: Table already exists, no-op
        """
        # Ensure .scout directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with self._lock:
            conn = await self._get_connection()
            
            # Create plans table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    steps_json TEXT NOT NULL,
                    url_pattern TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    last_run TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Create indexes for efficient lookups
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_plans_goal ON plans(goal)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_plans_url_pattern ON plans(url_pattern)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_plans_last_run ON plans(last_run)"
            )
            
            await conn.commit()
            logger.info(f"Initialized plan store at {self.db_path}")
    
    def _normalize_goal(self, goal: str) -> str:
        """Normalize goal string for matching.
        
        Execution Traces:
        - Happy: Returns lowercase, stripped goal
        - Failure: N/A (always returns string)
        - Edge: Empty goal returns empty string
        
        Args:
            goal: Original goal string
            
        Returns:
            Normalized goal string
        """
        return goal.lower().strip()
    
    def _generate_plan_id(self, goal: str, url_pattern: Optional[str] = None) -> str:
        """Generate a unique plan ID from goal and URL pattern.
        
        Execution Traces:
        - Happy: Returns deterministic hash
        - Failure: N/A (always returns string)
        - Edge: Same goal always produces same ID
        
        Args:
            goal: User's goal string
            url_pattern: Optional URL pattern
            
        Returns:
            SHA256 hash as hex string
        """
        key = f"{goal}|{url_pattern or ''}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    async def store_plan(
        self, 
        goal: str, 
        steps: List[Dict[str, Any]], 
        url_pattern: Optional[str] = None,
    ) -> str:
        """Store a plan in the cache.
        
        Execution Traces:
        - Happy: Plan stored successfully, returns plan_id
        - Failure: Database error, logged and re-raised
        - Edge: Plan with same ID exists, updates it
        
        Args:
            goal: Original user goal
            steps: List of step dictionaries
            url_pattern: Optional regex pattern for URL matching
            
        Returns:
            The plan_id (hash of goal + params)
        """
        plan_id = self._generate_plan_id(goal, url_pattern)
        steps_json = json.dumps(steps)
        now = datetime.now(timezone.utc).isoformat()
        
        async with self._lock:
            conn = await self._get_connection()
            
            await conn.execute("""
                INSERT INTO plans (plan_id, goal, steps_json, url_pattern, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    steps_json = excluded.steps_json,
                    url_pattern = excluded.url_pattern,
                    last_run = excluded.last_run
            """, (plan_id, self._normalize_goal(goal), steps_json, url_pattern, now))
            
            await conn.commit()
        
        logger.info(f"Stored plan: plan_id={plan_id}, goal={goal[:50]}...")
        return plan_id
    
    async def get_plan(
        self, 
        goal: str, 
        current_url: Optional[str] = None,
    ) -> Optional[Tuple[str, List[Dict[str, Any]]]]:
        """Get best matching plan from cache.
        
        Execution Traces:
        - Happy: Exact goal match found, returns (plan_id, steps)
        - Failure: No match, returns None
        - Edge: Multiple matches, returns one with highest success_count
        
        Matching Algorithm (v1):
        1. Normalize goal (lowercase, strip whitespace)
        2. Exact match on normalized goal
        3. If current_url provided, filter by url_pattern regex
        4. Return plan with highest success_count if multiple matches
        
        Note: Future versions will use BM25F for fuzzy matching.
        
        Args:
            goal: User's goal string
            current_url: Optional current URL for pattern matching
            
        Returns:
            Tuple of (plan_id, steps) if found, None otherwise
        """
        normalized_goal = self._normalize_goal(goal)
        
        async with self._lock:
            conn = await self._get_connection()
            
            # Try exact match first
            cursor = await conn.execute("""
                SELECT plan_id, goal, steps_json, url_pattern, success_count, failure_count
                FROM plans 
                WHERE goal = ?
                ORDER BY success_count DESC
                LIMIT 1
            """, (normalized_goal,))
            
            row = await cursor.fetchone()
            
            if not row:
                logger.debug(f"Plan cache miss: goal={goal[:50]}...")
                return None
            
            # Check URL pattern if current_url provided
            url_pattern = row["url_pattern"]
            if current_url and url_pattern:
                try:
                    if not re.match(url_pattern, current_url):
                        logger.debug(
                            f"Plan URL pattern mismatch: pattern={url_pattern}, url={current_url}"
                        )
                        return None
                except re.error as e:
                    logger.warning(f"Invalid URL pattern regex: {url_pattern}, error={e}")
            
            # Parse steps
            try:
                steps = json.loads(row["steps_json"])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse steps_json for plan {row['plan_id']}: {e}")
                return None
            
            # Update last_run timestamp
            now = datetime.now(timezone.utc).isoformat()
            await conn.execute(
                "UPDATE plans SET last_run = ? WHERE plan_id = ?",
                (now, row["plan_id"])
            )
            await conn.commit()
            
            logger.info(
                f"Plan cache hit: plan_id={row['plan_id']}, "
                f"success_count={row['success_count']}, goal={goal[:50]}..."
            )
            
            return (row["plan_id"], steps)
    
    async def update_plan_stats(
        self, 
        plan_id: str, 
        success: bool,
    ) -> None:
        """Update success/failure counts for a plan.
        
        Execution Traces:
        - Happy: Stats updated successfully
        - Failure: Plan not found or DB error, logged and ignored
        - Edge: N/A
        
        Args:
            plan_id: The plan's unique ID
            success: Whether the plan execution succeeded
        """
        column = "success_count" if success else "failure_count"
        
        async with self._lock:
            conn = await self._get_connection()
            
            await conn.execute(f"""
                UPDATE plans 
                SET {column} = {column} + 1,
                    last_run = ?
                WHERE plan_id = ?
            """, (datetime.now(timezone.utc).isoformat(), plan_id))
            
            await conn.commit()
        
        logger.debug(f"Updated plan stats: plan_id={plan_id}, {column} += 1")
    
    async def cleanup_old_plans(
        self, 
        max_age_days: int = DEFAULT_CACHE_DAYS,
    ) -> Dict[str, int]:
        """Remove old plans that have no successes.
        
        Execution Traces:
        - Happy: Old plans removed, returns count
        - Failure: DB error, logged and returns empty dict
        - Edge: Plans with successes are preserved
        
        Plans older than max_age_days with fewer than MIN_SUCCESS_COUNT
        successes are candidates for deletion.
        
        Args:
            max_age_days: Maximum age in days (default 30)
            
        Returns:
            Dict with "deleted" count
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        cutoff_iso = cutoff.isoformat()
        
        async with self._lock:
            conn = await self._get_connection()
            
            # Find candidates for deletion
            cursor = await conn.execute("""
                SELECT COUNT(*) as count
                FROM plans
                WHERE created_at < ?
                AND success_count < ?
            """, (cutoff_iso, self.MIN_SUCCESS_COUNT))
            
            row = await cursor.fetchone()
            delete_count = row["count"] if row else 0
            
            if delete_count > 0:
                # Delete old plans with low success
                await conn.execute("""
                    DELETE FROM plans
                    WHERE created_at < ?
                    AND success_count < ?
                """, (cutoff_iso, self.MIN_SUCCESS_COUNT))
                
                await conn.commit()
                logger.info(f"Cleaned up {delete_count} old plans")
        
        return {"deleted": delete_count}
    
    async def get_plan_stats(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific plan.
        
        Execution Traces:
        - Happy: Returns stats dict
        - Failure: Plan not found, returns None
        - Edge: N/A
        
        Args:
            plan_id: The plan's unique ID
            
        Returns:
            Dict with plan stats or None
        """
        async with self._lock:
            conn = await self._get_connection()
            
            cursor = await conn.execute("""
                SELECT plan_id, goal, success_count, failure_count, 
                       last_run, created_at, url_pattern
                FROM plans
                WHERE plan_id = ?
            """, (plan_id,))
            
            row = await cursor.fetchone()
            
            if not row:
                return None
            
            return {
                "plan_id": row["plan_id"],
                "goal": row["goal"],
                "success_count": row["success_count"],
                "failure_count": row["failure_count"],
                "last_run": row["last_run"],
                "created_at": row["created_at"],
                "url_pattern": row["url_pattern"],
            }
