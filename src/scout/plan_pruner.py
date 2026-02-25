from __future__ import annotations
"""
Plan storage optimization: compression and pruning.

Provides gzip compression for old plans and automatic deduplication.
"""

import gzip
import json
from datetime import datetime, timedelta
from typing import Optional

from scout.plan_io import (
    PLAN_STORAGE_DIR,
    BY_DATE_DIR,
    BY_MODEL_DIR,
    list_plans,
)
from scout.similarity import find_duplicates

COMPRESSION_AGE_DAYS = 30
MAX_TOTAL_PLANS = 1000
PRUNED_DIR = PLAN_STORAGE_DIR / "pruned"


def compress_plan(plan: dict, force: bool = False) -> dict:
    """
    Compress plan content using gzip.

    Only compresses if plan is older than COMPRESSION_AGE_DAYS
    unless force=True.
    """
    if plan.get("compressed"):
        return plan

    created = plan.get("created_at", "")
    if not force and created:
        try:
            plan_date = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age = datetime.utcnow() - plan_date.replace(tzinfo=None)
            if age.days < COMPRESSION_AGE_DAYS:
                return plan
        except (ValueError, TypeError):
            pass

    plan_text = plan.get("plan", "")
    if len(plan_text) < 1024:
        return plan

    compressed = gzip.compress(plan_text.encode("utf-8"))
    plan["plan"] = compressed.decode("latin-1")
    plan["compressed"] = True

    return plan


def decompress_plan(plan: dict) -> dict:
    """Decompress a compressed plan."""
    if not plan.get("compressed"):
        return plan

    try:
        compressed = plan["plan"].encode("latin-1")
        plan["plan"] = gzip.decompress(compressed).decode("utf-8")
        plan["compressed"] = False
    except Exception as e:
        print(f"Failed to decompress plan: {e}")

    return plan


def prune_duplicates(threshold: float = 0.9) -> int:
    """
    Find and remove duplicate plans, keeping the newest.

    Returns number of plans removed.
    """
    plans = list_plans()

    if len(plans) < 2:
        return 0

    duplicates = find_duplicates(plans, threshold=threshold)

    removed = 0
    for group in duplicates:
        group_sorted = sorted(
            [p for p in plans if p.get("id") in group],
            key=lambda p: p.get("created_at", ""),
            reverse=True,
        )

        for duplicate in group_sorted[1:]:
            plan_id = duplicate.get("id")
            for plan_file in PLAN_STORAGE_DIR.glob(f"**/{plan_id}.json"):
                if plan_file.name not in ("index.json", "schema.json"):
                    plan_file.unlink()
                    removed += 1

    return removed


def prune_old_plans(max_age_days: int = 90) -> int:
    """Remove plans older than max_age_days, keeping at least MAX_TOTAL_PLANS."""
    plans = list_plans()
    cutoff = datetime.utcnow() - timedelta(days=max_age_days)

    to_remove = []
    for plan in plans:
        created = plan.get("created_at", "")
        if created:
            try:
                plan_date = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if plan_date.replace(tzinfo=None) < cutoff:
                    to_remove.append(plan)
            except (ValueError, TypeError):
                pass

    keep_count = min(MAX_TOTAL_PLANS, len(plans) - len(to_remove))
    if len(to_remove) > 0 and (len(plans) - len(to_remove)) < keep_count:
        to_remove = to_remove[: len(plans) - keep_count]

    removed = 0
    for plan in to_remove:
        plan_id = plan.get("id")
        for plan_file in PLAN_STORAGE_DIR.glob(f"**/{plan_id}.json"):
            if plan_file.name not in ("index.json", "schema.json"):
                plan_file.unlink()
                removed += 1

    return removed


def compress_old_plans() -> int:
    """Compress plans older than COMPRESSION_AGE_DAYS."""
    plans = list_plans()
    compressed = 0

    for plan in plans:
        if plan.get("compressed"):
            continue

        try:
            plan_date = datetime.fromisoformat(
                plan.get("created_at", "").replace("Z", "+00:00")
            )
            age = datetime.utcnow() - plan_date.replace(tzinfo=None)

            if age.days >= COMPRESSION_AGE_DAYS:
                compressed_plan = compress_plan(plan, force=True)

                plan_id = plan.get("id")
                for plan_file in PLAN_STORAGE_DIR.glob(f"**/{plan_id}.json"):
                    if plan_file.name not in ("index.json", "schema.json"):
                        with open(plan_file, "w") as f:
                            json.dump(compressed_plan, f)
                        compressed += 1
                        break
        except (ValueError, TypeError, IOError):
            continue

    return compressed


def run_optimization() -> dict:
    """
    Run full optimization: compress, dedupe, prune.

    Returns summary of actions taken.
    """
    results = {
        "compressed": compress_old_plans(),
        "duplicates_removed": prune_duplicates(),
        "old_removed": prune_old_plans(),
        "total_plans": len(list_plans()),
    }

    return results


class PlanStorageManager:
    """Manage plan storage with automatic optimization."""

    def __init__(
        self,
        compression_age_days: int = COMPRESSION_AGE_DAYS,
        max_plans: int = MAX_TOTAL_PLANS,
        auto_optimize: bool = True,
    ):
        self.compression_age_days = compression_age_days
        self.max_plans = max_plans
        self.auto_optimize = auto_optimize

    def add(self, plan: dict) -> dict:
        """Add a plan with automatic optimization."""
        if self.auto_optimize and len(list_plans()) >= self.max_plans:
            run_optimization()

        return compress_plan(plan)

    def get(self, plan_id: str) -> Optional[dict]:
        """Get and decompress a plan."""
        from vivarium.scout.plan_io import read_plan

        plan = read_plan(plan_id)
        if plan:
            return decompress_plan(plan)
        return None

    def optimize(self) -> dict:
        """Manually trigger optimization."""
        return run_optimization()
