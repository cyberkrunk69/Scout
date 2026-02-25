"""
TrustLearner - Bayesian-based dynamic penalty adjustment from audit data.

Philosophy: Spaceship - system learns from its own decisions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles

from .constants import (
    DEFAULT_LEARNER_MIN_SAMPLES,
    DEFAULT_LEARNER_CONFIDENCE_THRESHOLD,
)
from .store import TrustStore
from .auditor import TrustAuditor, AuditLevel

logger = logging.getLogger(__name__)


class TrustLearner:
    """
    Learn from audit logs to adjust trust penalties using Bayesian inference.

    Philosophy: Spaceship - self-improving system
    Uses Bayesian updating: penalty = P(failure) = (failures + 1) / (total + 2)
    """

    def __init__(
        self,
        store: TrustStore,
        auditor: TrustAuditor,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.store = store
        self.auditor = auditor
        self.config = config or {}

        # Learning parameters
        self.min_samples = self.config.get(
            "learner_min_samples", DEFAULT_LEARNER_MIN_SAMPLES
        )
        self.confidence_threshold = self.config.get(
            "learner_confidence_threshold", DEFAULT_LEARNER_CONFIDENCE_THRESHOLD
        )

    async def adjust_penalties(self, repo_root: Optional[Path] = None) -> Dict[str, Any]:
        """
        Adjust penalties based on historical success/failure using Bayesian update.

        Reads audit events to determine success/failure outcomes,
        then updates store counts and recomputes penalties.
        """
        audit_path = self.auditor.audit_path or (
            repo_root / ".scout" / "audit.jsonl" if repo_root else None
        )

        if not audit_path or not audit_path.exists():
            logger.debug("No audit log found, skipping learning cycle")
            return {"adjusted": 0, "reason": "no_audit_log"}

        # Parse recent audit events for confidence_calc events
        events = await self._parse_confidence_events(audit_path)

        # Group by source_path
        outcomes: Dict[str, Dict[str, int]] = {}
        for event in events:
            source = event.get("source")
            if not source:
                continue

            if source not in outcomes:
                outcomes[source] = {"success": 0, "failure": 0, "total": 0}

            saved = event.get("saved_llm_call", False)
            if saved:
                outcomes[source]["success"] += 1
            else:
                # If confidence was low and used anyway, count as partial failure
                conf = event.get("final_confidence", 0)
                if conf < 50:
                    outcomes[source]["failure"] += 1

            outcomes[source]["total"] += 1

        adjusted = 0

        for source, counts in outcomes.items():
            if counts["total"] < self.min_samples:
                continue

            # Get current record
            record = await self.store.get(source)
            if not record:
                continue

            # Update store with new counts from events
            await self.store.update_learning_counts(source, counts["success"] > 0)

            # Calculate new penalty using Bayesian formula
            old_penalty = record.penalty
            total = counts["total"]
            failures = counts["failure"]

            # Bayesian: P(failure) with Laplace smoothing
            new_penalty = (failures + 1) / (total + 2)

            # Only adjust if significant change
            if abs(new_penalty - old_penalty) > 0.05:
                await self.store.update_penalty(source, new_penalty)
                await self.auditor.log_penalty_adjustment(
                    source,
                    old_penalty,
                    new_penalty,
                    "bayesian_update",
                    repo_root=repo_root,
                )
                adjusted += 1

        logger.info(f"Learning cycle complete: adjusted {adjusted} penalties")
        return {"adjusted": adjusted, "total_analyzed": len(outcomes)}

    async def _parse_confidence_events(
        self,
        audit_path: Path,
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Parse recent confidence_calc events from audit log asynchronously."""
        events = []
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        try:
            async with aiofiles.open(audit_path, "r") as f:
                async for line in f:
                    try:
                        event = json.loads(line)
                        # Look for both confidence_calc and validation_outcome events
                        if event.get("event") not in (
                            "trust_confidence_calc",
                            "trust_validation_outcome",
                        ):
                            continue

                        ts = datetime.fromisoformat(
                            event.get("ts", "").replace("Z", "")
                        )
                        if ts > cutoff:
                            events.append(event)
                    except (json.JSONDecodeError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Error parsing audit log: {e}")

        return events
