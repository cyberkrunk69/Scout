"""
TrustOrchestrator - Coordinates all trust components.

Philosophy: DRY (single entry point) + Hyper-Minimal Friction + Spaceship
Mirrors ToolOrchestrator pattern.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .auditor import AuditLevel, TrustAuditor
from .constants import (
    DEFAULT_AUTO_REPAIR_ENABLED,
    DEFAULT_AUTO_REPAIR_THRESHOLD,
)
from .models import PenaltyResult, TrustRecord, TrustResult
from .penalizer import TrustPenalizer
from .store import TrustStore
from .verifier import TrustLevel, TrustVerifier

logger = logging.getLogger(__name__)


@dataclass
class TrustConfig:
    """Trust configuration options."""
    strictness: str = "normal"  # "permissive", "normal", "strict"
    auto_repair_enabled: bool = DEFAULT_AUTO_REPAIR_ENABLED
    auto_repair_threshold: int = DEFAULT_AUTO_REPAIR_THRESHOLD
    min_confidence: int = 70  # Added for strictness levels

    def __post_init__(self):
        if self.strictness == "permissive":
            self.min_confidence = 50
            self.auto_repair_threshold = 10
        elif self.strictness == "strict":
            self.min_confidence = 80
            self.auto_repair_threshold = 3
        else:  # normal
            self.min_confidence = 70
            self.auto_repair_threshold = 5


class TrustOrchestrator:
    """
    Single entry point for all trust operations.

    Philosophy:
    - DRY: One class coordinates everything
    - Hyper-Minimal: Simple interface, complex underneath
    - Spaceship: Self-learning via TrustLearner
    """

    def __init__(
        self,
        repo_root: Path,
        config: Optional[TrustConfig] = None,
    ):
        self.repo_root = repo_root
        self.config = config or TrustConfig()

        # Initialize components
        self.store = TrustStore(repo_root)
        self.verifier = TrustVerifier(repo_root)
        # Create single auditor
        self.auditor = TrustAuditor(repo_root / ".scout" / "audit.jsonl")
        # Pass auditor to penalizer
        self.penalizer = TrustPenalizer(self.store, auditor=self.auditor)
        # Import here to avoid circular import
        from .learner import TrustLearner

        self.learner = TrustLearner(self.store, self.auditor)

        # Track background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._background_loop_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Initialize trust store and start background tasks."""
        await self.store.initialize()
        await self.start_background_tasks()
        logger.info("TrustOrchestrator initialized")

    async def start_background_tasks(self) -> None:
        """Start background repair and learning loop."""
        self._background_loop_task = asyncio.create_task(self._background_loop())

    async def _background_loop(self) -> None:
        """Run auto-repair and learning every hour."""
        while True:
            try:
                await asyncio.sleep(3600)  # 1 hour
                await self.auto_repair()
                await self.learn()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background task error: {e}")

    async def verify(
        self,
        source_path: Path,
        compute_confidence: bool = True,
        normalized: float = 0.5,
        gap_top2: float = 0.5,
        is_exact_match: bool = False,
        kind: str = "",
    ) -> TrustResult:
        """
        Verify trust for a single file.

        Coordinates: verify → penalize → audit
        """
        # Verify
        verification = await self.verifier.verify(source_path)

        # Compute penalty
        penalty_result = await self.penalizer.compute_penalty(
            verification.trust_level,
            source_path,
        )

        # Compute confidence if requested
        confidence = 0
        if (
            compute_confidence
            and verification.trust_level != TrustLevel.MISSING
        ):
            confidence = self.penalizer.compute_confidence(
                normalized=normalized,
                gap_top2=gap_top2,
                penalty=penalty_result.final_penalty,
                is_exact_match=is_exact_match,
                kind=kind,
            )

        # Audit
        await self.auditor.log_verification(
            str(source_path),
            verification.trust_level,
            penalty_result.final_penalty,
            repo_root=self.repo_root,
        )

        # Track if we saved an LLM call
        saved_llm_call = (
            verification.trust_level == TrustLevel.TRUSTED and confidence >= 70
        )

        await self.auditor.log_confidence_calc(
            str(source_path),
            normalized,
            penalty_result.final_penalty,
            confidence,
            saved_llm_call=saved_llm_call,
            estimated_savings=0.01 if saved_llm_call else 0.0,
            repo_root=self.repo_root,
        )

        return TrustResult(
            source_path=source_path,
            trust_level=verification.trust_level,
            penalty=penalty_result.final_penalty,
            confidence=confidence,
            gap_message=verification.gap_message,
            verification=verification,
            penalty_result=penalty_result,
        )

    async def verify_batch(
        self,
        source_paths: List[Path],
        compute_confidence: bool = True,
    ) -> List[TrustResult]:
        """
        Verify multiple files in parallel.

        Philosophy: Async/Parallel - gather all verifications
        """
        import time

        start = time.monotonic()

        # Parallel verification
        results = await self.verifier.verify_batch(source_paths)

        trust_results = []
        trusted = stale = missing = 0

        for verification in results:
            # Compute penalty
            penalty_result = await self.penalizer.compute_penalty(
                verification.trust_level,
                verification.source_path,
            )

            # Compute confidence (default scores, would be overridden by caller)
            confidence = 0
            if (
                compute_confidence
                and verification.trust_level != TrustLevel.MISSING
            ):
                confidence = self.penalizer.compute_confidence(
                    normalized=0.5,
                    gap_top2=0.5,
                    penalty=penalty_result.final_penalty,
                )

            # Update counts
            if verification.trust_level == TrustLevel.TRUSTED:
                trusted += 1
            elif verification.trust_level == TrustLevel.STALE:
                stale += 1
            else:
                missing += 1

            # Update store with verification results
            record = TrustRecord(
                source_path=str(verification.source_path),
                doc_path=str(verification.doc_path),
                trust_level=verification.trust_level,
                embedded_checksum=verification.embedded_checksum,
                current_checksum=verification.current_checksum,
                stale_symbols=verification.stale_symbols,
                fresh_symbols=verification.fresh_symbols,
                penalty=penalty_result.final_penalty,
                last_validated=datetime.utcnow().isoformat(),
            )
            await self.store.upsert(record)

            # Increment query count for this source
            await self.store.increment_query_count(str(verification.source_path))

            trust_results.append(
                TrustResult(
                    source_path=verification.source_path,
                    trust_level=verification.trust_level,
                    penalty=penalty_result.final_penalty,
                    confidence=confidence,
                    gap_message=verification.gap_message,
                    verification=verification,
                    penalty_result=penalty_result,
                )
            )

        duration_ms = (time.monotonic() - start) * 1000

        # Audit batch summary
        await self.auditor.log_batch_summary(
            total=len(source_paths),
            trusted=trusted,
            stale=stale,
            missing=missing,
            duration_ms=duration_ms,
            repo_root=self.repo_root,
        )

        return trust_results

    async def record_outcome(
        self, source_path: str, success: bool
    ) -> None:
        """Record navigation/validation outcome for learning."""
        await self.store.update_learning_counts(source_path, success)

    async def auto_repair(self) -> Dict[str, Any]:
        """
        Auto-refresh stale docs that are frequently queried.

        Philosophy: Spaceship - system repairs itself
        """
        if not self.config.auto_repair_enabled:
            return {"refreshed": 0, "skipped": 0, "reason": "disabled"}

        # Get stale docs with high query count
        stale_records = await self.store.get_stale(
            limit=100
        )

        if not stale_records:
            return {"refreshed": 0, "skipped": 0}

        refreshed = 0
        skipped = 0

        for record in stale_records:
            if record.query_count < self.config.auto_repair_threshold:
                skipped += 1
                continue

            # Schedule background refresh (non-blocking)
            task = asyncio.create_task(
                self._refresh_doc(Path(record.source_path))
            )
            self._background_tasks.append(task)
            refreshed += 1

        return {"refreshed": refreshed, "skipped": skipped}

    async def _refresh_doc(self, source_path: Path) -> None:
        """Refresh documentation for a source file (non-blocking).

        Note: Full implementation requires doc_sync.sync_one() to be available.
        For now, this logs the refresh intent.
        """
        try:
            logger.info(f"Auto-refresh triggered for {source_path} (not implemented)")
            # TODO: Import and call doc sync when available
            # from scout.doc_sync import sync_one
            # await sync_one(source_path)

            # For now, just verify and update trust level
            result = await self.verify(source_path)

            await self.auditor.log_auto_refresh(
                str(source_path),
                "stale",
                result.trust_level,
                cost=0.0,
                repo_root=self.repo_root,
            )
        except Exception as e:
            logger.error(f"Failed to refresh doc for {source_path}: {e}")

    async def learn(self) -> None:
        """
        Run learning cycle - adjust penalties based on audit data.

        Philosophy: Spaceship - system learns from decisions
        """
        await self.learner.adjust_penalties(repo_root=self.repo_root)

    async def shutdown(self) -> None:
        """Wait for background tasks to complete."""
        if hasattr(self, '_background_loop_task') and self._background_loop_task:
            self._background_loop_task.cancel()
            try:
                await self._background_loop_task
            except asyncio.CancelledError:
                pass
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        logger.info("TrustOrchestrator shutdown complete")
