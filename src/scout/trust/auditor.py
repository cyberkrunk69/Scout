"""
TrustAuditor - Audit logging with levels and sampling.

Philosophy: Auditability by Default + Cost as Feature + Async/Parallel
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Optional

import aiofiles
import asyncio

from .constants import (
    DEFAULT_AUDIT_LEVEL,
    DEFAULT_AUDIT_SAMPLE_RATE,
    AUDIT_LOG_FILENAME,
)

logger = logging.getLogger(__name__)


class AuditLevel(IntEnum):
    """Audit event levels with numeric values for comparison."""
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40


class TrustAuditor:
    """
    Audit trust decisions with level control and async file writes.

    Philosophy: Auditability by Default
    - DEBUG: Every call detail (sampled)
    - INFO: Summaries
    - WARN: Warnings
    - ERROR: Errors only
    - Cost tracking: Shows savings from trust penalties
    """

    def __init__(
        self,
        audit_path: Optional[Path] = None,
        level: AuditLevel = AuditLevel.INFO,
        sample_rate: float = DEFAULT_AUDIT_SAMPLE_RATE,
    ):
        self.audit_path = audit_path
        self.level = level
        self.sample_rate = sample_rate
        self._lock = asyncio.Lock()

    def set_level(self, level: AuditLevel) -> None:
        """Change audit level at runtime."""
        self.level = level

    def _get_audit_path(self, repo_root: Path) -> Path:
        """Get audit path if not set."""
        if self.audit_path:
            return self.audit_path
        return repo_root / ".scout" / AUDIT_LOG_FILENAME

    async def log(
        self,
        event: str,
        level: AuditLevel = AuditLevel.DEBUG,
        repo_root: Optional[Path] = None,
        **kwargs: Any,
    ) -> None:
        """
        Log audit event with level and sampling.

        Philosophy: Cost as Feature - show savings
        """
        # Filter by level
        if level < self.level:
            return

        # Apply sampling for DEBUG (high frequency)
        if level == AuditLevel.DEBUG and random.random() > self.sample_rate:
            return

        # Determine path
        audit_path = self._get_audit_path(repo_root) if repo_root else self.audit_path
        if not audit_path:
            logger.warning("No audit path configured, skipping audit log")
            return

        # Ensure parent directory exists
        audit_path.parent.mkdir(parents=True, exist_ok=True)

        # Build event record
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": f"trust_{event}",
            "level": level.name.lower(),
            **kwargs,
        }

        # Write async
        async with self._lock:
            try:
                async with aiofiles.open(audit_path, "a") as f:
                    await f.write(json.dumps(record) + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit log: {e}")

    async def log_verification(
        self,
        source_path: str,
        trust_level: str,
        penalty: float,
        repo_root: Optional[Path] = None,
    ) -> None:
        """Log verification decision."""
        await self.log(
            "verify",
            AuditLevel.DEBUG,
            repo_root=repo_root,
            source=source_path,
            trust_level=trust_level,
            penalty=penalty,
        )

    async def log_batch_summary(
        self,
        total: int,
        trusted: int,
        stale: int,
        missing: int,
        duration_ms: float,
        repo_root: Optional[Path] = None,
    ) -> None:
        """Log batch verification summary."""
        await self.log(
            "batch_summary",
            AuditLevel.INFO,
            repo_root=repo_root,
            total=total,
            trusted=trusted,
            stale=stale,
            missing=missing,
            duration_ms=duration_ms,
        )

    async def log_confidence_calc(
        self,
        source_path: str,
        normalized: float,
        penalty: float,
        final_confidence: int,
        saved_llm_call: bool = False,
        estimated_savings: float = 0.0,
        repo_root: Optional[Path] = None,
    ) -> None:
        """
        Log confidence calculation with cost savings.

        Philosophy: Cost as Feature - show what trust saved
        """
        await self.log(
            "confidence_calc",
            AuditLevel.DEBUG if not saved_llm_call else AuditLevel.INFO,
            repo_root=repo_root,
            source=source_path,
            normalized=normalized,
            penalty=penalty,
            final_confidence=final_confidence,
            saved_llm_call=saved_llm_call,
            estimated_savings=estimated_savings,
        )

    async def log_penalty_adjustment(
        self,
        source_path: str,
        old_penalty: float,
        new_penalty: float,
        reason: str,
        repo_root: Optional[Path] = None,
    ) -> None:
        """Log dynamic penalty adjustment."""
        await self.log(
            "penalty_adjustment",
            AuditLevel.INFO,
            repo_root=repo_root,
            source=source_path,
            old_penalty=old_penalty,
            new_penalty=new_penalty,
            reason=reason,
        )

    async def log_auto_refresh(
        self,
        source_path: str,
        old_trust_level: str,
        new_trust_level: str,
        cost: float,
        repo_root: Optional[Path] = None,
    ) -> None:
        """Log auto-refresh event."""
        await self.log(
            "auto_refresh",
            AuditLevel.INFO,
            repo_root=repo_root,
            source=source_path,
            old_trust_level=old_trust_level,
            new_trust_level=new_trust_level,
            cost=cost,
        )
