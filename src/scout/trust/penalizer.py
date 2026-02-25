"""
TrustPenalizer - Applies config and Bayesian adjustments to trust penalties.

Philosophy: Spaceship (system learns from decisions) + Cost as Feature
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

from .constants import (
    DEFAULT_PENALTIES,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_FALLBACKS,
    BM25_BASE,
    BM25_RANGE,
    BM25_CLARITY_MAX,
    BM25_EXACT_BOOST,
    BM25_CLASS_BOOST,
)
from .models import PenaltyResult, TrustRecord
from .store import TrustStore

if TYPE_CHECKING:
    from .auditor import TrustAuditor

logger = logging.getLogger(__name__)


class TrustPenalizer:
    """
    Apply penalties based on trust level and Bayesian dynamic adjustments.

    Philosophy:
    - Cost as Feature: Penalty affects confidence = visible cost
    - Spaceship: Learns from historical data using Bayesian inference
    """

    def __init__(
        self,
        store: TrustStore,
        config: Optional[Dict] = None,
        auditor: Optional["TrustAuditor"] = None,
    ):
        self.store = store
        self.config = config or {}
        self.auditor = auditor

        # User-facing config (with defaults)
        self.min_confidence = self.config.get(
            "min_nav_confidence", DEFAULT_MIN_CONFIDENCE
        )
        self.base_penalties = self.config.get("trust_penalty", DEFAULT_PENALTIES)
        self.fallbacks = self.config.get("confidence_fallback", DEFAULT_FALLBACKS)

    async def compute_penalty(
        self,
        trust_level: str,
        source_path: Optional[Path] = None,
    ) -> PenaltyResult:
        """
        Compute penalty for trust level.

        Applies base penalty + Bayesian dynamic adjustment from store.
        Uses Laplace smoothing: penalty = (failures + 1) / (total + 2)
        """
        # Base penalty from config
        base = self.base_penalties.get(trust_level, 0.5)

        # Dynamic adjustment from historical data using Bayesian formula
        dynamic = 0.0
        reason = "base"

        if source_path:
            record = await self.store.get(str(source_path))
            if record:
                # Bayesian update with Laplace smoothing
                total = record.success_count + record.failure_count
                if total > 0:
                    # penalty = P(failure) with Laplace smoothing
                    dynamic = (record.failure_count + 1) / (total + 2) - base
                    reason = "bayesian_learned"
                elif record.query_count > 10 and trust_level in (
                    "stale",
                    "no_checksum",
                ):
                    # Frequently queried stale docs get slight reduction
                    dynamic = -0.1
                    reason = "frequently_queried"

        final = max(0.0, min(1.0, base + dynamic))

        return PenaltyResult(
            base_penalty=base,
            dynamic_adjustment=dynamic,
            final_penalty=final,
            reason=reason,
        )

    def get_confidence_threshold(self) -> int:
        """Get minimum confidence threshold."""
        return self.min_confidence

    def get_fallback_confidence(self, error_code: str) -> int:
        """Get fallback confidence for validation error."""
        return self.fallbacks.get(error_code, 50)

    def compute_confidence(
        self,
        normalized: float,
        gap_top2: float,
        penalty: float,
        is_exact_match: bool = False,
        kind: str = "",
    ) -> int:
        """
        Compute confidence with trust penalty.

        Args:
            normalized: BM25 normalized score (0-1)
            gap_top2: Gap between top 2 results
            penalty: Trust penalty (0-1)
            is_exact_match: Whether this is an exact match
            kind: Type of match (class, function, etc.)

        Returns:
            Confidence score (0-100)
        """
        # Base from normalized
        base = BM25_BASE + (normalized * BM25_RANGE)

        # Clarity bonus
        clarity = max(0, (gap_top2**0.5) * BM25_CLARITY_MAX * 0.8 - 8)
        clarity = min(clarity, BM25_CLARITY_MAX)

        # Trust penalty
        trust_factor = 1.0 - penalty

        # Boosts
        exact_boost = BM25_EXACT_BOOST if is_exact_match else 1.0
        kind_boost = BM25_CLASS_BOOST if kind == "class" else 1.0

        total = (base + clarity) * trust_factor * exact_boost * kind_boost

        return int(min(total, 100))
