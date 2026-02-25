"""
Scout Trust Subsystem - Production-Grade Implementation

Philosophy-aligned:
- Right-Size: Deterministic verification before LLM
- DRY: Single source of truth
- Auditability: Every decision logged
- Spaceship: Self-learning via TrustLearner
- Cost as Feature: Every operation logged, every penny visible
"""

from __future__ import annotations

from .orchestrator import TrustOrchestrator
from .store import TrustStore
from .verifier import TrustVerifier
from .penalizer import TrustPenalizer
from .auditor import TrustAuditor
from .learner import TrustLearner
from .models import (
    TrustRecord,
    VerificationResult,
    PenaltyResult,
    TrustResult,
)

__all__ = [
    "TrustOrchestrator",
    "TrustStore",
    "TrustVerifier",
    "TrustPenalizer",
    "TrustAuditor",
    "TrustLearner",
    "TrustRecord",
    "VerificationResult",
    "PenaltyResult",
    "TrustResult",
]
