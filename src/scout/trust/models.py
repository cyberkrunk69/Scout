"""
Shared dataclasses for trust subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List


@dataclass
class TrustRecord:
    """Database record for a single source file."""
    source_path: str
    doc_path: str
    trust_level: str
    embedded_checksum: Optional[str]
    current_checksum: Optional[str]
    stale_symbols: List[str] = field(default_factory=list)
    fresh_symbols: List[str] = field(default_factory=list)
    penalty: float = 0.0
    last_validated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    query_count: int = 0
    last_queried: Optional[str] = None

    # For Bayesian learning
    success_count: int = 0
    failure_count: int = 0


@dataclass
class VerificationResult:
    source_path: Path
    doc_path: Path
    trust_level: str
    embedded_checksum: Optional[str]
    current_checksum: Optional[str]
    stale_symbols: List[str]
    fresh_symbols: List[str]
    gap_message: Optional[str]


@dataclass
class PenaltyResult:
    base_penalty: float
    dynamic_adjustment: float
    final_penalty: float
    reason: str


@dataclass
class TrustResult:
    source_path: Path
    trust_level: str
    penalty: float
    confidence: int
    gap_message: Optional[str]
    verification: VerificationResult
    penalty_result: PenaltyResult
