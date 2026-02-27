"""
Gate Registry and Quality Gates for the Adaptive Engine.

Provides:
- QualityGate base class
- ThresholdGate (confidence-based validation)
- BallotGate (voting-based validation)
- GateRegistry for managing multiple gates
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class GateDecision(str, Enum):
    """Quality gate decision states."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATE = "escalate"


class GateStage(str, Enum):
    """Gate stages for validation."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"


@dataclass
class GateResult:
    """Result of gate validation."""
    decision: GateDecision
    message: str = ""
    details: dict = field(default_factory=dict)
    confidence: Optional[float] = None


class QualityGate(ABC):
    """Abstract base class for quality gates."""

    @abstractmethod
    def validate(self, context: Any) -> GateResult:
        """Validate the given context.
        
        Args:
            context: Validation context (format depends on gate type)
            
        Returns:
            GateResult with decision and details
        """
        pass

    def is_enabled(self) -> bool:
        """Check if this gate is enabled."""
        return True


class ThresholdGate(QualityGate):
    """
    Threshold-based quality gate.
    
    Validates based on a confidence threshold. Returns APPROVED if confidence
    meets or exceeds threshold, REJECTED otherwise.
    
    Extracted from MiddleManagerGate in middle_manager.py.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        stage: GateStage = GateStage.INTEGRATION,
    ):
        self.threshold = threshold
        self.stage = stage

    def validate(self, context: Any) -> GateResult:
        """Validate based on confidence threshold.
        
        Args:
            context: Dict with 'confidence' key (float 0.0-1.0)
                   or object with confidence attribute
            
        Returns:
            GateResult with APPROVED if threshold met
        """
        # Extract confidence from various context formats
        confidence: Optional[float] = None

        if isinstance(context, dict):
            confidence = context.get("confidence")
        elif hasattr(context, "confidence"):
            confidence = context.confidence

        if confidence is None:
            return GateResult(
                decision=GateDecision.REJECTED,
                message="No confidence score provided",
                details={"context_type": type(context).__name__},
            )

        if confidence >= self.threshold:
            return GateResult(
                decision=GateDecision.APPROVED,
                message=f"Confidence {confidence:.2f} meets threshold {self.threshold}",
                details={"confidence": confidence, "threshold": self.threshold},
                confidence=confidence,
            )
        else:
            return GateResult(
                decision=GateDecision.REJECTED,
                message=f"Confidence {confidence:.2f} below threshold {self.threshold}",
                details={"confidence": confidence, "threshold": self.threshold},
                confidence=confidence,
            )


@dataclass
class BallotVote:
    """A single vote in a ballot."""
    voter: str
    decision: str  # "approved" or "rejected"
    reason: str = ""


@dataclass
class Ballot:
    """A ballot for voting-based validation."""
    ballot_id: str
    stage: GateStage
    votes: List[BallotVote] = field(default_factory=list)
    status: GateDecision = GateDecision.PENDING
    created_at: str = field(
        default_factory=lambda: __import__("datetime").datetime.utcnow().isoformat()
    )
    decided_at: Optional[str] = None


class BallotGate(QualityGate):
    """
    Ballot-based voting quality gate.
    
    Uses a voting mechanism where multiple voters can approve or reject.
    A ballot is approved if it meets the voting threshold.
    
    Extracted from ToolOutputGate in quality_gates.py.
    """

    def __init__(
        self,
        approval_threshold: float = 0.5,  # 50% + 1 for majority
        min_votes: int = 1,
        stage: GateStage = GateStage.INTEGRATION,
    ):
        self.approval_threshold = approval_threshold
        self.min_votes = min_votes
        self.stage = stage
        self._ballots: dict[str, Ballot] = {}

    def create_ballot(self, stage: Optional[GateStage] = None) -> str:
        """Create a new ballot.
        
        Args:
            stage: Gate stage for this ballot
            
        Returns:
            Ballot ID
        """
        ballot_id = f"ballot_{uuid.uuid4().hex[:10]}"
        self._ballots[ballot_id] = Ballot(
            ballot_id=ballot_id,
            stage=stage or self.stage,
        )
        return ballot_id

    def cast_vote(
        self,
        ballot_id: str,
        voter: str,
        decision: str,
        reason: str = "",
    ) -> None:
        """Cast a vote on a ballot.
        
        Args:
            ballot_id: ID of ballot to vote on
            voter: Identifier for the voter
            decision: "approved" or "rejected"
            reason: Optional reason for vote
            
        Raises:
            ValueError: If ballot not found or decision invalid
        """
        if ballot_id not in self._ballots:
            raise ValueError(f"Ballot {ballot_id} not found")

        if decision not in ("approved", "rejected"):
            raise ValueError("Decision must be 'approved' or 'rejected'")

        ballot = self._ballots[ballot_id]
        ballot.votes.append(BallotVote(voter=voter, decision=decision, reason=reason))

        # Update ballot status
        self._update_ballot_status(ballot)

    def _update_ballot_status(self, ballot: Ballot) -> None:
        """Update ballot status based on votes."""
        if not ballot.votes:
            ballot.status = GateDecision.PENDING
            return

        approved = sum(1 for v in ballot.votes if v.decision == "approved")
        total = len(ballot.votes)
        approval_ratio = approved / total if total > 0 else 0

        if total >= self.min_votes and approval_ratio >= self.approval_threshold:
            ballot.status = GateDecision.APPROVED
        elif total >= self.min_votes:
            ballot.status = GateDecision.REJECTED
        else:
            ballot.status = GateDecision.PENDING

    def get_ballot_status(self, ballot_id: str) -> GateDecision:
        """Get current status of a ballot."""
        ballot = self._ballots.get(ballot_id)
        return ballot.status if ballot else GateDecision.PENDING

    def validate(self, context: Any) -> GateResult:
        """Validate using ballot voting.
        
        For Phase 1, this creates a ballot and returns PENDING.
        The actual voting happens through cast_vote().
        
        Args:
            context: Dict with 'is_valid' key or object with is_valid attribute
            
        Returns:
            GateResult with PENDING (voting in progress) or final decision
        """
        # Extract validation result
        is_valid = False
        if isinstance(context, dict):
            is_valid = context.get("is_valid", False)
        elif hasattr(context, "is_valid"):
            is_valid = context.is_valid

        # For Phase 1, auto-vote based on validation result
        # In later phases, this will be more sophisticated
        ballot_id = self.create_ballot()

        if is_valid:
            self.cast_vote(ballot_id, "auto", "approved", "deterministic_validation_passed")
        else:
            self.cast_vote(ballot_id, "auto", "rejected", "validation_failed")

        status = self.get_ballot_status(ballot_id)

        return GateResult(
            decision=status,
            message=f"Ballot {ballot_id}: {status.value}",
            details={"ballot_id": ballot_id, "vote_count": len(self._ballots[ballot_id].votes)},
        )

    def get_ballots(self) -> List[Ballot]:
        """Get all ballots."""
        return list(self._ballots.values())

    def list_pending(self) -> List[Ballot]:
        """Get all pending ballots."""
        return [b for b in self._ballots.values() if b.status == GateDecision.PENDING]


class GateRegistry:
    """
    Registry for managing multiple quality gates.
    
    Provides a unified interface to run multiple gates in sequence
    and aggregate results.
    """

    def __init__(self):
        self._gates: list[QualityGate] = []

    def add_gate(self, gate: QualityGate) -> None:
        """Add a gate to the registry."""
        self._gates.append(gate)

    def remove_gate(self, gate: QualityGate) -> None:
        """Remove a gate from the registry."""
        self._gates.remove(gate)

    def run_all(self, context: Any) -> List[GateResult]:
        """Run all gates in sequence.
        
        Args:
            context: Validation context passed to each gate
            
        Returns:
            List of GateResult for each gate
        """
        results = []
        for gate in self._gates:
            result = gate.validate(context)
            results.append(result)
        return results

    def get_gates(self) -> List[QualityGate]:
        """Get all registered gates."""
        return self._gates.copy()

    def clear(self) -> None:
        """Remove all gates from the registry."""
        self._gates.clear()


__all__ = [
    "GateDecision",
    "GateStage",
    "GateResult",
    "QualityGate",
    "ThresholdGate",
    "BallotGate",
    "Ballot",
    "BallotVote",
    "GateRegistry",
]
