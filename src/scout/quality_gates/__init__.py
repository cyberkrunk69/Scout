"""
ToolOutputGate - Quality Gate integration for ToolOutput.

Wraps QualityGateManager to validate tool outputs through staged gates:
- UNIT: Deterministic validators (schema, content type, paths)
- INTEGRATION: Heuristic validators (length, confidence, duplicates)
- E2E: Optional LLM-based validation

Phase 2 of Unified Tool Framework.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from scout.audit import AuditLog
from scout.config import ScoutConfig
from scout.tool_output import ToolOutput
from scout.validation_pipeline import (
    ValidationPipeline,
    ValidationResult,
    ValidationStage,
)
from .runtime import QualityGateManager

# Import Adaptive Engine components
from scout.adaptive_engine.gates import BallotGate, GateDecision, GateStage


@dataclass
class ToolOutputBallot:
    """Ballot for tool output quality voting."""
    ballot_id: str
    output_id: str
    tool_name: str
    stage: GateStage
    decision: GateDecision = GateDecision.PENDING
    votes: List[Dict[str, Any]] = field(default_factory=list)
    submitted_at: str = field(default_factory=lambda: __import__("datetime").datetime.utcnow().isoformat())
    decided_at: Optional[str] = None


class ToolOutputGate:
    """
    Wrapper around QualityGateManager for tool outputs.

    Translates ToolOutput validation into the blind-vote system.
    Stages:
    - UNIT: OutputSchemaValidator, ContentTypeValidator, PathReferenceValidator
    - INTEGRATION: LengthValidator, ConfidenceThresholdValidator, DuplicateOutputValidator
    - E2E: LLM-based validation (optional, not implemented in Phase 2)
    """

    def __init__(
        self,
        workspace: Optional[Path] = None,
        config: Optional[ScoutConfig] = None,
        audit: Optional[AuditLog] = None,
    ):
        self.workspace = workspace or Path.cwd()
        self.config = config or ScoutConfig()
        self.audit = audit or AuditLog()
        self._manager = QualityGateManager(workspace=self.workspace)

        # Load or initialize tool output state
        state = self._manager.load_state()
        if "tool_outputs" not in state:
            state["tool_outputs"] = {}
        if "output_ballots" not in state:
            state["output_ballots"] = {}

    def is_enabled(self) -> bool:
        """Check if quality gates are enabled in config."""
        return self.config.get_quality_gates_config().get("enabled", False)

    def get_enabled_stages(self) -> List[GateStage]:
        """Get list of enabled gate stages."""
        stages = self.config.get_quality_gates_config().get("stages", ["unit", "integration"])
        return [GateStage(s) for s in stages]

    def submit_for_vote(
        self,
        output: ToolOutput,
        validation_result: ValidationResult,
        submitter: str = "auto",
    ) -> str:
        """
        Submit ToolOutput to quality gates.

        Creates a ballot and triggers voting based on validation result.
        Returns ballot_id.
        """
        if not self.is_enabled():
            raise ValueError("Quality gates are not enabled")

        ballot_id = f"out_{uuid.uuid4().hex[:10]}"

        # Determine initial stage based on validation result
        if validation_result.stage_reached == ValidationStage.DETERMINISTIC:
            stage = GateStage.UNIT
        elif validation_result.stage_reached == ValidationStage.HEURISTIC:
            stage = GateStage.INTEGRATION
        else:
            stage = GateStage.E2E

        # Create ballot
        ballot = ToolOutputBallot(
            ballot_id=ballot_id,
            output_id=output.output_id,
            tool_name=output.tool_name,
            stage=stage,
        )

        # Store in state
        state = self._manager.load_state()
        state["output_ballots"][ballot_id] = {
            "ballot_id": ballot_id,
            "output_id": output.output_id,
            "tool_name": output.tool_name,
            "stage": stage.value,
            "decision": GateDecision.PENDING.value,
            "votes": [],
            "submitted_at": ballot.submitted_at,
            "validation": {
                "is_valid": validation_result.is_valid,
                "error_count": len(validation_result.errors),
                "warning_count": len(validation_result.warnings),
                "stage_reached": validation_result.stage_reached.value,
            },
        }
        self._manager.save_state(state)

        # Log audit event
        self.audit.log(
            "tool_output_submitted",
            tool_name=output.tool_name,
            output_id=output.output_id,
            ballot_id=ballot_id,
            stage=stage.value,
            validation_valid=validation_result.is_valid,
        )

        # Auto-vote based on validation result (deterministic for Phase 2)
        if validation_result.is_valid:
            self.record_vote(ballot_id, "approved", reason="deterministic_validation_passed")
        else:
            self.record_vote(ballot_id, "rejected", reason="validation_failed")

        return ballot_id

    def record_vote(
        self,
        ballot_id: str,
        decision: str,
        reason: Optional[str] = None,
    ) -> None:
        """
        Record a vote on a ballot.

        decision: "approved" or "rejected"
        """
        if decision not in ("approved", "rejected"):
            raise ValueError("Decision must be 'approved' or 'rejected'")

        state = self._manager.load_state()
        ballot = state["output_ballots"].get(ballot_id)

        if not ballot:
            raise ValueError(f"Ballot {ballot_id} not found")

        # Record vote
        vote = {
            "voter": "auto",  # In Phase 2, validators auto-vote
            "decision": decision,
            "reason": reason or "",
        }
        ballot["votes"].append(vote)

        # Update decision
        ballot["decision"] = decision
        ballot["decided_at"] = __import__("datetime").datetime.utcnow().isoformat()

        self._manager.save_state(state)

        # Log audit event
        self.audit.log(
            "tool_output_voted",
            ballot_id=ballot_id,
            decision=decision,
            reason=reason,
        )

    def get_ballot_status(self, ballot_id: str) -> GateDecision:
        """Check current status of a ballot."""
        state = self._manager.load_state()
        ballot = state["output_ballots"].get(ballot_id)

        if not ballot:
            return GateDecision.PENDING

        return GateDecision(ballot["decision"])

    def is_approved(self, output_id: str) -> bool:
        """Check if output has passed all configured gates."""
        state = self._manager.load_state()

        # Find all ballots for this output
        ballots = [
            b for b in state["output_ballots"].values()
            if b["output_id"] == output_id
        ]

        if not ballots:
            return False

        # Check if all enabled stages have approved
        enabled_stages = self.get_enabled_stages()
        for stage in enabled_stages:
            stage_ballots = [b for b in ballots if b["stage"] == stage.value]
            if not stage_ballots:
                return False
            if not all(b["decision"] == "approved" for b in stage_ballots):
                return False

        return True

    def list_pending(self) -> List[Dict[str, Any]]:
        """List outputs awaiting vote."""
        state = self._manager.load_state()
        pending = [
            b for b in state["output_ballots"].values()
            if b["decision"] == GateDecision.PENDING.value
        ]
        return pending

    def get_output_ballots(self, output_id: str) -> List[Dict[str, Any]]:
        """Get all ballots for a specific output."""
        state = self._manager.load_state()
        return [
            b for b in state["output_ballots"].values()
            if b["output_id"] == output_id
        ]


# === Convenience Functions ===

def gate_output(
    output: ToolOutput,
    validation_result: ValidationResult,
    config: Optional[ScoutConfig] = None,
) -> bool:
    """
    Convenience function to gate a tool output.

    Returns True if approved, False if rejected.
    Raises ValueError if gates not enabled.
    """
    gate = ToolOutputGate(config=config)

    if not gate.is_enabled():
        raise ValueError("Quality gates are not enabled")

    # Submit for vote
    ballot_id = gate.submit_for_vote(output, validation_result)

    # Check result
    return gate.get_ballot_status(ballot_id) == GateDecision.APPROVED
