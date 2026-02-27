"""Self-Improvement Engine - Governance Integration Module.

Handles submission of improvement suggestions to ToolOutputGate
and polling for approval decisions.
"""

import asyncio
import time
from typing import Optional

from scout.tool_output import ToolOutput
from scout.validation_pipeline import (
    ValidationResult,
    ValidationStage,
    ValidationError,
)
from scout.validation_pipeline import ValidationErrorCode
from scout.quality_gates import ToolOutputGate, GateDecision


class SuggestionValidator:
    """Validates improvement suggestions before submission to governance.
    
    Ensures suggestions have required fields and valid actions.
    """
    
    def validate(self, suggestion_output: ToolOutput) -> ValidationResult:
        """Run validation on suggestion ToolOutput."""
        errors: list[ValidationError] = []
        
        # Check required metadata fields
        metadata = suggestion_output.metadata or {}
        required_fields = ["recommendation_type", "reason", "evidence"]
        
        for field in required_fields:
            if field not in metadata:
                errors.append(ValidationError(
                    code=ValidationErrorCode.SCHEMA,
                    message=f"Missing required metadata field: {field}",
                    validator="SuggestionValidator"
                ))
        
        # Validate action is allowed
        action = metadata.get("recommendation_type")
        allowed_actions = {"update_prompt", "add_validator", "adjust_metadata", "escalate_to_human"}
        if action and action not in allowed_actions:
            errors.append(ValidationError(
                code=ValidationErrorCode.SCHEMA,
                message=f"Invalid action: {action}. Allowed: {allowed_actions}",
                validator="SuggestionValidator"
            ))
        
        # Check content is not empty
        if not suggestion_output.content or len(suggestion_output.content) < 10:
            errors.append(ValidationError(
                code=ValidationErrorCode.CONTENT_TYPE,
                message="Suggestion content too short",
                validator="SuggestionValidator"
            ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            stage_reached=ValidationStage.DETERMINISTIC,
            errors=errors,
            warnings=[],
            duration_ms=0.0
        )


async def submit_for_approval(recommendation) -> tuple[str, ValidationResult]:
    """Submit improvement suggestion to ToolOutputGate after validation.
    
    Returns (ballot_id, validation_result). If validation fails,
    returns ("", validation_result) with errors.
    """
    gate = ToolOutputGate()
    validator = SuggestionValidator()
    
    # Create ToolOutput representing the suggestion
    output = ToolOutput(
        tool_name=f"self_improvement:{recommendation.tool_name}",
        content=recommendation.suggestion,
        metadata={
            "recommendation_type": recommendation.action,
            "reason": recommendation.reason,
            "evidence": recommendation.evidence,
            "auto_generated": True
        },
        validation_errors=[]
    )
    
    # Validate suggestion BEFORE submission
    validation_result = validator.validate(output)
    
    # If validation fails, log but don't submit
    if not validation_result.is_valid:
        return ("", validation_result)
    
    # Submit to gate with validation result
    ballot_id = gate.submit_for_vote(output, validation_result, submitter="self_improvement")
    
    return (ballot_id, validation_result)


async def wait_for_decision(ballot_id: str, timeout_seconds: int = 3600) -> bool:
    """Poll for gate decision with timeout.
    
    Raises TimeoutError if timeout exceeded.
    """
    if not ballot_id:
        return False
        
    gate = ToolOutputGate()
    start = time.time()
    
    while time.time() - start < timeout_seconds:
        status = gate.get_ballot_status(ballot_id)
        if status in [GateDecision.APPROVED, GateDecision.REJECTED]:
            return status == GateDecision.APPROVED
        await asyncio.sleep(30)
    
    raise TimeoutError(f"Ballot {ballot_id} timed out waiting for decision")
