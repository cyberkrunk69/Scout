"""Improvement Tracker - Records improvement outcomes for self-improvement analysis.

This module provides:
- record_improvement_outcome(): Store improvement results in ToolOutputRegistry
- get_improvement_stats(): Query improvement success rates, common plan types, etc.

Integrates with the audit system to track improvement suggestions, acceptance, and outcomes.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from scout.audit import AuditLog
from scout.tool_output import ToolOutput, get_registry


@dataclass
class ImprovementOutcome:
    """Structured outcome for an improvement application.

    Fields:
        outcome_id: Unique identifier
        file: File that was improved
        plan_type: Type of improvement plan (e.g., "refactor", "optimize", "fix")
        success: Whether the improvement was successfully applied
        confidence: Confidence score from the improvement plan (0.0-1.0)
        error: Error message if application failed
        created_at: Timestamp of outcome recording
    """

    outcome_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file: str = ""
    plan_type: str = ""
    success: bool = False
    confidence: Optional[float] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        result = asdict(self)
        result["created_at"] = self.created_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImprovementOutcome":
        """Deserialize from dictionary."""
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


def record_improvement_outcome(
    file: str,
    plan_type: str,
    success: bool,
    confidence: Optional[float] = None,
    error: Optional[str] = None,
) -> ImprovementOutcome:
    """Record an improvement outcome to the ToolOutputRegistry.

    Also logs to the AuditLog for real-time tracking.

    Args:
        file: Path to the file that was improved
        plan_type: Type of improvement (e.g., "refactor", "optimize", "fix")
        success: Whether the improvement was successfully applied
        confidence: Optional confidence score from the plan
        error: Optional error message if application failed

    Returns:
        The recorded ImprovementOutcome
    """
    outcome = ImprovementOutcome(
        file=file,
        plan_type=plan_type,
        success=success,
        confidence=confidence,
        error=error,
    )

    # Store in ToolOutputRegistry for later analysis
    registry = get_registry()
    tool_output = ToolOutput(
        tool_name="improvement_tracker",
        content=outcome.to_dict(),
        cost_usd=0.0,
        metadata={
            "outcome_type": "improvement_outcome",
            "file": file,
            "plan_type": plan_type,
            "success": success,
        },
    )
    registry.save(tool_output)

    # Also log to AuditLog for real-time monitoring
    audit = AuditLog()
    audit.log(
        "improvement_outcome",
        files=[file] if file else None,
        confidence=int(confidence * 100) if confidence is not None else None,
        reason=plan_type,
        success=success,
        error=error,
    )

    # Log successful application separately
    if success:
        audit.log(
            "improvement_applied",
            files=[file] if file else None,
            confidence=int(confidence * 100) if confidence is not None else None,
            reason=plan_type,
        )

    return outcome


def record_suggestion_accepted(
    file: str,
    plan_type: str,
    confidence: Optional[float] = None,
) -> None:
    """Record when a user accepts an improvement suggestion.

    Args:
        file: Path to the file
        plan_type: Type of improvement
        confidence: Confidence score from the suggestion
    """
    audit = AuditLog()
    audit.log(
        "improvement_suggestion_accepted",
        files=[file] if file else None,
        confidence=int(confidence * 100) if confidence is not None else None,
        reason=plan_type,
    )


def get_improvement_stats(days: int = 7) -> Dict[str, Any]:
    """Query improvement outcomes for a given time period.

    Args:
        days: Number of days to look back (default 7)

    Returns:
        Dictionary with:
        - total_improvements: Total number of improvements in period
        - success_count: Number of successful improvements
        - failure_count: Number of failed improvements
        - success_rate: Percentage of successful improvements
        - by_plan_type: Dict mapping plan_type to success rate and count
        - recent_outcomes: List of recent outcome dicts
    """
    registry = get_registry()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    outcomes: List[ImprovementOutcome] = []
    for output_id in registry.list():
        output = registry.load(output_id)
        if output and output.tool_name == "improvement_tracker":
            metadata = output.metadata or {}
            if metadata.get("outcome_type") == "improvement_outcome":
                # Reconstruct from content
                if isinstance(output.content, dict):
                    outcome = ImprovementOutcome.from_dict(output.content)
                    if outcome.created_at >= cutoff:
                        outcomes.append(outcome)

    # Calculate statistics
    total = len(outcomes)
    if total == 0:
        return {
            "total_improvements": 0,
            "success_count": 0,
            "failure_count": 0,
            "success_rate": 0.0,
            "by_plan_type": {},
            "recent_outcomes": [],
        }

    success_count = sum(1 for o in outcomes if o.success)
    failure_count = total - success_count
    success_rate = (success_count / total) * 100 if total > 0 else 0.0

    # Group by plan_type
    by_plan_type: Dict[str, Dict[str, Any]] = {}
    for outcome in outcomes:
        pt = outcome.plan_type or "unknown"
        if pt not in by_plan_type:
            by_plan_type[pt] = {"total": 0, "success": 0, "success_rate": 0.0}
        by_plan_type[pt]["total"] += 1
        if outcome.success:
            by_plan_type[pt]["success"] += 1

    # Calculate per-type success rates
    for pt, stats in by_plan_type.items():
        stats["success_rate"] = (
            (stats["success"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        )

    # Recent outcomes (last 10)
    outcomes.sort(key=lambda x: x.created_at, reverse=True)
    recent_outcomes = [o.to_dict() for o in outcomes[:10]]

    return {
        "total_improvements": total,
        "success_count": success_count,
        "failure_count": failure_count,
        "success_rate": round(success_rate, 2),
        "by_plan_type": by_plan_type,
        "recent_outcomes": recent_outcomes,
    }


def get_high_success_plan_types(min_success_rate: float = 70.0) -> List[str]:
    """Get plan types with high success rates.

    Args:
        min_success_rate: Minimum success rate percentage (default 70%)

    Returns:
        List of plan types that meet the threshold
    """
    stats = get_improvement_stats()
    high_success = [
        pt
        for pt, data in stats.get("by_plan_type", {}).items()
        if data["success_rate"] >= min_success_rate
    ]
    return high_success


def get_low_success_plan_types(max_success_rate: float = 50.0) -> List[str]:
    """Get plan types with low success rates that may need attention.

    Args:
        max_success_rate: Maximum success rate percentage (default 50%)

    Returns:
        List of plan types below the threshold
    """
    stats = get_improvement_stats()
    low_success = [
        pt
        for pt, data in stats.get("by_plan_type", {}).items()
        if data["success_rate"] <= max_success_rate and data["total"] >= 2
    ]
    return low_success
