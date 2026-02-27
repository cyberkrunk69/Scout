"""
State management for the Adaptive Engine.

Provides PlanningContext (StateManager) for bidirectional context flow
between parent plans and sub-plans.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Default max depth if not provided
DEFAULT_MAX_DEPTH = 5


# Alias for future migration - StateManager is the preferred name
StateManager = None  # Will be set after PlanningContext is defined


class PlanningContext:
    """
    Bidirectional context flow for dynamic re-synthesis planning.

    This class manages context that flows both downward (parent → sub-plan)
    and upward (sub-plan → parent), enabling adaptive execution.

    Downward flow (parent → sub-plan):
    - summary: High-level summary of parent plan
    - parent_goals: What parent is trying to achieve
    - constraints: Budget, timeline, etc.

    Upward reporting (sub-plan → parent):
    - discoveries: Findings from sub-plans
    - pivots_needed: Pivots triggered by discoveries
    - sub_plan_outcomes: Results of executed sub-plans
    """

    def __init__(
        self,
        request: str,
        depth: int = 0,
        max_depth: Optional[int] = None,
        summary: str = "",
        parent_goals: Optional[list[str]] = None,
        constraints: Optional[list[str]] = None,
    ):
        self.request = request
        self.depth = depth
        # Lazy import to avoid circular dependency
        if max_depth is None:
            try:
                from scout.cli.plan import get_max_depth
                max_depth = get_max_depth()
            except ImportError:
                max_depth = DEFAULT_MAX_DEPTH
        self.max_depth = max_depth
        self.remaining_depth = self.max_depth - self.depth

        # Upward reporting (sub-plan → parent)
        self.discoveries: list[dict] = []
        self.pivots_needed: list[str] = []
        self.sub_plan_outcomes: list[dict] = []

        # Downward flow (parent → sub-plan)
        self.summary = summary
        self.parent_goals = parent_goals or []
        self.constraints = constraints or []

        # State management
        self.is_pivoting = False
        self.pivot_reason: str = ""
        self.replan_required = False
        self.plan_id: Optional[str] = None

    def remaining_depth_for_child(self) -> int:
        """Calculate remaining depth for sub-plan."""
        return max(0, self.remaining_depth - 1)

    def add_discovery(self, discovery: dict) -> None:
        """Add a discovery from sub-plan execution.
        
        Args:
            discovery: Dict with keys: type, detail, findings (optional)
        """
        self.discoveries.append(discovery)
        logger.debug("Added discovery: %s", discovery.get("type", "unknown"))

    def add_sub_plan_outcome(self, outcome: dict) -> None:
        """Record the outcome of a sub-plan execution.
        
        Args:
            outcome: Dict with keys: plan_id, status, result, error (optional)
        """
        self.sub_plan_outcomes.append(outcome)

    def mark_pivot(self, reason: str) -> None:
        """Mark that a pivot is needed.
        
        Args:
            reason: The pivot trigger type
        """
        self.is_pivoting = True
        self.pivot_reason = reason
        self.pivots_needed.append(reason)
        logger.info("Pivot marked: %s", reason)

    def reset(self) -> None:
        """Reset context for reuse (e.g., in a new planning cycle)."""
        self.discoveries = []
        self.pivots_needed = []
        self.sub_plan_outcomes = []
        self.is_pivoting = False
        self.pivot_reason = ""
        self.replan_required = False

    def to_dict(self) -> dict:
        """Serialize context to dict for logging/persistence."""
        return {
            "request": self.request,
            "depth": self.depth,
            "max_depth": self.max_depth,
            "remaining_depth": self.remaining_depth,
            "summary": self.summary,
            "parent_goals": self.parent_goals,
            "constraints": self.constraints,
            "discoveries": self.discoveries,
            "pivots_needed": self.pivots_needed,
            "sub_plan_outcomes": self.sub_plan_outcomes,
            "is_pivoting": self.is_pivoting,
            "pivot_reason": self.pivot_reason,
            "replan_required": self.replan_required,
            "plan_id": self.plan_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PlanningContext:
        """Deserialize context from dict."""
        ctx = cls(
            request=data.get("request", ""),
            depth=data.get("depth", 0),
            max_depth=data.get("max_depth"),
            summary=data.get("summary", ""),
            parent_goals=data.get("parent_goals"),
            constraints=data.get("constraints"),
        )
        ctx.discoveries = data.get("discoveries", [])
        ctx.pivots_needed = data.get("pivots_needed", [])
        ctx.sub_plan_outcomes = data.get("sub_plan_outcomes", [])
        ctx.is_pivoting = data.get("is_pivoting", False)
        ctx.pivot_reason = data.get("pivot_reason", "")
        ctx.replan_required = data.get("replan_required", False)
        ctx.plan_id = data.get("plan_id")
        return ctx


# Set up the alias - StateManager is an alias for PlanningContext
# This allows gradual migration to the clearer name
StateManager = PlanningContext


__all__ = [
    "PlanningContext",
    "StateManager",  # Alias for PlanningContext
]
