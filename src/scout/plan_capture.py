# TODO: This module is not yet integrated into the main application.
# It is planned for future use as part of the Plan Execution framework.
# See ADR-007 and ADR-008 for design context.
from __future__ import annotations
"""
Plan capture module for automatically storing Scout plan outputs.

Provides hooks to capture plan outputs with full metadata.
"""

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Any

from scout.plan_io import write_plan, generate_plan_id, validate_plan


@dataclass
class ScoutPlanOutput:
    """Structured output from Scout plan generation."""

    request: str
    plan: str
    model: str
    tokens: int
    cost: float
    created_at: str = ""
    structured: bool = False
    steps: Optional[list[dict]] = None
    path_validation: Optional[dict] = None
    tags: list[str] = None
    parent_id: Optional[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()

    @property
    def id(self) -> str:
        return generate_plan_id(self.plan, self.request)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["id"] = self.id
        return data


class ScoutPlanCapture:
    """
    Capture hook for Scout plan outputs.

    Usage:
        capture = ScoutPlanCapture()

        # After plan generation:
        result = capture.capture(
            request="add user auth",
            plan_text="# Plan...\n",
            model="minimax",
            tokens=1234,
            cost=0.05
        )
    """

    def __init__(self, auto_save: bool = True):
        self.auto_save = auto_save
        self._captured: list[ScoutPlanOutput] = []

    def capture(
        self,
        request: str,
        plan_text: str,
        model: str,
        tokens: int,
        cost: float,
        structured: bool = False,
        steps: Optional[list[dict]] = None,
        path_validation: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        parent_id: Optional[str] = None,
    ) -> ScoutPlanOutput:
        """Capture a plan output with metadata."""
        output = ScoutPlanOutput(
            request=request,
            plan=plan_text,
            model=model,
            tokens=tokens,
            cost=cost,
            created_at=datetime.utcnow().isoformat(),
            structured=structured,
            steps=steps,
            path_validation=path_validation,
            tags=tags or [],
            parent_id=parent_id,
        )

        self._captured.append(output)

        if self.auto_save:
            self.save(output)

        return output

    def save(self, output: ScoutPlanOutput) -> bool:
        """Save a plan output to storage."""
        try:
            plan_dict = output.to_dict()
            valid, errors = validate_plan(plan_dict)
            if not valid:
                print(f"Plan validation errors: {errors}")
                return False

            write_plan(plan_dict)
            return True
        except Exception as e:
            print(f"Failed to save plan: {e}")
            return False

    def capture_from_json(
        self, json_str: str, request: str
    ) -> Optional[ScoutPlanOutput]:
        """Capture a plan from JSON output (e.g., from CLI)."""
        try:
            data = json.loads(json_str)

            return self.capture(
                request=request,
                plan_text=data.get("plan", ""),
                model=data.get("model", "minimax"),
                tokens=data.get("tokens", 0),
                cost=data.get("cost", 0.0),
                structured=bool(data.get("steps")),
                steps=data.get("steps"),
                path_validation=data.get("path_validation"),
            )
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            return None

    @property
    def captured_count(self) -> int:
        """Number of plans captured in this session."""
        return len(self._captured)

    def get_captured(self) -> list[ScoutPlanOutput]:
        """Get all captured plans in this session."""
        return self._captured.copy()


def create_capture_hook():
    """Create a global capture instance."""
    return ScoutPlanCapture(auto_save=True)
