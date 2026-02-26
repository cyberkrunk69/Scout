"""
Trigger Registry for the Adaptive Engine.

Provides TriggerRegistry for managing pivot triggers with priorities,
weights, and heuristic keyword matching. This encapsulates the logic
previously in plan.py's PIVOT_TRIGGERS and related functions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default trigger configuration
PIVOT_TRIGGERS: dict = {
    # Priority 1: Critical (always trigger, even at max depth)
    "security_finding": {
        "priority": 1,
        "weight": 10,
        "heuristic_kw": ["security", "vulnerability", "exploit"]
    },
    "impossible_step": {
        "priority": 1,
        "weight": 10,
        "heuristic_kw": ["impossible", "cannot", "can't do"]
    },

    # Priority 2: High
    "dependency_conflict": {
        "priority": 2,
        "weight": 7,
        "heuristic_kw": ["conflict", "contradict", "incompatible"]
    },
    "new_critical_path": {
        "priority": 2,
        "weight": 7,
        "heuristic_kw": ["must have", "required", "essential"]
    },

    # Priority 3: Medium
    "scope_change": {
        "priority": 3,
        "weight": 5,
        "heuristic_kw": ["also need", "additionally", "extended"]
    },
    "performance_constraint": {
        "priority": 3,
        "weight": 5,
        "heuristic_kw": ["slow", "performance", "latency", "timeout"]
    },

    # Priority 4: Low (heuristics only)
    "resource_limit": {
        "priority": 4,
        "weight": 3,
        "heuristic_kw": ["memory", "cpu", "disk", "quota"]
    },
    "api_change": {
        "priority": 4,
        "weight": 3,
        "heuristic_kw": ["api changed", "deprecated", "breaking"]
    },

    # Priority 5: Subtle (LLM detection recommended)
    "user_feedback": {
        "priority": 5,
        "weight": 2,
        "heuristic_kw": []
    },
    "missing_context": {
        "priority": 5,
        "weight": 2,
        "heuristic_kw": ["need more", "unclear", "what about"]
    },
}

# Default configuration constants
DEFAULT_PIVOT_THRESHOLD = 0.3
PIVOT_FEEDBACK_FILE = Path(".scout/pivot_feedback.jsonl")


@dataclass
class TriggerConfig:
    """Configuration for a single trigger."""
    name: str
    priority: int
    weight: int
    heuristic_kw: list[str]


class TriggerRegistry:
    """
    Registry for managing pivot triggers with priority-weighted evaluation.
    
    This class encapsulates trigger management and evaluation logic,
    replacing the static PIVOT_TRIGGERS dict from the original implementation.
    """

    def __init__(self, triggers: Optional[dict[str, dict]] = None):
        """Initialize the registry with triggers.
        
        Args:
            triggers: Dict mapping trigger name to config dict.
                     If None, uses default PIVOT_TRIGGERS.
        """
        self._triggers = triggers or PIVOT_TRIGGERS.copy()
        self._trigger_configs: dict[str, TriggerConfig] = {}
        self._build_configs()

    def _build_configs(self) -> None:
        """Build TriggerConfig objects from trigger dict."""
        for name, config in self._triggers.items():
            self._trigger_configs[name] = TriggerConfig(
                name=name,
                priority=config.get("priority", 99),
                weight=config.get("weight", 1),
                heuristic_kw=config.get("heuristic_kw", []),
            )

    def get_trigger(self, name: str) -> Optional[TriggerConfig]:
        """Get trigger config by name."""
        return self._trigger_configs.get(name)

    def list_triggers(self) -> list[TriggerConfig]:
        """List all triggers sorted by priority."""
        return sorted(
            self._trigger_configs.values(),
            key=lambda c: c.priority
        )

    def should_pivot(self, discoveries: list[dict]) -> bool:
        """Check if any discovery warrants a pivot.
        
        Args:
            discoveries: List of discovery dicts from planning context
            
        Returns:
            True if any discovery matches a trigger
        """
        if not discoveries:
            return False

        for discovery in discoveries:
            discovery_type = discovery.get("type", "")
            # Check against trigger keys
            if discovery_type in self._trigger_configs:
                return True

            # Also check nested findings - check heuristic keywords
            if discovery_type == "step_result":
                findings = discovery.get("findings", [])
                for finding in findings:
                    if isinstance(finding, str):
                        finding_lower = finding.lower()
                        # Check each trigger's keywords against the finding
                        for trigger_name, config in self._trigger_configs.items():
                            for kw in config.heuristic_kw:
                                if kw.lower() in finding_lower:
                                    return True
        return False

    def determine_pivot_reason(self, discoveries: list[dict]) -> str:
        """Return highest-priority (lowest number) trigger type.
        
        Args:
            discoveries: List of discovery dicts
            
        Returns:
            Trigger type with highest priority, or "unknown"
        """
        types_with_priority: list[tuple[int, str, str]] = []

        for d in discoveries:
            d_type = d.get("type", "")
            config = self._trigger_configs.get(d_type)
            if config:
                types_with_priority.append((config.priority, d_type, d.get("detail", "")))

            # Check nested findings
            if d_type == "step_result":
                for f in d.get("findings", []):
                    if isinstance(f, str):
                        for trigger_name, cfg in self._trigger_configs.items():
                            if trigger_name in f.lower():
                                types_with_priority.append(
                                    (cfg.priority, trigger_name, f[:100])
                                )

        # Sort by priority, return highest (lowest number)
        types_with_priority.sort(key=lambda x: x[0])

        if types_with_priority:
            _, dtype, _ = types_with_priority[0]
            return dtype

        return "unknown"

    def compute_heuristic_score(self, sub_plan_results: list[dict]) -> float:
        """Compute heuristic score for pivot likelihood (0.0 to 1.0).
        
        Uses keyword matching from registered triggers.
        Returns low score if subtle triggers may be present.
        
        Args:
            sub_plan_results: List of sub-plan result dicts
            
        Returns:
            Score from 0.0 to 1.0
        """
        total_score = 0.0
        max_possible = 0.0

        for result in sub_plan_results:
            plan_text = (result.get("plan", "") + " " + result.get("summary", "")).lower()

            for trigger_name, config in self._trigger_configs.items():
                keywords = config.heuristic_kw
                weight = config.weight
                max_possible += weight

                for kw in keywords:
                    if kw.lower() in plan_text:
                        total_score += weight
                        break  # Count each trigger once per result

        if max_possible == 0:
            return 0.0

        return min(1.0, total_score / max_possible)

    def is_critical(self, trigger_name: str) -> bool:
        """Check if a trigger is critical (Priority 1).
        
        Args:
            trigger_name: Name of the trigger
            
        Returns:
            True if priority is 1
        """
        config = self._trigger_configs.get(trigger_name)
        return config is not None and config.priority == 1


def create_default_registry() -> TriggerRegistry:
    """Create a TriggerRegistry with default triggers.
    
    This factory function provides backward compatibility
    for code that used the original PIVOT_TRIGGERS dict.
    
    Returns:
        TriggerRegistry instance with default triggers
    """
    return TriggerRegistry(PIVOT_TRIGGERS.copy())


# === Feedback and Threshold Functions (moved from plan.py) ===

def log_pivot_outcome(
    trigger_type: str,
    confirmed: bool,
    plan_id: Optional[str] = None
) -> None:
    """Log pivot trigger outcome for threshold adaptation.

    Records whether an LLM-detected pivot was confirmed or rejected,
    enabling adaptive threshold tuning based on historical precision.

    Args:
        trigger_type: Type of pivot trigger (e.g., 'security_finding')
        confirmed: True if pivot was confirmed by human/action, False if rejected
        plan_id: Optional plan ID for tracking
    """
    PIVOT_FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    from datetime import datetime, timezone

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger_type": trigger_type,
        "confirmed": confirmed,
        "plan_id": plan_id,
    }

    with open(PIVOT_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def compute_optimal_threshold() -> float:
    """Compute pivot threshold that maximizes precision using logged feedback.

    Loads historical feedback from .scout/pivot_feedback.jsonl and calculates
    the threshold that maximizes precision (confirmed/total).

    Returns:
        Optimized threshold (0.0-1.0), or DEFAULT_PIVOT_THRESHOLD if insufficient data
    """
    import os

    # Check for manual override
    manual = os.environ.get("SCOUT_PLAN_PIVOT_THRESHOLD")
    if manual:
        try:
            return float(manual)
        except (ValueError, TypeError):
            pass  # Invalid value, use computed/default

    # Load feedback data
    if not PIVOT_FEEDBACK_FILE.exists():
        return DEFAULT_PIVOT_THRESHOLD

    try:
        feedback_entries = []
        with open(PIVOT_FEEDBACK_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    feedback_entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # Need minimum data points for meaningful threshold computation
        min_samples = 10
        if len(feedback_entries) < min_samples:
            return DEFAULT_PIVOT_THRESHOLD

        # Count confirmed vs total for threshold analysis
        confirmed = sum(1 for e in feedback_entries if e.get("confirmed"))
        total = len(feedback_entries)

        if total > 0:
            precision = confirmed / total
            # Adjust threshold based on precision
            if precision > 0.7:
                return 0.2  # More sensitive
            elif precision < 0.3:
                return 0.5  # Less sensitive
            else:
                return DEFAULT_PIVOT_THRESHOLD

    except Exception as e:
        logger.warning("threshold_computation_failed", error=str(e))

    return DEFAULT_PIVOT_THRESHOLD


__all__ = [
    "TriggerRegistry",
    "TriggerConfig",
    "create_default_registry",
    "PIVOT_TRIGGERS",
    "DEFAULT_PIVOT_THRESHOLD",
    "PIVOT_FEEDBACK_FILE",
    "log_pivot_outcome",
    "compute_optimal_threshold",
]
