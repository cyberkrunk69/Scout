"""Self-Improvement Analyzer - Data Collection & Analysis Module.

Analyzes historical validation data from ToolOutputRegistry to identify
failure patterns in tools.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from scout.tool_output import ToolOutput, get_registry


@dataclass
class ToolAnalysis:
    """Analysis result for a tool's validation performance over a period."""
    tool_name: str
    period_days: int
    total_runs: int
    failure_rate: float
    error_categories: dict[str, int]  # e.g., {"SCHEMA": 5, "PATH": 2}
    outputs: List[ToolOutput]


@dataclass
class ImprovementAnalysis:
    """Analysis result for improvement outcomes over a period."""
    period_days: int
    total_improvements: int
    success_rate: float
    by_plan_type: dict[str, dict[str, float]]
    high_success_types: List[str]
    low_success_types: List[str]
    recent_outcomes: List[dict]


def analyze_tool_validation(tool_name: str, days: int = 7) -> ToolAnalysis:
    """Query ToolOutputRegistry for outputs in time window.
    
    Computes metrics: total_runs, failure_rate, error_categories.
    
    Uses structured error codes when available, falls back to parsing
    validation_errors strings if needed.
    """
    registry = get_registry()
    outputs = []
    
    cutoff = datetime.now() - timedelta(days=days)
    for output_id in registry.list():
        output = registry.load(output_id)
        if output and output.tool_name == tool_name:
            created = datetime.fromisoformat(output.created_at)
            if created >= cutoff:
                outputs.append(output)
    
    total = len(outputs)
    failures = [o for o in outputs if o.validation_errors]
    failure_rate = len(failures) / total if total > 0 else 0
    
    # Categorize errors - use structured codes when available
    error_counts: dict[str, int] = {}
    for f in failures:
        for err in f.validation_errors:
            # Try to extract [CODE] prefix, default to UNKNOWN if malformed
            if ']' in err:
                code = err.split(']')[0].replace('[', '')
            else:
                # Fallback: mark as UNKNOWN if no code prefix
                code = "UNKNOWN"
            error_counts[code] = error_counts.get(code, 0) + 1
    
    return ToolAnalysis(
        tool_name=tool_name,
        period_days=days,
        total_runs=total,
        failure_rate=failure_rate,
        error_categories=error_counts,
        outputs=outputs
    )


def analyze_improvements(days: int = 7) -> ImprovementAnalysis:
    """Analyze improvement outcomes from the tracker.

    Uses improvement_tracker.get_improvement_stats() to gather data
    and structures it for the recommender.

    Args:
        days: Number of days to look back (default 7)

    Returns:
        ImprovementAnalysis with metrics and categorized plan types
    """
    # Import here to avoid circular imports
    from scout.self_improvement.improvement_tracker import (
        get_high_success_plan_types,
        get_improvement_stats,
        get_low_success_plan_types,
    )

    stats = get_improvement_stats(days=days)

    high_success = get_high_success_plan_types(min_success_rate=70.0)
    low_success = get_low_success_plan_types(max_success_rate=50.0)

    return ImprovementAnalysis(
        period_days=days,
        total_improvements=stats["total_improvements"],
        success_rate=stats["success_rate"],
        by_plan_type=stats["by_plan_type"],
        high_success_types=high_success,
        low_success_types=low_success,
        recent_outcomes=stats["recent_outcomes"],
    )
