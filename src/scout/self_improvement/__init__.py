"""Self-Improvement Module.

Analyzes validation data, generates recommendations, and coordinates
with governance for tool improvement.
"""

from scout.self_improvement.analyzer import (
    ImprovementAnalysis,
    ToolAnalysis,
    analyze_improvements,
    analyze_tool_validation,
)
from scout.self_improvement.recommender import (
    ImprovementRecommendation,
    generate_combined_recommendations,
    generate_improvement_recommendation,
    generate_recommendation,
)
from scout.self_improvement.engine import (
    SuggestionValidator,
    submit_for_approval,
    wait_for_decision,
)
from scout.self_improvement.applier import apply_improvement
from scout.self_improvement.improvement_tracker import (
    get_improvement_stats,
    get_high_success_plan_types,
    get_low_success_plan_types,
    record_improvement_outcome,
    record_suggestion_accepted,
)


__all__ = [
    "ToolAnalysis",
    "ImprovementAnalysis",
    "analyze_tool_validation",
    "analyze_improvements",
    "ImprovementRecommendation",
    "generate_recommendation",
    "generate_improvement_recommendation",
    "generate_combined_recommendations",
    "SuggestionValidator",
    "submit_for_approval",
    "wait_for_decision",
    "apply_improvement",
    "record_improvement_outcome",
    "record_suggestion_accepted",
    "get_improvement_stats",
    "get_high_success_plan_types",
    "get_low_success_plan_types",
]
