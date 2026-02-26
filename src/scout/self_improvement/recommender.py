"""Self-Improvement Recommender - Rule-Based Recommendation Engine.

Generates improvement recommendations based on error pattern analysis.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

from scout.self_improvement.analyzer import ImprovementAnalysis, ToolAnalysis


ALLOWED_ACTIONS = Literal["update_prompt", "add_validator", "adjust_metadata", "escalate_to_human", "focus_plan_type", "improve_plan_generator"]


@dataclass
class ImprovementRecommendation:
    """Recommendation for improving a tool based on validation analysis."""
    tool_name: str
    action: ALLOWED_ACTIONS
    reason: str
    suggestion: str
    evidence: dict


FAILURE_RATE_THRESHOLD = 0.15  # 15% triggers analysis
MIN_FAILURES_FOR_CATEGORY = 2  # Need at least 2 to establish pattern


RULES = {
    "SCHEMA": {
        "action": "update_prompt",
        "reason": "Schema validation failures indicate output format issues",
        "suggestion": "Add explicit JSON schema example to system prompt"
    },
    "PATH": {
        "action": "update_prompt", 
        "reason": "Path reference failures indicate missing file verification",
        "suggestion": "Add instruction to verify file existence before referencing"
    },
    "SYMBOL": {
        "action": "add_validator",
        "reason": "Symbol reference failures indicate function/class hallucinations",
        "suggestion": "Add SymbolReferenceValidator to tool metadata"
    },
    "CONFIDENCE": {
        "action": "adjust_metadata",
        "reason": "Low confidence outputs",
        "suggestion": "Lower confidence threshold or switch to higher-quality model"
    },
    "UNKNOWN": {
        "action": "escalate_to_human",
        "reason": "Unknown error patterns require manual investigation",
        "suggestion": "Review raw validation errors for patterns"
    }
}


def generate_recommendation(analysis: ToolAnalysis) -> Optional[ImprovementRecommendation]:
    """Rule-based recommendation engine for scout_plan.
    
    Returns None if no valid recommendation can be made.
    """
    # Threshold check
    if analysis.failure_rate < FAILURE_RATE_THRESHOLD:
        return None
    
    # Filter categories to those with minimum count
    significant_errors = {
        k: v for k, v in analysis.error_categories.items() 
        if v >= MIN_FAILURES_FOR_CATEGORY
    }
    
    if not significant_errors:
        return None
    
    # Find dominant error category
    dominant_error = max(significant_errors.items(), key=lambda x: x[1])
    
    rule = RULES.get(dominant_error[0], RULES["UNKNOWN"])
    
    return ImprovementRecommendation(
        tool_name=analysis.tool_name,
        action=rule["action"],
        reason=rule["reason"],
        suggestion=rule["suggestion"],
        evidence={
            "failure_rate": analysis.failure_rate,
            "error_counts": significant_errors,
            "dominant_error": dominant_error,
            "total_failures": sum(significant_errors.values())
        }
    )


# Improvement outcome-based rules
IMPROVEMENT_RULES = {
    "high_success": {
        "action": "focus_plan_type",
        "reason": "High success rate indicates effective plan type",
        "suggestion": "Prioritize this plan type in future improvements"
    },
    "low_success": {
        "action": "improve_plan_generator",
        "reason": "Low success rate indicates plan generator needs improvement",
        "suggestion": "Review and improve the plan generation logic for this type"
    },
    "no_data": {
        "action": "focus_plan_type",
        "reason": "No improvement data yet",
        "suggestion": "Apply more improvements to gather data for analysis"
    }
}

MIN_IMPROVEMENTS_FOR_ANALYSIS = 3  # Need at least 3 improvements to analyze
HIGH_SUCCESS_THRESHOLD = 70.0  # 70% success rate
LOW_SUCCESS_THRESHOLD = 50.0  # 50% success rate


def generate_improvement_recommendation(
    analysis: ImprovementAnalysis,
) -> Optional[ImprovementRecommendation]:
    """Generate recommendations based on improvement outcome analysis.

    Returns None if insufficient data or no actionable patterns found.
    """
    # Check for sufficient data
    if analysis.total_improvements < MIN_IMPROVEMENTS_FOR_ANALYSIS:
        return None

    # Check for high success plan types
    if analysis.high_success_types:
        primary_type = analysis.high_success_types[0]
        stats = analysis.by_plan_type.get(primary_type, {})
        return ImprovementRecommendation(
            tool_name="improvement_engine",
            action="focus_plan_type",
            reason=f"High success rate ({stats.get('success_rate', 0):.1f}%) for plan type '{primary_type}'",
            suggestion=f"Prioritize '{primary_type}' improvements - they have high success rate",
            evidence={
                "plan_type": primary_type,
                "success_rate": stats.get("success_rate", 0),
                "total_attempts": stats.get("total", 0),
                "high_success_types": analysis.high_success_types,
            }
        )

    # Check for low success plan types that need attention
    if analysis.low_success_types:
        primary_type = analysis.low_success_types[0]
        stats = analysis.by_plan_type.get(primary_type, {})
        return ImprovementRecommendation(
            tool_name="improvement_engine",
            action="improve_plan_generator",
            reason=f"Low success rate ({stats.get('success_rate', 0):.1f}%) for plan type '{primary_type}'",
            suggestion=f"Improve plan generation for '{primary_type}' - low success rate",
            evidence={
                "plan_type": primary_type,
                "success_rate": stats.get("success_rate", 0),
                "total_attempts": stats.get("total", 0),
                "low_success_types": analysis.low_success_types,
            }
        )

    return None


def generate_combined_recommendations(
    tool_analysis: Optional[ToolAnalysis] = None,
    improvement_analysis: Optional[ImprovementAnalysis] = None,
) -> List[ImprovementRecommendation]:
    """Generate combined recommendations from both validation and improvement analysis.

    Args:
        tool_analysis: Optional ToolAnalysis from validation failures
        improvement_analysis: Optional ImprovementAnalysis from improvement outcomes

    Returns:
        List of recommendations (may be empty)
    """
    recommendations: List[ImprovementRecommendation] = []

    # Add tool validation recommendations
    if tool_analysis:
        tool_rec = generate_recommendation(tool_analysis)
        if tool_rec:
            recommendations.append(tool_rec)

    # Add improvement outcome recommendations
    if improvement_analysis:
        improvement_rec = generate_improvement_recommendation(improvement_analysis)
        if improvement_rec:
            recommendations.append(improvement_rec)

    return recommendations
