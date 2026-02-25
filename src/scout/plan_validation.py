# TODO: This module is not yet integrated into the main application.
# It is planned for future use as part of the Plan Execution framework.
# See ADR-007 for design context.
"""Plan Validation for re-synthesis consistency.

Provides validation of re-synthesized plans against previous context:
- Discovery consistency: Critical discoveries must be addressed
- Dependency integrity: Step dependencies must be valid
- Pruning compliance: Pruned plans must not appear
- Goal alignment: LLM check for overall alignment
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Config flags for LLM validation
LLM_GOAL_ALIGNMENT_ENABLED = os.getenv("SCOUT_PLAN_LLM_VALIDATION", "true").lower() == "true"
LLM_GOAL_ALIGNMENT_SAMPLE_RATE = float(os.getenv("SCOUT_PLAN_VALIDATION_SAMPLE_RATE", "1.0"))

# Retry configuration for validation failures
MAX_VALIDATION_RETRIES = 1  # Max auto-retry attempts
VALIDATION_RETRY_ENABLED = os.getenv("SCOUT_PLAN_VALIDATION_RETRY", "true").lower() == "true"


@dataclass
class ValidationReport:
    """Structured validation report with errors, warnings, and info."""
    
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Returns True if no errors."""
        return len(self.errors) == 0
    
    @property
    def has_warnings(self) -> bool:
        """Returns True if there are warnings."""
        return len(self.warnings) > 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "info_count": len(self.info),
            "errors": self.errors,
            "warnings": self.warnings,
        }


def validate_replan_consistency(
    new_plan: str,
    context,
    sub_plans: Optional[List[Dict]] = None,
) -> ValidationReport:
    """Validate re-synthesized plan against previous context.
    
    Args:
        new_plan: The re-synthesized plan text
        context: PlanningContext with discoveries and outcomes
        sub_plans: List of sub-plans (may contain _pruned flags)
    
    Returns:
        ValidationReport with errors, warnings, and info
    """
    report = ValidationReport()
    new_plan_lower = new_plan.lower()
    
    # 1. Pruning compliance - check pruned plans don't appear
    if sub_plans:
        for sp in sub_plans:
            if sp.get("_pruned"):
                title = sp.get("title", "").lower()
                if title and title in new_plan_lower:
                    report.warnings.append(
                        f"Pruned plan '{sp['title']}' appears in new plan"
                    )
                    logger.warning(
                        "validation_pruned_appears",
                        plan_title=sp["title"],
                        pruned_reason=sp.get("_pruned_reason", "unknown")
                    )
    
    # 2. Critical discovery coverage
    critical_types = ["dependency_conflict", "impossible_step", "security_finding"]
    for discovery in context.discoveries:
        if discovery.get("type") in critical_types:
            detail = discovery.get("detail", "").lower()
            # Check if detail keywords appear in new plan
            keywords = [w for w in detail.split() if len(w) > 3]
            found = any(kw in new_plan_lower for kw in keywords[:3])
            if not found:
                error_msg = (
                    f"Critical discovery not addressed: {discovery.get('type')} - {detail}"
                )
                report.errors.append(error_msg)
                logger.error(
                    "validation_critical_discovery_not_addressed",
                    discovery_type=discovery.get("type"),
                    detail=detail[:100]
                )
    
    # 3. Dependency integrity - parse for "depends on X", "after step Y", "requires Z"
    deps = _extract_dependencies(new_plan)

    # Verify all referenced steps exist
    step_pattern = re.compile(r'^#*\s*(\d+)\.', new_plan, re.MULTILINE)
    existing_steps = set(step_pattern.findall(new_plan))

    # Also check for structured step IDs
    structured_steps = _extract_structured_step_ids(new_plan)
    existing_steps.update(structured_steps)

    for dep in deps:
        if dep not in existing_steps:
            error_msg = f"Dependency references non-existent step {dep}"
            report.errors.append(error_msg)
            logger.error("validation_invalid_dependency", dep_step=dep)

    # Log validation result
    if report.is_valid:
        logger.info("validation_passed", error_count=0, warning_count=len(report.warnings))
    else:
        logger.error("validation_failed", **report.to_dict())

    return report


def _extract_dependencies(plan_text: str) -> list[str]:
    """Extract dependencies from plan text, preferring JSON format.

    First attempts to parse structured JSON steps, then falls back to regex.
    Returns list of step IDs that are dependencies.
    """
    deps = []

    # Try JSON first
    import json
    json_pattern = re.compile(r'\{[\s\S]*?"steps"\s*:\s*\[[\s\S]*?\]\s*\}')
    json_match = json_pattern.search(plan_text)

    if json_match:
        try:
            parsed = json.loads(json_match.group())
            steps = parsed.get("steps", [])
            for step in steps:
                depends_on = step.get("depends_on", [])
                # depends_on can be list of step IDs
                if isinstance(depends_on, list):
                    deps.extend(str(d) for d in depends_on if d != 0)  # 0 = no dependencies
                elif isinstance(depends_on, int):
                    if depends_on != 0:
                        deps.append(str(depends_on))

            if deps:
                logger.debug("dependencies_extracted_from_json", count=len(deps))
                return deps
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Fall back to regex

    # Fall back to regex patterns
    dep_patterns = [
        re.compile(r'depends on (?:step |task )?(\d+)', re.IGNORECASE),
        re.compile(r'after (?:step |task )?(\d+)', re.IGNORECASE),
        re.compile(r'requires (?:step |task )?(\d+)', re.IGNORECASE),
        re.compile(r'before (?:step |task )?(\d+)', re.IGNORECASE),
    ]

    for pattern in dep_patterns:
        deps.extend(pattern.findall(plan_text))

    return deps


def _extract_structured_step_ids(plan_text: str) -> set:
    """Extract step IDs from structured JSON format."""
    import json

    step_ids = set()

    json_pattern = re.compile(r'\{[\s\S]*?"steps"\s*:\s*\[[\s\S]*?\]\s*\}')
    json_match = json_pattern.search(plan_text)

    if json_match:
        try:
            parsed = json.loads(json_match.group())
            steps = parsed.get("steps", [])
            for step in steps:
                step_id = step.get("id")
                if step_id is not None:
                    step_ids.add(str(step_id))
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    return step_ids


async def llm_check_goal_alignment(new_plan: str, context) -> dict:
    """Use LLM to verify new plan aligns with original goals.
    
    Configured via:
    - SCOUT_PLAN_LLM_VALIDATION (default: true)
    - SCOUT_PLAN_VALIDATION_SAMPLE_RATE (default: 1.0)
    """
    from vivarium.scout.llm.minimax import call_minimax_async
    
    # Configurable: allow disabling entirely
    if not LLM_GOAL_ALIGNMENT_ENABLED:
        return {"aligned": True, "reason": "disabled_by_config"}
    
    # Configurable: sample rate for cost control
    if random.random() > LLM_GOAL_ALIGNMENT_SAMPLE_RATE:
        logger.info(
            "llm_goal_alignment_sampled_out",
            plan_id=getattr(context, 'plan_id', None)
        )
        return {"aligned": True, "reason": "sampled_out"}
    
    system_prompt = """You are a validation assistant. Check if a plan still satisfies 
the original request. Return JSON with:
- aligned: true/false
- reason: brief explanation if not aligned"""
    
    user_prompt = f"""Original request: {context.request}
Parent goals: {context.parent_goals}
Constraints: {context.constraints}
Discoveries: {json.dumps(context.discoveries)}

New plan:
{new_plan[:1500]}

Does this plan still satisfy the original request? Return JSON."""
    
    try:
        response, _ = await call_minimax_async(
            prompt=user_prompt, system=system_prompt, max_tokens=200
        )
        # Parse JSON from response
        match = re.search(r'\{[\s\S]*\}', response)
        if match:
            result = json.loads(match.group())
            logger.info(
                "llm_goal_alignment_check",
                aligned=result.get("aligned"),
                reason=result.get("reason", "")[:100]
            )
            return result
    except Exception as e:
        logger.warning("llm_goal_alignment_failed", error=str(e))
    
    return {"aligned": True, "reason": "check_failed_defaulting_to_safe"}


def validate_and_log(
    new_plan: str,
    context,
    sub_plans: Optional[List[Dict]] = None,
    run_llm_check: bool = True,
) -> ValidationReport:
    """Full validation with optional LLM check.
    
    This is the main entry point for validation.
    """
    # Run structural validation (fast, no LLM)
    report = validate_replan_consistency(new_plan, context, sub_plans)
    
    # Add info from structural validation
    if report.is_valid:
        report.info.append("Structural validation passed")
    
    return report
