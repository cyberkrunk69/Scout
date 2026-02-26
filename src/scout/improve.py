"""
Scout Improve Pipeline – Autonomous code improvement orchestration.

This module implements the "scout-improve" composite pattern:
hotspot → analyze → plan → apply

Supports invocation from CLI (scout improve) or via composite pattern executor.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from scout.analysis.hotspots import scout_hotspots
from scout.audit import AuditLog
from scout.config import DEFAULT_CONFIG, ScoutConfig
from scout.improve_safe.safety import (
    classify_edit,
    get_backup_path,
    get_safety_description,
    is_backup_stale,
)
from scout.self_improvement import improvement_tracker

logger = logging.getLogger(__name__)

# Pipeline step identifiers
STEP_HOTSPOT = "hotspot"
STEP_ANALYZE = "analyze"
STEP_PLAN = "plan"
STEP_APPLY = "apply"


class ImprovementResult:
    """Structured result from the improvement pipeline."""

    def __init__(
        self,
        status: str,
        hotspots: Optional[list[dict[str, Any]]] = None,
        analyses: Optional[list[dict[str, Any]]] = None,
        plans: Optional[list[dict[str, Any]]] = None,
        applied: Optional[list[dict[str, Any]]] = None,
        errors: Optional[list[str]] = None,
        cost: float = 0.0,
    ):
        self.status = status
        self.hotspots = hotspots or []
        self.analyses = analyses or []
        self.plans = plans or []
        self.applied = applied or []
        self.errors = errors or []
        self.cost = cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "hotspots": self.hotspots,
            "analyses": self.analyses,
            "plans": self.plans,
            "applied": self.applied,
            "errors": self.errors,
            "cost": self.cost,
        }


async def run_improve_pipeline(
    goal: str,
    target: Optional[str] = None,
    apply_fixes: bool = False,
    repo_root: Optional[Path] = None,
    dry_run: bool = False,
) -> ImprovementResult:
    """
    Execute the autonomous code improvement pipeline.

    Pipeline steps:
        1. hotspot: Identify high-impact files (git churn, complexity, errors)
        2. analyze: For each hotspot, spawn code analysis agent
        3. plan: Generate improvement plan based on findings
        4. apply: Optionally apply safe fixes

    Args:
        goal: Improvement goal/description (e.g., "reduce complexity", "fix bugs")
        target: Optional specific file/directory to target
        apply_fixes: Whether to automatically apply safe fixes
        repo_root: Repository root path (defaults to cwd)
        dry_run: If True, show diffs without applying changes

    Returns:
        ImprovementResult with pipeline execution details
    """
    audit = AuditLog()
    repo_root = repo_root or Path.cwd()

    logger.info(f"Starting improve pipeline: goal={goal}, target={target}")

    result = ImprovementResult(status="running")

    # Step 1: Identify hotspots
    logger.info("Step 1/4: Identifying hotspots...")
    hotspots = await _identify_hotspots(goal, target, repo_root)
    result.hotspots = hotspots

    if not hotspots:
        result.status = "no_hotspots"
        logger.info("No hotspots identified")
        return result

    # Step 2: Analyze each hotspot (parallel)
    logger.info(f"Step 2/4: Analyzing {len(hotspots)} hotspots...")
    analyses = await _analyze_hotspots_parallel(hotspots, goal, repo_root)
    result.analyses = analyses

    # Step 3: Generate improvement plans
    logger.info(f"Step 3/4: Generating plans for {len(analyses)} analyses...")
    plans = []
    for analysis in analyses:
        plan = await _generate_plan(analysis, goal, repo_root)
        plans.append(plan)
    result.plans = plans

    # Step 4: Optionally apply fixes
    if apply_fixes:
        logger.info(f"Step 4/4: Applying {len(plans)} planned fixes...")
        applied = []
        for plan in plans:
            app_result = await _apply_fix(plan, repo_root, dry_run=dry_run)
            applied.append(app_result)
        result.applied = applied
        result.status = "completed_with_apply"
    else:
        result.status = "completed_analysis_only"

    # Log cost estimate
    estimated_cost = _estimate_cost(result)
    result.cost = estimated_cost
    audit.log(
        "improve_pipeline_complete",
        status=result.status,
        hotspot_count=len(hotspots),
        analysis_count=len(analyses),
        plan_count=len(plans),
        applied_count=len(result.applied),
        cost=estimated_cost,
    )

    logger.info(f"Pipeline complete: status={result.status}, cost=${estimated_cost:.4f}")
    return result


async def _identify_hotspots(
    goal: str,
    target: Optional[str],
    repo_root: Path,
) -> list[dict[str, Any]]:
    """
    Identify high-impact files that would benefit from improvement.

    Uses scout_hotspots() to analyze git churn, error rates, and impact
    to identify files that are frequently changed, error-prone, and affect
    many downstream symbols.
    """
    # If user specified a target, use it directly
    if target:
        return [
            {
                "file": target,
                "reason": "user_specified",
                "impact": "high",
                "score": 1.0,
            }
        ]

    # Use scout_hotspots for real data-driven detection
    try:
        result = scout_hotspots(
            days=30,
            limit=10,
            include_impact=True,
            repo_path=str(repo_root),
        )

        hotspots_raw = result.get("hotspots", [])

        # Transform to pipeline format
        return [
            {
                "file": h["file"],
                "score": h["score"],
                "reason": "hotspot_analysis",
                "impact": "high" if h["score"] > 0.7 else "medium" if h["score"] > 0.4 else "low",
                "churn": h.get("churn", 0),
                "errors": h.get("errors", 0),
                "impact_count": h.get("impact_count", 0),
            }
            for h in hotspots_raw
        ]

    except Exception as e:
        logger.warning(f"Hotspot detection failed: {e}")
        return []


async def _analyze_single_hotspot(
    hotspot: dict[str, Any],
    goal: str,
    repo_root: Path,
) -> dict[str, Any]:
    """
    Analyze a single hotspot file for improvement opportunities.

    Uses scout_roast for code analysis. In full implementation,
    this would spawn a CodeAnalysisAgent (Track Alpha).
    """
    file_path = hotspot.get("file", "")
    full_path = repo_root / file_path if file_path else None

    if not full_path or not full_path.exists():
        return {
            "file": file_path,
            "status": "error",
            "error": "file_not_found",
            "issues": [],
        }

    # Use scout_roast for analysis
    try:
        cmd = [sys.executable, "-m", "vivarium.scout.cli.roast", str(full_path)]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=60.0
        )

        output = stdout.decode() if stdout else ""
        error = stderr.decode() if stderr else ""

        if proc.returncode != 0:
            logger.warning(f"Roast analysis failed for {file_path}: {error}")
            return {
                "file": file_path,
                "status": "partial",
                "issues": [],
                "error": error[:200],
            }

        # Parse roast output for issues
        issues = _parse_roast_output(output)

        return {
            "file": file_path,
            "status": "success",
            "issues": issues,
            "summary": output[:500] if output else "No issues found",
        }

    except asyncio.TimeoutError:
        logger.warning(f"Roast analysis timeout for {file_path}")
        return {
            "file": file_path,
            "status": "timeout",
            "issues": [],
            "error": "Analysis timed out",
        }
    except Exception as e:
        logger.exception(f"Error analyzing {file_path}")
        return {
            "file": file_path,
            "status": "error",
            "issues": [],
            "error": str(e)[:200],
        }


async def _analyze_hotspots_parallel(
    hotspots: list[dict[str, Any]],
    goal: str,
    repo_root: Path,
) -> list[dict[str, Any]]:
    """
    Analyze multiple hotspots in parallel using asyncio.gather with semaphore.

    Uses config for max_agents, agent_timeout_seconds, and max_cost_per_session.
    Falls back to sequential if parallel fails or hotspots count is below threshold.

    Args:
        hotspots: List of hotspot dicts to analyze
        goal: Improvement goal
        repo_root: Repository root path

    Returns:
        List of analysis results
    """
    if not hotspots:
        return []

    # Load improvement config
    config = ScoutConfig()
    improvement_config = config.get("improvement") or {}
    max_agents = improvement_config.get("max_agents", 3)
    agent_timeout = improvement_config.get("agent_timeout_seconds", 120)
    max_cost = improvement_config.get("max_cost_per_session", 1.00)

    # Use sequential for small hotspot counts (threshold = 2)
    if len(hotspots) < 2:
        logger.info(f"Using sequential analysis for {len(hotspots)} hotspot(s)")
        analyses = []
        for hotspot in hotspots:
            analysis = await _analyze_single_hotspot(hotspot, goal, repo_root)
            analyses.append(analysis)
        return analyses

    logger.info(f"Using parallel analysis for {len(hotspots)} hotspots (max_agents={max_agents})")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_agents)
    total_cost = 0.0
    budget_exceeded = False

    async def bounded_analyze(hotspot: dict[str, Any]) -> dict[str, Any]:
        nonlocal total_cost, budget_exceeded

        async with semaphore:
            # Check budget before executing
            if budget_exceeded or total_cost >= max_cost:
                budget_exceeded = True
                return {
                    "file": hotspot.get("file", ""),
                    "status": "skipped",
                    "issues": [],
                    "error": "Budget exceeded",
                }

            try:
                result = await asyncio.wait_for(
                    _analyze_single_hotspot(hotspot, goal, repo_root),
                    timeout=agent_timeout,
                )
                # Estimate cost per analysis (rough)
                total_cost += 0.01
                return result
            except asyncio.TimeoutError:
                logger.warning(f"Analysis timeout for {hotspot.get('file')}")
                return {
                    "file": hotspot.get("file", ""),
                    "status": "timeout",
                    "issues": [],
                    "error": f"Analysis timed out after {agent_timeout}s",
                }
            except Exception as e:
                logger.exception(f"Error analyzing {hotspot.get('file')}")
                return {
                    "file": hotspot.get("file", ""),
                    "status": "error",
                    "issues": [],
                    "error": str(e)[:200],
                }

    # Execute all analyses in parallel
    tasks = [bounded_analyze(hotspot) for hotspot in hotspots]
    analyses = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results, converting exceptions to error results
    processed_analyses = []
    for i, result in enumerate(analyses):
        if isinstance(result, Exception):
            processed_analyses.append({
                "file": hotspots[i].get("file", ""),
                "status": "error",
                "issues": [],
                "error": str(result)[:200],
            })
        else:
            processed_analyses.append(result)

    if budget_exceeded:
        logger.warning("Budget exceeded during parallel analysis")

    logger.info(f"Parallel analysis complete: {len(processed_analyses)} results")
    return processed_analyses


# Backward compatibility alias
_analyze_hotspot = _analyze_single_hotspot


async def _generate_plan(
    analysis: dict[str, Any],
    goal: str,
    repo_root: Path,
) -> dict[str, Any]:
    """
    Generate an improvement plan based on analysis findings.

    Uses scout_plan to generate actionable improvements.
    """
    file_path = analysis.get("file", "")
    issues = analysis.get("issues", [])

    if not issues:
        return {
            "file": file_path,
            "status": "no_issues",
            "plan": "No improvements needed",
        }

    # Build prompt for planning
    issue_summary = "\n".join(f"- {i}" for i in issues[:5])
    prompt = f"""Improve {file_path}:

Issues found:
{issue_summary}

Goal: {goal}

Generate a prioritized list of improvements with specific fixes."""

    try:
        cmd = [
            sys.executable,
            "-m",
            "vivarium.scout.cli.plan",
            "--goal",
            prompt,
            "--target",
            file_path,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(repo_root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=120.0
        )

        output = stdout.decode() if stdout else ""
        error = stderr.decode() if stderr else ""

        if proc.returncode != 0:
            return {
                "file": file_path,
                "status": "error",
                "plan": "",
                "error": error[:200],
            }

        return {
            "file": file_path,
            "status": "success",
            "plan": output[:2000] if output else "No plan generated",
        }

    except asyncio.TimeoutError:
        return {
            "file": file_path,
            "status": "timeout",
            "plan": "",
            "error": "Planning timed out",
        }
    except Exception as e:
        logger.exception(f"Error generating plan for {file_path}")
        return {
            "file": file_path,
            "status": "error",
            "plan": "",
            "error": str(e)[:200],
        }


async def _apply_fix(
    plan: dict[str, Any],
    repo_root: Path,
    dry_run: bool = False,
    auto_approved: bool = False,
) -> dict[str, Any]:
    """
    Apply safe fixes from the improvement plan with safety checks.

    Safety workflow:
        1. Classify edit safety (safe/review/dangerous)
        2. If dangerous, reject immediately
        3. If review, queue for human review (log pending)
        4. If safe:
            a. Create backup
            b. If dry_run, show diff and return without applying
            c. Apply edit via scout_edit
            d. On success, log audit event
            e. On failure, restore from backup

    Args:
        plan: The improvement plan dict containing file and plan text
        repo_root: Repository root path
        dry_run: If True, show diff without applying changes
        auto_approved: If True, skip confirmation prompts

    Returns:
        Dict with status: "applied", "pending_review", "rejected", "dry_run", or "error"
    """
    file_path = plan.get("file", "")
    status = plan.get("status", "")
    plan_text = plan.get("plan", "")
    plan_type = plan.get("plan_type", "unknown")
    confidence = plan.get("confidence")

    full_path = repo_root / file_path if file_path else None

    if status != "success":
        improvement_tracker.record_improvement_outcome(
            file=file_path,
            plan_type=plan_type,
            success=False,
            confidence=confidence,
            error=f"plan status: {status}",
        )
        return {
            "file": file_path,
            "status": "skipped",
            "reason": f"plan status: {status}",
        }

    if not full_path or not full_path.exists():
        return {
            "file": file_path,
            "status": "error",
            "reason": f"file not found: {file_path}",
        }

    # Step 1: Classify safety
    safety_level = classify_edit(plan)
    logger.info(f"Safety classification for {file_path}: {safety_level}")

    # Step 2: Handle based on safety level
    if safety_level == "dangerous":
        logger.warning(f"Rejecting dangerous edit: {file_path}")
        return {
            "file": file_path,
            "status": "rejected",
            "reason": "Safety check failed: dangerous edit",
            "safety_level": safety_level,
        }

    if safety_level == "review":
        logger.info(f"Queuing for review: {file_path}")
        return {
            "file": file_path,
            "status": "pending_review",
            "reason": "Requires human review",
            "safety_level": safety_level,
            "safety_description": get_safety_description(safety_level),
        }

    # safety_level == "safe" - proceed with application

    # Step 3: Create backup before applying
    backup_path = get_backup_path(str(full_path))
    try:
        shutil.copy2(str(full_path), backup_path)
        logger.info(f"Created backup: {backup_path}")
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return {
            "file": file_path,
            "status": "error",
            "reason": f"Failed to create backup: {e}",
        }

    # Step 4: If dry_run, get diff and return without applying
    if dry_run:
        try:
            diff_result = await _get_diff_preview(full_path, plan_text)
            return {
                "file": file_path,
                "status": "dry_run",
                "safety_level": safety_level,
                "diff": diff_result.get("diff", ""),
                "backup_path": backup_path,
            }
        except Exception as e:
            return {
                "file": file_path,
                "status": "error",
                "reason": f"Failed to generate diff: {e}",
            }

    # Step 5: Apply the edit via scout_edit
    try:
        result = await _apply_edit_via_scout(full_path, plan_text)

        if result.get("error"):
            # Restore from backup on failure
            logger.error(f"Edit failed, restoring backup: {result.get('error')}")
            shutil.copy2(backup_path, str(full_path))
            return {
                "file": file_path,
                "status": "error",
                "reason": result.get("error", "Edit failed"),
                "restored_from_backup": True,
            }

        # Success - log audit event
        audit = AuditLog()
        audit.log(
            "apply_fix_success",
            file=file_path,
            plan_type=plan_type,
            confidence=confidence,
            safety_level=safety_level,
        )

        # Record successful application
        improvement_tracker.record_improvement_outcome(
            file=file_path,
            plan_type=plan_type,
            success=True,
            confidence=confidence,
        )

        logger.info(f"Successfully applied fix to {file_path}")
        return {
            "file": file_path,
            "status": "applied",
            "safety_level": safety_level,
            "backup_path": backup_path,
        }

    except Exception as e:
        # Restore from backup on unexpected error
        logger.exception(f"Unexpected error applying fix: {e}")
        try:
            shutil.copy2(backup_path, str(full_path))
        except Exception as restore_err:
            logger.error(f"Failed to restore backup: {restore_err}")

        return {
            "file": file_path,
            "status": "error",
            "reason": str(e)[:200],
        }


async def _get_diff_preview(file_path: Path, instruction: str) -> dict[str, Any]:
    """
    Get a diff preview of what the edit would look like.

    Uses scout_edit with dry_run=True to generate the diff.

    Args:
        file_path: Path to the file to preview
        instruction: Edit instruction

    Returns:
        Dict with 'diff' key containing the unified diff
    """
    from scout.cli_enhanced.mcp_bridge.client import get_mcp_client

    client = get_mcp_client()
    result = await client.call_tool("scout_edit", {
        "file_path": str(file_path),
        "instruction": instruction,
        "dry_run": True,
    })

    if "error" in result:
        return {"error": result.get("error"), "diff": ""}

    # Parse result
    tool_result = result.get("result", {})
    if isinstance(tool_result, str):
        try:
            tool_result = json.loads(tool_result)
        except json.JSONDecodeError:
            pass

    return tool_result


async def _apply_edit_via_scout(file_path: Path, instruction: str) -> dict[str, Any]:
    """
    Apply an edit using scout_edit tool.

    Args:
        file_path: Path to the file to edit
        instruction: Edit instruction

    Returns:
        Dict with 'status', 'result', or 'error'
    """
    from scout.cli_enhanced.mcp_bridge.client import get_mcp_client

    client = get_mcp_client()
    result = await client.call_tool("scout_edit", {
        "file_path": str(file_path),
        "instruction": instruction,
        "dry_run": False,
    })

    if "error" in result:
        return {"error": result.get("error")}

    return result


def rollback_from_backup(file_path: str) -> dict[str, Any]:
    """
    Rollback a file from its .bak backup.

    Args:
        file_path: Path to the file to rollback

    Returns:
        Dict with 'status' and 'message'
    """
    backup_path = get_backup_path(file_path)

    if not os.path.exists(backup_path):
        return {
            "status": "error",
            "message": f"No backup found: {backup_path}",
        }

    try:
        shutil.copy2(backup_path, file_path)
        return {
            "status": "success",
            "message": f"Rolled back from backup: {backup_path}",
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to rollback: {e}",
        }


def _parse_roast_output(output: str) -> list[str]:
    """Parse scout_roast output to extract issue strings."""
    issues = []

    # Simple parsing - look for common patterns
    lines = output.split("\n")
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and len(line) > 10:
            # Heuristic: lines that look like issues
            if any(
                keyword in line.lower()
                for keyword in ["issue", "problem", "complex", "fix", "improve"]
            ):
                issues.append(line[:200])

    return issues[:10]  # Limit to top 10


def _estimate_cost(result: ImprovementResult) -> float:
    """
    Estimate pipeline cost based on operations performed.

    Rough estimates:
    - Hotspot detection: $0.001 (local analysis)
    - Analysis per file: $0.01 (roast LLM)
    - Planning per file: $0.02 (plan LLM)
    - Apply per file: $0.005 (edit LLM)
    """
    base = 0.001
    analysis_cost = len(result.analyses) * 0.01
    plan_cost = len(result.plans) * 0.02
    apply_cost = len(result.applied) * 0.005

    return base + analysis_cost + plan_cost + apply_cost


# === CLI Integration ===

def main():
    """CLI entry point for `scout improve`."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Autonomous code improvement pipeline"
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Specific file or directory to improve",
    )
    parser.add_argument(
        "--goal",
        default="improve code quality",
        help="Improvement goal (e.g., 'reduce complexity', 'fix bugs')",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Automatically apply safe fixes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show diff without applying changes",
    )
    parser.add_argument(
        "--rollback",
        metavar="FILE",
        help="Rollback a file from its .bak backup",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Handle rollback separately
    if args.rollback:
        result = rollback_from_backup(args.rollback)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"Rollback {result['status']}: {result.get('message', '')}")
        return

    # Run pipeline
    result = asyncio.run(
        run_improve_pipeline(
            goal=args.goal,
            target=args.target,
            apply_fixes=args.apply,
            dry_run=args.dry_run,
        )
    )

    # Output
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_human_output(result)


def _print_human_output(result: ImprovementResult) -> None:
    """Print human-readable pipeline output."""
    print(f"\n## Scout Improve Results")
    print(f"Status: {result.status}")
    print(f"Estimated cost: ${result.cost:.4f}")
    print()

    if result.hotspots:
        print(f"### Hotspots ({len(result.hotspots)})")
        for h in result.hotspots:
            print(f"  - {h.get('file')}: {h.get('reason')} (impact: {h.get('impact')})")
        print()

    if result.analyses:
        print(f"### Analyses ({len(result.analyses)})")
        for a in result.analyses:
            issues = a.get("issues", [])
            print(f"  - {a.get('file')}: {len(issues)} issues found")
        print()

    if result.plans:
        print(f"### Plans ({len(result.plans)})")
        for p in result.plans:
            print(f"  - {p.get('file')}: {p.get('status')}")
        print()

    if result.applied:
        print(f"### Applied ({len(result.applied)})")
        for a in result.applied:
            print(f"  - {a.get('file')}: {a.get('status')}")
        print()

    if result.errors:
        print("### Errors")
        for e in result.errors:
            print(f"  - {e}")
        print()


if __name__ == "__main__":
    main()
