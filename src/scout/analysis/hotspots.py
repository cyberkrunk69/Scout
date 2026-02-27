"""Hotspot Detection Module - Identifies high-impact areas of the codebase."""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from scout.audit import AuditLog
from scout.config import HOTSPOT_WEIGHT_CHURN, HOTSPOT_WEIGHT_ERROR, HOTSPOT_WEIGHT_IMPACT
from scout.graph import impact_analysis


# Aliases for backwards compatibility - import from scout.config instead
DEFAULT_CHURN_WEIGHT = HOTSPOT_WEIGHT_CHURN
DEFAULT_ERROR_WEIGHT = HOTSPOT_WEIGHT_ERROR
DEFAULT_IMPACT_WEIGHT = HOTSPOT_WEIGHT_IMPACT

# Error event types to track
ERROR_EVENT_TYPES = frozenset({"validation_fail", "llm_error", "llm_retry", "node_failed"})

# Heuristic patterns to extract module from stack traces
STACK_TRACE_PATTERNS = [
    re.compile(r"File\s+\"([^\"]+)\""),  # File in traceback
    re.compile(r"([\w.]+)\.py:\d+"),  # module.function:line
]


def get_file_churn(
    days: int = 30,
    repo_path: Optional[Path] = None,
) -> dict[str, int]:
    """
    Calculate file modification frequency from git history.

    Args:
        days: Number of days to look back (default: 30)
        repo_path: Repository root path (default: current directory)

    Returns:
        Dictionary mapping file paths to modification counts
    """
    repo_path = repo_path or Path.cwd()
    since_date = datetime.now(timezone.utc) - timedelta(days=days)
    since_str = since_date.strftime("%Y-%m-%d")

    # Get git log with file modifications
    # Using --name-only to get just the filenames
    cmd = [
        "git",
        "log",
        f"--since={since_str}",
        "--name-only",
        "--pretty=format:",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, Exception):
        # Return empty dict if not a git repo, git not installed, or other errors
        return {}

    # Parse the output - each file is on its own line between commit blocks
    file_counts: dict[str, int] = {}
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip non-file entries (like merge commit messages in output)
        if "/" not in line and not line.endswith(".py"):
            continue
        file_counts[line] = file_counts.get(line, 0) + 1

    return file_counts


def get_error_rates(
    days: int = 30,
    audit_path: Optional[Path] = None,
) -> dict[str, int]:
    """
    Extract error rates from audit logs.

    Args:
        days: Number of days to look back (default: 30)
        audit_path: Path to audit log file (default: ~/.scout/audit.jsonl)

    Returns:
        Dictionary mapping module/file paths to error counts
    """
    audit_path = audit_path or Path("~/.scout/audit.jsonl").expanduser()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Query audit logs for error events
    try:
        log = AuditLog(path=audit_path)
    except Exception:
        # Return empty dict if audit log cannot be initialized
        return {}

    error_events: list[dict[str, Any]] = []

    for event_type in ERROR_EVENT_TYPES:
        try:
            events = log.query(since=cutoff, event_type=event_type)
            error_events.extend(events)
        except Exception:
            # Skip this event type if query fails
            continue

    # Extract module/file from each error event
    module_counts: dict[str, int] = {}

    for event in error_events:
        module = _extract_module_from_event(event)
        if module:
            module_counts[module] = module_counts.get(module, 0) + 1

    return module_counts


def _extract_module_from_event(event: dict[str, Any]) -> Optional[str]:
    """
    Extract module/file path from an audit event.

    Uses heuristics to parse stack traces or extract from metadata.
    """
    # Try to get from 'files' field (commonly present in nav events)
    files = event.get("files", [])
    if files and isinstance(files, list):
        return files[0] if files else None

    # Try to extract from 'reason' or 'raw_brief_path'
    reason = event.get("reason", "")
    if reason:
        match = STACK_TRACE_PATTERNS[0].search(reason)
        if match:
            return match.group(1)

    # Try raw_brief_path
    raw_brief = event.get("raw_brief_path", "")
    if raw_brief and isinstance(raw_brief, str):
        # Extract module from path like /path/to/module.py
        # Try to find .py file in the path
        match = re.search(r"([\w/]+\.py)", raw_brief)
        if match:
            py_path = match.group(1)
            # If it's an absolute path, try to make it relative
            if py_path.startswith("/"):
                # Try to extract relative path by removing common prefixes
                parts = py_path.split("/")
                for i, part in enumerate(parts):
                    if part == "src" or part == "lib":
                        return "/".join(parts[i:])
            return py_path

    return None


def get_impact_counts(
    files: list[str],
    repo_path: Optional[Path] = None,
) -> dict[str, int]:
    """
    Get impact counts for given files using graph.impact_analysis.

    Args:
        files: List of file paths to analyze
        repo_path: Repository root path (default: current directory)

    Returns:
        Dictionary mapping file paths to number of affected symbols
    """
    repo_path = repo_path or Path.cwd()
    impact_counts: dict[str, int] = {}

    for file_path in files:
        try:
            affected = impact_analysis(file_path, repo_root=repo_path)
            impact_counts[file_path] = len(affected)
        except Exception:
            # If impact analysis fails, treat as unknown
            impact_counts[file_path] = 0

    return impact_counts


def compute_hotspot_score(
    churn: int,
    errors: int,
    impact_count: int,
    max_churn: int = 10,
    max_errors: int = 10,
    max_impact: int = 50,
    churn_weight: float = DEFAULT_CHURN_WEIGHT,
    error_weight: float = DEFAULT_ERROR_WEIGHT,
    impact_weight: float = DEFAULT_IMPACT_WEIGHT,
) -> float:
    """
    Compute normalized hotspot score from raw metrics.

    Args:
        churn: Number of file modifications
        errors: Number of errors in the file
        impact_count: Number of symbols affected by this file
        max_churn: Maximum churn for normalization (default: 10)
        max_errors: Maximum errors for normalization (default: 10)
        max_impact: Maximum impact count for normalization (default: 50)
        churn_weight: Weight for churn component (default: 0.4)
        error_weight: Weight for error component (default: 0.4)
        impact_weight: Weight for impact component (default: 0.2)

    Returns:
        Normalized score between 0 and 1
    """
    # Normalize each component to 0-1 range
    norm_churn = min(churn / max_churn, 1.0) if max_churn > 0 else 0.0
    norm_errors = min(errors / max_errors, 1.0) if max_errors > 0 else 0.0
    norm_impact = min(impact_count / max_impact, 1.0) if max_impact > 0 else 0.0

    # Weighted sum
    score = (
        churn_weight * norm_churn
        + error_weight * norm_errors
        + impact_weight * norm_impact
    )

    return round(score, 3)


def scout_hotspots(
    days: int = 30,
    limit: int = 10,
    include_impact: bool = True,
    churn_weight: float = DEFAULT_CHURN_WEIGHT,
    error_weight: float = DEFAULT_ERROR_WEIGHT,
    impact_weight: float = DEFAULT_IMPACT_WEIGHT,
    repo_path: Optional[str] = None,
    audit_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Identify hotspots - high-impact areas of the codebase.

    Combines git churn, error rates, and dependency impact to identify
    files that are frequently changed, error-prone, and affect many
    downstream symbols.

    Args:
        days: Number of days to analyze (default: 30)
        limit: Maximum number of hotspots to return (default: 10)
        include_impact: Whether to compute impact analysis (default: True)
        churn_weight: Weight for churn component (default: 0.4)
        error_weight: Weight for error component (default: 0.4)
        impact_weight: Weight for impact component (default: 0.2)
        repo_path: Repository root path (default: auto-detect)
        audit_path: Path to audit log (default: ~/.scout/audit.jsonl)

    Returns:
        Dictionary with hotspots list and metadata
    """
    repo = Path(repo_path) if repo_path else None
    audit = Path(audit_path) if audit_path else None

    # Get raw metrics
    churn_data = get_file_churn(days=days, repo_path=repo)
    error_data = get_error_rates(days=days, audit_path=audit)

    # Get all files from both sources
    all_files = set(churn_data.keys()) | set(error_data.keys())

    # Get impact counts if requested
    impact_data: dict[str, int] = {}
    if include_impact:
        impact_data = get_impact_counts(list(all_files), repo_path=repo)

    # Calculate max values for normalization
    max_churn = max(churn_data.values()) if churn_data else 1
    max_errors = max(error_data.values()) if error_data else 1
    max_impact = max(impact_data.values()) if impact_data else 1

    # Build hotspot entries
    hotspots: list[dict[str, Any]] = []
    for file_path in all_files:
        churn = churn_data.get(file_path, 0)
        errors = error_data.get(file_path, 0)
        impact_count = impact_data.get(file_path, 0)

        score = compute_hotspot_score(
            churn=churn,
            errors=errors,
            impact_count=impact_count,
            max_churn=max_churn,
            max_errors=max_errors,
            max_impact=max_impact,
            churn_weight=churn_weight,
            error_weight=error_weight,
            impact_weight=impact_weight,
        )

        hotspots.append({
            "file": file_path,
            "churn": churn,
            "errors": errors,
            "impact_count": impact_count,
            "score": score,
        })

    # Sort by score descending
    hotspots.sort(key=lambda x: x["score"], reverse=True)

    # Return top N
    return {
        "hotspots": hotspots[:limit],
        "metadata": {
            "days_analyzed": days,
            "total_files_evaluated": len(all_files),
            "weights": {
                "churn": churn_weight,
                "errors": error_weight,
                "impact": impact_weight,
            },
        },
    }
