"""Self-Improvement Suggestion Engine.

Generates actionable suggestions from audit log analysis:
- Cost reduction: Identify high-cost sessions and suggest model adjustments
- Tool creation: Detect frequent searches without corresponding tools
- File ignore: Find files with repeated validation failures
- Timeout adjustment: Detect frequent timeout skips
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

from scout.self_improvement.audit_analyzer import (
    load_audit_events,
    compute_cost_metrics,
    compute_frequency_metrics,
)

# Paths
AUDIT_MONITOR_DB = Path("~/.scout/audit_monitor.db").expanduser()
SUGGESTIONS_CONFIG = Path("~/.scout/suggestions.yaml").expanduser()

# Default thresholds
DEFAULT_THRESHOLDS = {
    "cost_per_session_high": 2.00,  # USD
    "frequent_search_min_count": 10,
    "frequent_search_days": 7,
    "validation_failure_file_threshold": 5,
    "timeout_skip_min_count": 10,
}

DEFAULT_ACTIONS = {
    "model_recommendations": True,
    "tool_creation": True,
    "file_ignore": True,
    "timeout_adjustment": True,
}


@dataclass
class Suggestion:
    """A self-improvement suggestion generated from audit analysis."""

    source: str  # "audit_cost" | "audit_pattern" | "validation_failure"
    action: str  # "adjust_config" | "create_tool" | "add_ignore" | "adjust_timeout"
    tool_name: Optional[str]
    reason: str
    suggestion: str
    evidence: dict
    severity: str  # "low" | "medium" | "high"


def load_config() -> dict:
    """Load suggestions configuration from YAML file."""
    import yaml

    config = {
        "thresholds": DEFAULT_THRESHOLDS.copy(),
        "actions": DEFAULT_ACTIONS.copy(),
    }

    if SUGGESTIONS_CONFIG.exists():
        try:
            with open(SUGGESTIONS_CONFIG) as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    if "thresholds" in user_config:
                        config["thresholds"].update(user_config["thresholds"])
                    if "actions" in user_config:
                        config["actions"].update(user_config["actions"])
        except Exception:
            pass

    return config


def save_config(config: dict):
    """Save suggestions configuration to YAML file."""
    import yaml

    SUGGESTIONS_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(SUGGESTIONS_CONFIG, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_existing_tools() -> list[str]:
    """Get list of existing tool names from the tools directory."""
    tools_dir = Path(__file__).parent.parent / "tools"
    if not tools_dir.exists():
        return []

    tools = []
    for f in tools_dir.glob("*.py"):
        if f.name.startswith("_"):
            continue
        tools.append(f.stem)

    return tools


def find_similar_tools(
    query: str, existing_tools: list[str], threshold: float = 0.8
) -> list[tuple[str, float]]:
    """Find tools with similar names using fuzzy matching."""
    matches = []
    query_lower = query.lower()

    for tool in existing_tools:
        # Check exact substring match
        if query_lower in tool.lower():
            matches.append((tool, 1.0))
            continue

        # Check fuzzy match
        ratio = SequenceMatcher(None, query_lower, tool.lower()).ratio()
        if ratio >= threshold:
            matches.append((tool, ratio))

    return sorted(matches, key=lambda x: x[1], reverse=True)


def analyze_cost_reduction(events: list[dict], config: dict) -> list[Suggestion]:
    """Analyze cost patterns and generate cost reduction suggestions."""
    suggestions = []
    thresholds = config["thresholds"]

    cost_metrics = compute_cost_metrics(events)
    session_costs = cost_metrics.get("session_costs", [])

    # Check for high-cost sessions
    for session_id, session_cost in session_costs:
        if session_cost > thresholds["cost_per_session_high"]:
            # Suggest using cheaper model
            suggestions.append(
                Suggestion(
                    source="audit_cost",
                    action="adjust_config",
                    tool_name="scout",
                    reason=f"Session {session_id[:8]} cost ${session_cost:.2f} exceeds threshold ${thresholds['cost_per_session_high']}",
                    suggestion="Consider switching to a cheaper model for high-volume sessions or set a session budget cap",
                    evidence={
                        "session_id": session_id,
                        "session_cost": session_cost,
                        "threshold": thresholds["cost_per_session_high"],
                    },
                    severity=(
                        "high"
                        if session_cost > thresholds["cost_per_session_high"] * 2
                        else "medium"
                    ),
                )
            )

    # Check for expensive models
    per_model_cost = cost_metrics.get("per_model_cost", {})
    if per_model_cost:
        most_expensive = max(per_model_cost.items(), key=lambda x: x[1])
        if most_expensive[1] > thresholds["cost_per_session_high"]:
            suggestions.append(
                Suggestion(
                    source="audit_cost",
                    action="adjust_config",
                    tool_name="scout",
                    reason=f"Model {most_expensive[0]} cost ${most_expensive[1]:.2f} is highest",
                    suggestion=f"Consider using a cheaper model like 'llama-3.1-8b-instant' instead of '{most_expensive[0]}' for non-critical tasks",
                    evidence={
                        "model": most_expensive[0],
                        "total_cost": most_expensive[1],
                        "all_models": per_model_cost,
                    },
                    severity="medium",
                )
            )

    return suggestions


def analyze_tool_creation(events: list[dict], config: dict) -> list[Suggestion]:
    """Analyze search patterns and suggest new tool creation."""
    suggestions = []
    thresholds = config["thresholds"]
    existing_tools = get_existing_tools()

    frequency_metrics = compute_frequency_metrics(events)
    most_searched = frequency_metrics.get("most_searched_symbols", [])

    # Count searches per symbol in the time window
    days = thresholds.get("frequent_search_days", 7)
    min_count = thresholds.get("frequent_search_min_count", 10)

    for symbol, count in most_searched:
        if count < min_count:
            continue

        # Check if similar tool already exists
        similar = find_similar_tools(symbol, existing_tools, threshold=0.8)

        if similar:
            # Suggest using existing tool instead
            suggestions.append(
                Suggestion(
                    source="audit_pattern",
                    action="adjust_metadata",
                    tool_name="search",
                    reason=f"Query '{symbol}' matches existing tool '{similar[0][0]}' with {count} searches",
                    suggestion=f"Consider using `scout {similar[0][0]}` instead of general search",
                    evidence={
                        "symbol": symbol,
                        "search_count": count,
                        "similar_tools": similar,
                        "days": days,
                    },
                    severity="low",
                )
            )
        else:
            # Create new tool suggestion
            suggestions.append(
                Suggestion(
                    source="audit_pattern",
                    action="create_tool",
                    tool_name=symbol,
                    reason=f"Symbol '{symbol}' searched {count} times in {days} days without existing tool",
                    suggestion=f"Create a new `scout {symbol}` command to handle this frequent query",
                    evidence={
                        "symbol": symbol,
                        "search_count": count,
                        "days": days,
                    },
                    severity="medium" if count > min_count * 2 else "low",
                )
            )

    return suggestions


def analyze_file_ignore(events: list[dict], config: dict) -> list[Suggestion]:
    """Analyze validation failures and suggest file ignore patterns."""
    suggestions = []
    thresholds = config["thresholds"]

    # Collect validation failures per file
    file_failures: dict[str, int] = {}

    for event in events:
        if event.get("event") == "validation_fail":
            files = event.get("files", [])
            if isinstance(files, list):
                for f in files:
                    file_failures[f] = file_failures.get(f, 0) + 1

    # Find files with repeated failures
    threshold = thresholds.get("validation_failure_file_threshold", 5)
    for file_path, count in file_failures.items():
        if count >= threshold:
            suggestions.append(
                Suggestion(
                    source="validation_failure",
                    action="add_ignore",
                    tool_name="scout",
                    reason=f"File {file_path} failed validation {count} times",
                    suggestion=f"Add {file_path} to ignore patterns in .scout/config.yaml or add validation exception",
                    evidence={
                        "file": file_path,
                        "failure_count": count,
                        "threshold": threshold,
                    },
                    severity="medium",
                )
            )

    return suggestions


def analyze_timeout_adjustment(events: list[dict], config: dict) -> list[Suggestion]:
    """Analyze timeout skips and suggest timeout adjustments."""
    suggestions = []
    thresholds = config["thresholds"]

    # Count timeout/skip events
    timeout_count = 0
    for event in events:
        event_type = event.get("event", "")
        reason = event.get("reason", "")

        if event_type in ("skip", "budget") or "timeout" in reason.lower():
            timeout_count += 1

    min_count = thresholds.get("timeout_skip_min_count", 10)

    if timeout_count >= min_count:
        suggestions.append(
            Suggestion(
                source="audit_pattern",
                action="adjust_timeout",
                tool_name="scout",
                reason=f"Detected {timeout_count} timeout/skip events",
                suggestion="Consider increasing timeout thresholds in config or optimizing the affected operations",
                evidence={
                    "timeout_count": timeout_count,
                    "threshold": min_count,
                },
                severity="medium",
            )
        )

    return suggestions


def generate_suggestions(days: int = 7) -> list[Suggestion]:
    """Generate all suggestions from audit analysis.

    Args:
        days: Number of days to analyze (default: 7)

    Returns:
        List of Suggestion objects
    """
    config = load_config()
    actions = config.get("actions", DEFAULT_ACTIONS)

    # Load events for the time window
    since = datetime.now(timezone.utc) - timedelta(days=days)
    until = datetime.now(timezone.utc)
    events = load_audit_events(since, until)

    if not events:
        return []

    all_suggestions = []

    # Generate suggestions based on enabled actions
    if actions.get("model_recommendations", True):
        all_suggestions.extend(analyze_cost_reduction(events, config))

    if actions.get("tool_creation", True):
        all_suggestions.extend(analyze_tool_creation(events, config))

    if actions.get("file_ignore", True):
        all_suggestions.extend(analyze_file_ignore(events, config))

    if actions.get("timeout_adjustment", True):
        all_suggestions.extend(analyze_timeout_adjustment(events, config))

    return all_suggestions


def save_suggestions_to_db(suggestions: list[Suggestion]) -> int:
    """Save generated suggestions to the database.

    Returns:
        Number of new suggestions saved
    """
    if not AUDIT_MONITOR_DB.exists():
        return 0

    conn = sqlite3.connect(str(AUDIT_MONITOR_DB))
    conn.row_factory = sqlite3.Row

    new_count = 0

    for suggestion in suggestions:
        # Check if similar suggestion already exists
        cursor = conn.execute(
            """
            SELECT id FROM suggestions 
            WHERE source = ? AND action = ? AND tool_name = ? 
            AND status = 'new'
            AND dismissed = 0
        """,
            (suggestion.source, suggestion.action, suggestion.tool_name),
        )

        if cursor.fetchone():
            continue  # Skip duplicate

        # Insert new suggestion
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO suggestions 
            (source, action, tool_name, reason, suggestion, evidence, severity, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?)
        """,
            (
                suggestion.source,
                suggestion.action,
                suggestion.tool_name,
                suggestion.reason,
                suggestion.suggestion,
                json.dumps(suggestion.evidence),
                suggestion.severity,
                now,
            ),
        )
        new_count += 1

    conn.commit()
    conn.close()

    return new_count


def get_suggestions_from_db(status: str = "new", limit: int = 20) -> list[dict]:
    """Get suggestions from database.

    Args:
        status: Filter by status (new, applied, dismissed, all)
        limit: Maximum number to return

    Returns:
        List of suggestion dictionaries
    """
    if not AUDIT_MONITOR_DB.exists():
        return []

    conn = sqlite3.connect(str(AUDIT_MONITOR_DB))
    conn.row_factory = sqlite3.Row

    if status == "all":
        cursor = conn.execute(
            """
            SELECT * FROM suggestions ORDER BY created_at DESC LIMIT ?
        """,
            (limit,),
        )
    else:
        cursor = conn.execute(
            """
            SELECT * FROM suggestions WHERE status = ? ORDER BY created_at DESC LIMIT ?
        """,
            (status, limit),
        )

    suggestions = []
    for row in cursor.fetchall():
        suggestions.append(
            {
                "id": row["id"],
                "source": row["source"],
                "action": row["action"],
                "tool_name": row["tool_name"],
                "reason": row["reason"],
                "suggestion": row["suggestion"],
                "evidence": json.loads(row["evidence"]) if row["evidence"] else {},
                "severity": row["severity"],
                "status": row["status"],
                "created_at": row["created_at"],
                "applied_at": row["applied_at"],
            }
        )

    conn.close()
    return suggestions


def update_suggestion_status(suggestion_id: int, status: str) -> bool:
    """Update suggestion status.

    Args:
        suggestion_id: ID of suggestion to update
        status: New status (applied, dismissed)

    Returns:
        True if successful, False otherwise
    """
    if not AUDIT_MONITOR_DB.exists():
        return False

    conn = sqlite3.connect(str(AUDIT_MONITOR_DB))

    now = datetime.now(timezone.utc).isoformat()

    if status == "applied":
        conn.execute(
            """
            UPDATE suggestions SET status = 'applied', applied_at = ? WHERE id = ?
        """,
            (now, suggestion_id),
        )
    elif status == "dismissed":
        conn.execute(
            """
            UPDATE suggestions SET status = 'dismissed', dismissed = 1 WHERE id = ?
        """,
            (suggestion_id,),
        )

    conn.commit()
    conn.close()

    return True
