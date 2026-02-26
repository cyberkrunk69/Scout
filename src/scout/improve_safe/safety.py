#!/usr/bin/env python
"""
Safety classifier for auto-apply edits.

Provides classification of edit safety levels to prevent destructive changes.
"""

from __future__ import annotations

from typing import Any

# Keywords that indicate safe edits (comments, whitespace, formatting, imports)
SAFE_KEYWORDS = [
    "add docstring",
    "add comment",
    "format",
    "formatting",
    "add import",
    "remove import",
    "add whitespace",
    "fix whitespace",
    "fix indent",
    "fix formatting",
    "organize imports",
    "sort imports",
    "add blank line",
    "remove blank line",
    "fix typo",
    "fix comment",
    "normalize",
    "cleanup",
    "clean up",
]

# Keywords that indicate dangerous edits (deletions, signature changes)
DANGEROUS_KEYWORDS = [
    "rm ",
    "delete ",
    "remove ",
    "drop ",
    "cut ",
    # Line-level dangerous operations
    "delete line",
    "remove line",
    "drop line",
    "delete function",
    "remove function",
    "delete class",
    "remove class",
    "delete method",
    "remove method",
    # Signature modifications
    "change function signature",
    "modify function signature",
    "change signature",
    "modify signature",
    "rename parameter",
    "change parameter",
    # Dangerous file operations
    "delete file",
    "remove file",
    "truncate",
    "overwrite",
    # SQL/database dangerous operations
    "drop table",
    "delete table",
    "drop database",
    # Git dangerous operations
    "git rm",
    "git remove",
    # Shell dangerous operations
    "rm -rf",
    "rm -r",
]


def classify_edit(plan: dict[str, Any]) -> str:
    """
    Classify the safety level of an edit based on the plan.

    Args:
        plan: A dictionary containing the edit plan with keys like:
            - file: file path
            - plan: description of the change
            - instruction: the edit instruction

    Returns:
        One of: "safe", "review", "dangerous"

    Classification Rules:
        - safe: Edits that only affect comments, whitespace, imports, or simple formatting
        - dangerous: Edits that delete lines, modify function signatures, or contain risky keywords
        - review: Everything else (requires human review)
    """
    plan_text = plan.get("plan", "") or plan.get("instruction", "") or ""
    plan_lower = plan_text.lower()

    # Check for dangerous keywords first (highest priority)
    for keyword in DANGEROUS_KEYWORDS:
        if keyword in plan_lower:
            return "dangerous"

    # Check for safe keywords
    for keyword in SAFE_KEYWORDS:
        if keyword in plan_lower:
            return "safe"

    # Default to review for anything else
    return "review"


def get_safety_description(safety_level: str) -> str:
    """
    Get a human-readable description of the safety level.

    Args:
        safety_level: One of "safe", "review", "dangerous"

    Returns:
        Human-readable description
    """
    descriptions = {
        "safe": "This edit only affects comments, whitespace, imports, or formatting. It can be automatically applied.",
        "review": "This edit requires human review before application. It may modify code behavior.",
        "dangerous": "This edit is potentially destructive (deletes code, modifies signatures, etc.). It will NOT be automatically applied.",
    }
    return descriptions.get(safety_level, "Unknown safety level")


# === Backup Management ===


def get_backup_path(file_path: str) -> str:
    """
    Get the backup file path for a given file.

    Args:
        file_path: Original file path

    Returns:
        Backup file path (original + .bak)
    """
    return f"{file_path}.bak"


def is_backup_stale(backup_path: str, max_age_hours: int = 24) -> bool:
    """
    Check if a backup file is older than the maximum age.

    Args:
        backup_path: Path to the backup file
        max_age_hours: Maximum age in hours (default: 24)

    Returns:
        True if the backup is stale (older than max_age_hours), False otherwise
    """
    import os
    from datetime import datetime, timedelta

    if not os.path.exists(backup_path):
        return True  # No backup = considered stale

    mtime = os.path.getmtime(backup_path)
    backup_time = datetime.fromtimestamp(mtime)
    age = datetime.now() - backup_time

    return age > timedelta(hours=max_age_hours)
