"""
Anonymizer Tool - PII Redaction for Scout

Deterministic PII anonymization for usernames and paths.
Based on DataClaw's anonymizer with audit logging and configuration support.

Usage:
    from scout.tools import AnonymizerTool

    tool = AnonymizerTool()
    result = tool.run({"mode": "text", "data": "Working in /Users/john/project"})
    print(result["result"])  # "Working in /user_hash/project"

Configuration (via scout config):
    anonymizer:
        extra_usernames:  # List of additional usernames to anonymize
            - github_handle
            - discord_name
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from scout.audit import AuditLog

logger = logging.getLogger(__name__)


# =============================================================================
# Internal Anonymizer Class (Adapted from DataClaw)
# =============================================================================

def _hash_username(username: str) -> str:
    """Hash username using SHA256, return first 8 chars with prefix."""
    return "user_" + hashlib.sha256(username.encode()).hexdigest()[:8]


def _detect_home_dir() -> tuple[str, str]:
    """Detect the user's home directory and username."""
    home = os.path.expanduser("~")
    username = os.path.basename(home)
    return home, username


def anonymize_path(
    path: str,
    username: str,
    username_hash: str,
    home: str | None = None,
) -> str:
    """Strip a path to project-relative and hash the username."""
    if not path:
        return path

    if home is None:
        home = os.path.expanduser("~")

    prefixes: set[str] = set()
    for base in (f"/Users/{username}", f"/home/{username}", home):
        for subdir in ("Documents", "Downloads", "Desktop"):
            prefixes.add(f"{base}/{subdir}/")
        prefixes.add(f"{base}/")

    # Try longest prefixes first (subdirectory matches before bare home)
    home_patterns = sorted(prefixes, key=len, reverse=True)

    for prefix in home_patterns:
        if path.startswith(prefix):
            rest = path[len(prefix):]
            if (
                "/Documents/" in prefix
                or "/Downloads/" in prefix
                or "/Desktop/" in prefix
            ):
                return rest
            return f"{username_hash}/{rest}"

    path = path.replace(f"/Users/{username}/", f"/{username_hash}/")
    path = path.replace(f"/home/{username}/", f"/{username_hash}/")

    return path


def anonymize_text(text: str, username: str, username_hash: str) -> str:
    """Replace username references with hash in text."""
    if not text or not username:
        return text

    escaped = re.escape(username)

    # Replace /Users/<username> and /home/<username>
    text = re.sub(
        rf"/Users/{escaped}(?=/|[^a-zA-Z0-9_-]|$)", f"/{username_hash}", text
    )
    text = re.sub(
        rf"/home/{escaped}(?=/|[^a-zA-Z0-9_-]|$)", f"/{username_hash}", text
    )

    # Catch hyphen-encoded paths: -Users-peteromalley- or -Users-peteromalley/
    text = re.sub(rf"-Users-{escaped}(?=-|/|$)", f"-Users-{username_hash}", text)
    text = re.sub(rf"-home-{escaped}(?=-|/|$)", f"-home-{username_hash}", text)

    # Catch temp paths like /private/tmp/claude-501/-Users-peteromalley/
    text = re.sub(
        rf"claude-\d+/-Users-{escaped}", f"claude-XXX/-Users-{username_hash}", text
    )

    # Final pass: replace bare username in remaining contexts (ls output, prose, etc.)
    # Only if username is >= 4 chars to avoid false positives
    if len(username) >= 4:
        text = re.sub(rf"\b{escaped}\b", username_hash, text)

    return text


def _replace_username(text: str, username: str, username_hash: str) -> str:
    """Replace username with hash (case-insensitive)."""
    if not text or not username or len(username) < 3:
        return text
    escaped = re.escape(username)
    text = re.sub(escaped, username_hash, text, flags=re.IGNORECASE)
    return text


class Anonymizer:
    """
    Stateful anonymizer that consistently hashes usernames.

    Detects the current user's home directory and username, then provides
    methods to anonymize paths and text content.
    """

    def __init__(self, extra_usernames: Optional[List[str]] = None):
        """
        Initialize the anonymizer.

        Args:
            extra_usernames: Additional usernames to anonymize (e.g., GitHub handles)
        """
        self.home, self.username = _detect_home_dir()
        self.username_hash = _hash_username(self.username)

        # Store extra usernames with their hashes
        self._extra_usernames: List[str] = []
        self._extra_hashes: List[str] = []

        for name in (extra_usernames or []):
            name = name.strip()
            if name and name != self.username:
                self._extra_usernames.append(name)
                self._extra_hashes.append(_hash_username(name))

    @property
    def extra_usernames(self) -> List[str]:
        """Return list of configured extra usernames."""
        return self._extra_usernames.copy()

    def path(self, file_path: str) -> str:
        """
        Anonymize a file path.

        Args:
            file_path: The path to anonymize

        Returns:
            Anonymized path with username replaced by hash
        """
        result = anonymize_path(
            file_path, self.username, self.username_hash, self.home
        )
        result = anonymize_text(result, self.username, self.username_hash)

        # Apply extra username replacements
        for name, hashed in zip(self._extra_usernames, self._extra_hashes):
            result = _replace_username(result, name, hashed)

        return result

    def text(self, content: str) -> str:
        """
        Anonymize text content.

        Args:
            content: The text to anonymize

        Returns:
            Anonymized text with all username references replaced
        """
        result = anonymize_text(content, self.username, self.username_hash)

        # Apply extra username replacements
        for name, hashed in zip(self._extra_usernames, self._extra_hashes):
            result = _replace_username(result, name, hashed)

        return result


# =============================================================================
# AnonymizerTool - Public Interface with Audit Logging
# =============================================================================

class AnonymizerTool:
    """
    Public tool interface for PII anonymization with audit logging.

    Provides a standardized interface for anonymizing usernames and paths,
    with full audit logging for compliance and debugging.
    """

    name: str = "anonymizer"
    description: str = (
        "Anonymize PII (usernames, paths) in text or file paths. "
        "Replaces user references with deterministic hashes."
    )

    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the anonymizer tool.

        Args:
            config: Optional ScoutConfig object. If not provided, uses defaults.
        """
        self._config = config
        self._audit: Optional[AuditLog] = None

        # Load extra usernames from config
        self._extra_usernames: List[str] = []
        if config is not None:
            try:
                anon_config = config.get("anonymizer") or {}
                self._extra_usernames = anon_config.get("extra_usernames", [])
            except Exception:
                pass  # Config not available, use defaults

        self._default_anonymizer = Anonymizer(extra_usernames=self._extra_usernames)

    @property
    def audit(self) -> AuditLog:
        """Lazy-load audit log."""
        if self._audit is None:
            self._audit = AuditLog()
        return self._audit

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the anonymizer tool.

        Args:
            input_data: Dictionary with keys:
                - mode: "text" or "path" (required)
                - data: string to anonymize (required)
                - extra_usernames: optional list of extra usernames

        Returns:
            Dictionary with:
                - result: anonymized string
                - metadata: dict with replacements count and extra_usernames used

        Raises:
            ValueError: If required fields are missing or mode is invalid
        """
        mode = input_data.get("mode")
        data = input_data.get("data")

        if mode is None or data is None:
            raise ValueError("Both 'mode' and 'data' are required.")

        if mode not in ("text", "path"):
            raise ValueError("mode must be 'text' or 'path'.")

        # Determine which anonymizer to use
        extra_usernames = input_data.get("extra_usernames")
        if extra_usernames:
            # Combine with default list (dedupe)
            combined = list(
                set(self._default_anonymizer.extra_usernames + extra_usernames)
            )
            anonymizer = Anonymizer(extra_usernames=combined)
            used_extra = extra_usernames
        else:
            anonymizer = self._default_anonymizer
            used_extra = anonymizer.extra_usernames

        # Perform anonymization
        if mode == "text":
            result = anonymizer.text(data)
        else:
            result = anonymizer.path(data)

        # Count replacements (simple heuristic: check if result differs from input)
        replacements = 0 if result == data else 1

        # Log the invocation
        try:
            self.audit.log(
                "tool_invocation",
                tool=self.name,
                mode=mode,
                input_length=len(data),
                output_length=len(result),
                replacements=replacements,
                extra_usernames_used=used_extra,
            )
        except Exception as e:
            logger.warning("Failed to log anonymizer invocation: %s", e)

        return {
            "result": result,
            "metadata": {
                "replacements": replacements,
                "extra_usernames_used": used_extra,
            },
        }


# =============================================================================
# Tool Registry
# =============================================================================

_TOOLS: Dict[str, type] = {
    "anonymizer": AnonymizerTool,
}


def get_tool(name: str) -> Optional[type]:
    """Get a tool class by name."""
    return _TOOLS.get(name)


def list_tools() -> List[str]:
    """List all available tool names."""
    return list(_TOOLS.keys())
