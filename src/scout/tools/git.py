from __future__ import annotations
"""
Git Operation Tools for Scout.

This module contains extracted git operation tools:
- scout_git_status: Show git status
- scout_git_diff: Show git diff
- scout_git_log: Show commit history
- scout_git_branch: List branches
- scout_git_show: Show commit details
- scout_git_add: Stage files
- scout_git_commit: Create commits
"""

import subprocess
from pathlib import Path
from typing import Any, Callable, TypeVar

from scout.tool_output import ToolOutput

# Simple passthrough decorator (replaces @simple_cache from vivarium)
# Cache functionality can be added later if needed
def simple_cache(ttl_seconds: int = 60, dependencies: list = None):
    """Passthrough decorator - cache functionality not yet implemented."""
    def decorator(func):
        return func
    return decorator

# Simple passthrough decorator (replaces @_log_invocation)
F = TypeVar('F', bound=Callable)

def _log_invocation(func: F) -> F:
    """Passthrough decorator - logs handled by audit.py."""
    return func

from scout.config.paths import REPO_ROOT, VENV_PYTHON


def _run_command(
    cmd: list[str],
    timeout: int = 30,
    cwd: Path = None,
    capture_output: bool = True,
    env: dict = None,
) -> subprocess.CompletedProcess:
    """Shared subprocess.run wrapper."""
    import os

    if cwd is None:
        cwd = REPO_ROOT
    if env is None:
        env = os.environ.copy()
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=capture_output,
        text=True,
        timeout=timeout,
        env=env,
    )


# =============================================================================
# Git Status Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@_log_invocation
def scout_git_status(
    full: bool = False,
    json_output: bool = False,
    summarize: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Show LOCAL git status - working tree status with modified, staged, and untracked files.

    IMPORTANT: This tool shows the LOCAL git repository status (files changed, staged, etc.).
    DO NOT use for GitHub issues - use scout_pr or scout_pr_info for GitHub PRs/issues.

    Use this tool when:
    - User asks about "git status", "modified files", "staged changes", "untracked files"
    - User wants to see what's changed in the local working directory

    Do NOT use when:
    - User asks about GitHub issues, PRs, or repo issues (use scout_pr)
    - User asks about "todo" files, quest logs, or task tracking
    - User asks about what to work on next

    Use this tool to check the current state of the repository.
    - Returns RAW git output by default (deterministic, no LLM)
    - Use summarize=True to get an LLM-generated summary (uses MiniMax)
    - Use full=True to get raw git output
    - Use json_output=True for structured data

    Example: `scout_git_status()` - returns raw git status
    Example: `scout_git_status(summarize=True)` - returns LLM summary

    Returns:
        ToolOutput with tool_name="git_status", content=result_string, cost_usd=0.0
    """
    cmd = [VENV_PYTHON, "-m", "vivarium.scout.cli.git", "status"]

    if full:
        cmd.append("--full")
    if json_output:
        cmd.append("--json")
    if summarize:
        cmd.append("--summarize")

    try:
        result = _run_command(cmd)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="git_status",
            content=content,
            cost_usd=0.0,
            metadata={"full": full, "json_output": json_output, "summarize": summarize},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="git_status",
            content=f"Error running scout-git status: {e}",
            cost_usd=0.0,
            metadata={"full": full, "json_output": json_output, "summarize": summarize},
        )


# =============================================================================
# Git Diff Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@_log_invocation
def scout_git_diff(
    cached: bool = False,
    full: bool = False,
    json_output: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Show git diff - changes between commits or between commit and working tree.

    Use this tool to see what has changed.
    - Use cached=True to show staged changes
    - Returns a concise summary by default
    - Use full=True to get raw diff output

    Example: `scout_git_diff()` - shows unstaged changes
    Example: `scout_git_diff(cached=True)` - shows staged changes

    Returns:
        ToolOutput with tool_name="git_diff", content=result_string, cost_usd=0.0
    """
    cmd = [VENV_PYTHON, "-m", "vivarium.scout.cli.git", "diff"]

    if cached:
        cmd.append("--cached")
    if full:
        cmd.append("--full")
    if json_output:
        cmd.append("--json")

    try:
        result = _run_command(cmd)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="git_diff",
            content=content,
            cost_usd=0.0,
            metadata={"cached": cached, "full": full, "json_output": json_output},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="git_diff",
            content=f"Error running scout-git diff: {e}",
            cost_usd=0.0,
            metadata={"cached": cached, "full": full, "json_output": json_output},
        )


# =============================================================================
# Git Log Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@_log_invocation
def scout_git_log(
    max_count: int = 10,
    full: bool = False,
    json_output: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Show git log - commit history.

    Use this tool to see recent commits.
    - max_count controls number of commits (default: 10)
    - Returns a concise summary by default

    Example: `scout_git_log()` - shows last 10 commits
    Example: `scout_git_log(max_count=5)` - shows last 5 commits

    Returns:
        ToolOutput with tool_name="git_log", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "vivarium.scout.cli.git",
        "log",
        f"--max-count={max_count}",
    ]

    if full:
        cmd.append("--full")
    if json_output:
        cmd.append("--json")

    try:
        result = _run_command(cmd)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="git_log",
            content=content,
            cost_usd=0.0,
            metadata={"max_count": max_count, "full": full, "json_output": json_output},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="git_log",
            content=f"Error running scout-git log: {e}",
            cost_usd=0.0,
            metadata={"max_count": max_count, "full": full, "json_output": json_output},
        )


# =============================================================================
# Git Branch Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@_log_invocation
def scout_git_branch(
    full: bool = False,
    json_output: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Show git branch - list branches.

    Use this tool to see available branches and current branch.
    - Returns a concise summary by default

    Example: `scout_git_branch()` - shows branches summary

    Returns:
        ToolOutput with tool_name="git_branch", content=result_string, cost_usd=0.0
    """
    cmd = [VENV_PYTHON, "-m", "vivarium.scout.cli.git", "branch"]

    if full:
        cmd.append("--full")
    if json_output:
        cmd.append("--json")

    try:
        result = _run_command(cmd)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="git_branch",
            content=content,
            cost_usd=0.0,
            metadata={"full": full, "json_output": json_output},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="git_branch",
            content=f"Error running scout-git branch: {e}",
            cost_usd=0.0,
            metadata={"full": full, "json_output": json_output},
        )


# =============================================================================
# Git Show Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@_log_invocation
def scout_git_show(
    ref: str = "HEAD",
    full: bool = False,
    json_output: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Show git show - details of a specific commit.

    Use this tool to see commit details (hash, author, date, message, stats).
    - ref is the commit reference (default: HEAD)

    Example: `scout_git_show()` - shows HEAD
    Example: `scout_git_show(ref="HEAD~3")` - shows 3 commits ago

    Returns:
        ToolOutput with tool_name="git_show", content=result_string, cost_usd=0.0
    """
    cmd = [VENV_PYTHON, "-m", "vivarium.scout.cli.git", "show", ref]

    if full:
        cmd.append("--full")
    if json_output:
        cmd.append("--json")

    try:
        result = _run_command(cmd)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="git_show",
            content=content,
            cost_usd=0.0,
            metadata={"ref": ref, "full": full, "json_output": json_output},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="git_show",
            content=f"Error running scout-git show: {e}",
            cost_usd=0.0,
            metadata={"ref": ref, "full": full, "json_output": json_output},
        )


# =============================================================================
# Git Add Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@_log_invocation
def scout_git_add(
    paths: list[str] = None,
    all_files: bool = False,
    patch: bool = False,
    json_output: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Stage files for commit.

    Use this tool to stage files before committing.
    - Use paths to specify files to stage (default: all modified files)
    - Use all_files=True to stage all files
    - Use patch=True for interactive staging

    Example: `scout_git_add(paths=[".cursor/hooks/before-mcp.js"])`
    Example: `scout_git_add(all_files=True)`

    Returns:
        ToolOutput with tool_name="git_add", content=result_string, cost_usd=0.0
    """
    cmd = [VENV_PYTHON, "-m", "vivarium.scout.cli.git", "add"]

    if paths:
        cmd.extend(paths)
    if all_files:
        cmd.append("--all")
    if patch:
        cmd.append("--patch")
    if json_output:
        cmd.append("--json")

    try:
        result = _run_command(cmd)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="git_add",
            content=content,
            cost_usd=0.0,
            metadata={"paths": paths, "all_files": all_files, "patch": patch, "json_output": json_output},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="git_add",
            content=f"Error running scout-git add: {e}",
            cost_usd=0.0,
            metadata={"paths": paths, "all_files": all_files, "patch": patch, "json_output": json_output},
        )


# =============================================================================
# Git Commit Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@_log_invocation
def scout_git_commit(
    message: str,
    allow_empty: bool = False,
    json_output: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Create a git commit with the given message.

    Use this tool to commit staged changes.
    - message is required and should describe what changed and why
    - Use allow_empty=True to create a commit without staged changes

    Example: `scout_git_commit(message="feat: add new feature")`

    Returns:
        ToolOutput with tool_name="git_commit", content=result_string, cost_usd=0.0
    """
    cmd = [VENV_PYTHON, "-m", "vivarium.scout.cli.git", "commit", "-m", message]

    if allow_empty:
        cmd.append("--allow-empty")
    if json_output:
        cmd.append("--json")

    try:
        result = _run_command(cmd)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="git_commit",
            content=content,
            cost_usd=0.0,
            metadata={"message": message, "allow_empty": allow_empty, "json_output": json_output},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="git_commit",
            content=f"Error running scout-git commit: {e}",
            cost_usd=0.0,
            metadata={"message": message, "allow_empty": allow_empty, "json_output": json_output},
        )


# =============================================================================
# Registration Helper
# =============================================================================
