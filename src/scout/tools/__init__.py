from __future__ import annotations
"""
Scout Tools - Modularized Tool Handlers

This package contains extracted tool implementations organized by domain:
- git: Git operation tools (status, diff, log, branch, commit, etc.)
- github: GitHub API tools (PR, issues, etc.)
- llm: LLM-powered tools (plan, nav, query, roast, etc.)
- validation: Linting and validation tools
- file_ops: File read/write/delete operations
- batch: Batch processing tools
- admin: Admin and debug tools (help, status, restart)

Each submodule exports tool functions that can be called directly.
"""

import functools
import time
from typing import Any, Callable

from scout.audit import get_audit
from scout.cache import simple_cache as _simple_cache

from .anonymizer import Anonymizer, AnonymizerTool, get_tool, list_tools


# Re-export simple_cache from scout.cache
simple_cache = _simple_cache


def log_tool_invocation(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that logs tool invocations to the audit log.

    Logs execution time, success/failure status, and any errors.
    Works with both sync and async functions.
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        audit = get_audit()
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            audit.log(
                "tool_invocation",
                tool=func.__name__,
                duration_ms=int(duration * 1000),
                success=True,
            )
            return result
        except Exception as e:
            duration = time.time() - start
            audit.log(
                "tool_invocation",
                tool=func.__name__,
                duration_ms=int(duration * 1000),
                success=False,
                error=str(e),
            )
            raise

    # Handle sync functions
    @functools.wraps(func)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        audit = get_audit()
        start = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start
            audit.log(
                "tool_invocation",
                tool=func.__name__,
                duration_ms=int(duration * 1000),
                success=True,
            )
            return result
        except Exception as e:
            duration = time.time() - start
            audit.log(
                "tool_invocation",
                tool=func.__name__,
                duration_ms=int(duration * 1000),
                success=False,
                error=str(e),
            )
            raise

    # Return appropriate wrapper based on function type
    import inspect

    if inspect.iscoroutinefunction(func):
        return wrapper
    else:
        return sync_wrapper


# =============================================================================
# Tool Metadata for Execution Engine (Phase 3)
# =============================================================================

from enum import Enum
from typing import TYPE_CHECKING, Optional


class CostTier(str, Enum):
    """Cost tier for tool budgeting."""

    FREE = "free"
    CHEAP = "cheap"
    STANDARD = "standard"
    EXPENSIVE = "expensive"


# Extended metadata for execution engine - maps tool names to cost, validators, timeouts
TOOL_METADATA: dict[str, dict] = {
    # Tools registered in executor.py initialize_executor()
    "lint": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 30,
        "retryable": False,
    },
    "write_file": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 10,
        "retryable": False,
    },
    "edit_file": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 10,
        "retryable": False,
    },
    "delete_file": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
    },
    "run_command": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 60,
        "retryable": False,
    },
    "run": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 60,
        "retryable": False,
    },
    "git_commit": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 10,
        "retryable": False,
    },
    "git_add": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
    },
    "shell": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 30,
        "retryable": False,
    },
    "nav": {
        "cost_tier": CostTier.STANDARD,
        "estimated_cost": 0.01,
        "validators": ["schema", "content_type", "length"],
        "timeout_seconds": 60,
        "retryable": True,
    },
    "query": {
        "cost_tier": CostTier.CHEAP,
        "estimated_cost": 0.001,
        "validators": ["schema", "content_type", "length"],
        "timeout_seconds": 30,
        "retryable": True,
    },
    "doc_sync": {
        "cost_tier": CostTier.EXPENSIVE,
        "estimated_cost": 0.05,
        "validators": ["schema", "content_type", "length", "confidence"],
        "timeout_seconds": 300,
        "retryable": True,
    },
    "validate_module": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 30,
        "retryable": False,
    },
    # Additional tools
    "index": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
        "skip_validation": True,
    },
    "export": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 30,
        "retryable": False,
    },
    "sync": {
        "cost_tier": CostTier.EXPENSIVE,
        "estimated_cost": 0.05,
        "validators": ["schema", "content_type", "length", "confidence"],
        "timeout_seconds": 300,
        "retryable": True,
    },
    "brief": {
        "cost_tier": CostTier.STANDARD,
        "estimated_cost": 0.02,
        "validators": ["schema", "content_type", "length", "confidence"],
        "timeout_seconds": 120,
        "retryable": True,
    },
    "status": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
        "skip_validation": True,
    },
    "branch_status": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
    },
    "help": {
        "cost_tier": CostTier.STANDARD,
        "estimated_cost": 0.01,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 60,
        "retryable": True,
        "skip_validation": True,
    },
    # Phase 4: Added metadata for all registered tools
    # Git tools
    "git_status": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
        "skip_validation": True,
    },
    "git_diff": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
        "skip_validation": True,
    },
    "git_log": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
        "skip_validation": True,
    },
    "git_branch": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
        "skip_validation": True,
    },
    "git_show": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
        "skip_validation": True,
    },
    # File operations
    "read_file": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
        "skip_validation": True,
    },
    # Batch operations
    "batch": {
        "cost_tier": CostTier.STANDARD,
        "estimated_cost": 0.02,
        "validators": ["schema"],
        "timeout_seconds": 300,
        "retryable": False,
    },
    # LLM tools
    "plan": {
        "cost_tier": CostTier.STANDARD,
        "estimated_cost": 0.02,
        "validators": ["schema", "content_type", "length"],
        "timeout_seconds": 120,
        "retryable": True,
    },
    "roast": {
        "cost_tier": CostTier.STANDARD,
        "estimated_cost": 0.02,
        "validators": ["schema", "content_type", "length"],
        "timeout_seconds": 60,
        "retryable": True,
    },
    "propose_tool": {
        "cost_tier": CostTier.STANDARD,
        "estimated_cost": 0.02,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 60,
        "retryable": True,
    },
    # Doc generation
    "generate_docs": {
        "cost_tier": CostTier.EXPENSIVE,
        "estimated_cost": 0.05,
        "validators": ["schema", "content_type", "length", "confidence"],
        "timeout_seconds": 300,
        "retryable": True,
    },
    "update_docs": {
        "cost_tier": CostTier.EXPENSIVE,
        "estimated_cost": 0.05,
        "validators": ["schema", "content_type", "length", "confidence"],
        "timeout_seconds": 300,
        "retryable": True,
    },
    # Validation tools
    "env": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
        "skip_validation": True,
    },
    "function_info": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 10,
        "retryable": False,
        "skip_validation": True,
    },
    # Admin tools
    "restart_server": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 30,
        "retryable": False,
    },
    "grep": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 30,
        "retryable": False,
        "skip_validation": True,
    },
    "ci_guard": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 60,
        "retryable": False,
        "skip_validation": True,
    },
    "add_decorator": {
        "cost_tier": CostTier.STANDARD,
        "estimated_cost": 0.01,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 60,
        "retryable": False,
    },
    "modify_decorator": {
        "cost_tier": CostTier.STANDARD,
        "estimated_cost": 0.01,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 60,
        "retryable": False,
    },
    "batch_add_decorator": {
        "cost_tier": CostTier.EXPENSIVE,
        "estimated_cost": 0.05,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 300,
        "retryable": False,
    },
    "batch_modify_decorator": {
        "cost_tier": CostTier.EXPENSIVE,
        "estimated_cost": 0.05,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 300,
        "retryable": False,
    },
    "search_docs": {
        "cost_tier": CostTier.CHEAP,
        "estimated_cost": 0.001,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 30,
        "retryable": True,
    },
    "reindex_docs": {
        "cost_tier": CostTier.FREE,
        "estimated_cost": 0.0,
        "validators": ["schema"],
        "timeout_seconds": 60,
        "retryable": False,
    },
    # GitHub tools
    "pr": {
        "cost_tier": CostTier.CHEAP,
        "estimated_cost": 0.001,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 30,
        "retryable": True,
    },
    "pr_info": {
        "cost_tier": CostTier.CHEAP,
        "estimated_cost": 0.001,
        "validators": ["schema", "content_type"],
        "timeout_seconds": 30,
        "retryable": True,
    },
}


def get_tool_metadata(name: str) -> Optional[dict]:
    """Get extended metadata for a tool (cost tier, validators, timeout, retryable).

    Returns None if tool not found in metadata.
    """
    return TOOL_METADATA.get(name)

# Re-export for backwards compatibility
from scout.tools.git import (
    scout_git_add,
    scout_git_branch,
    scout_git_commit,
    scout_git_diff,
    scout_git_log,
    scout_git_show,
    scout_git_status,
)

from scout.tools.github import scout_pr, scout_pr_info

# LLM tools - TODO: Extract in Phase 4
# from scout.tools.llm import (
#     scout_brief,
#     scout_nav,
#     scout_plan,
#     scout_propose_tool,
#     scout_query,
#     scout_roast,
# )

from scout.tools.validation import (
    scout_env,
    scout_function_info,
    scout_lint,
    scout_validate_module,
)

from scout.tools.file_ops import (
    scout_delete_with_review,
    scout_edit,
    scout_read_file,
    scout_write_with_review,
)

from scout.tools.batch import scout_batch, scout_run

# Doc gen tools - TODO: Extract later
# from scout.tools.doc_gen import (
#     scout_generate_docs,
#     scout_update_docs,
# )

# Admin tools - TODO: Extract later
# from scout.tools.admin import (
#     scout_add_decorator,
#     scout_batch_add_decorator,
#     scout_batch_modify_decorator,
#     scout_ci_guard,
#     scout_doc_sync,
#     scout_grep,
#     scout_help,
#     scout_index,
#     scout_modify_decorator,
#     scout_reindex_docs,
#     scout_restart_server,
#     scout_search_docs,
#     scout_shell,
#     scout_status,
# )

# Analysis tools - TODO: Extract in Phase 7 (hotspots)
# from scout.tools.analysis.hotspots import scout_hotspots

# For backward compatibility, also expose from top level
__all__ = [
    # Git tools
    "scout_git_add",
    "scout_git_branch",
    "scout_git_commit",
    "scout_git_diff",
    "scout_git_log",
    "scout_git_show",
    "scout_git_status",
    # GitHub tools
    "scout_pr",
    "scout_pr_info",
    # Validation tools
    "scout_env",
    "scout_function_info",
    "scout_lint",
    "scout_validate_module",
    # File ops tools
    "scout_delete_with_review",
    "scout_edit",
    "scout_read_file",
    "scout_write_with_review",
    # Batch tools
    "scout_batch",
    "scout_run",
    # Anonymizer tools
    "Anonymizer",
    "AnonymizerTool",
    "get_tool",
    "list_tools",
    # Decorators
    "simple_cache",
    "log_tool_invocation",
]


def get_tools() -> dict:
    """
    Return a dictionary of all available Scout tools.

    This function is used by answer_help_async to provide tool documentation.
    Returns a dict mapping tool names to their descriptions.
    """
    return {
        # Git tools
        "scout_git_add": "Stage files for commit",
        "scout_git_branch": "List git branches",
        "scout_git_commit": "Create a git commit",
        "scout_git_diff": "Show git diff",
        "scout_git_log": "Show git commit history",
        "scout_git_show": "Show git commit details",
        "scout_git_status": "Show git working tree status",
        # GitHub tools
        "scout_pr": "Create a GitHub pull request",
        "scout_pr_info": "Get GitHub pull request info",
        # Validation tools
        "scout_env": "Check environment variables",
        "scout_function_info": "Inspect function signatures",
        "scout_lint": "Run Scout linter on Python files",
        "scout_validate_module": "Run type checking and tests",
        # File ops tools
        "scout_delete_with_review": "Delete files with review",
        "scout_edit": "Edit files with AI assistance (supports dry-run)",
        "scout_read_file": "Read file contents",
        "scout_write_with_review": "Write files with review",
        # Batch tools
        "scout_batch": "Run batch operations",
        "scout_run": "Run Python modules",
    }


def get_valid_tool_names() -> list[str]:
    """Return list of valid tool names."""
    return list(get_tools().keys())


def register_all_tools(mcp: "FastMCP") -> None:
    """
    Register all modularized tools with the given FastMCP instance.

    This function provides a single entry point to register all tools,
    replacing the need to import and register each tool individually.
    """
    # Git tools
    mcp.add_tool(scout_git_add)
    mcp.add_tool(scout_git_branch)
    mcp.add_tool(scout_git_commit)
    mcp.add_tool(scout_git_diff)
    mcp.add_tool(scout_git_log)
    mcp.add_tool(scout_git_show)
    mcp.add_tool(scout_git_status)

    # GitHub tools
    mcp.add_tool(scout_pr)
    mcp.add_tool(scout_pr_info)

    # Validation tools
    mcp.add_tool(scout_env)
    mcp.add_tool(scout_function_info)
    mcp.add_tool(scout_lint)
    mcp.add_tool(scout_validate_module)

    # File ops tools
    mcp.add_tool(scout_delete_with_review)
    mcp.add_tool(scout_edit)
    mcp.add_tool(scout_read_file)
    mcp.add_tool(scout_write_with_review)

    # Batch tools
    mcp.add_tool(scout_batch)
    mcp.add_tool(scout_run)

    # Analysis tools
    mcp.add_tool(scout_hotspots)

def get_tools_minimal():
    return [{"name": t, "desc": ""} for t in get_tools()]
