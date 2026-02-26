from __future__ import annotations
"""
Admin and Debug Tools for Scout MCP Server.

This module contains extracted admin and debug tools:
- scout_help: List available Scout tools
- scout_status: Show Scout status
- scout_restart_server: Restart the Scout MCP server
- scout_shell: Execute whitelisted shell commands
- scout_grep: Search file contents
- scout_doc_sync: Documentation sync commands
- scout_index: Local code search commands
- scout_ci_guard: CI validation
- scout_add_decorator: Add decorators to tools
- scout_modify_decorator: Modify existing decorators
- scout_batch_add_decorator: Batch add decorators
- scout_batch_modify_decorator: Batch modify decorators
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from scout.tools import log_tool_invocation
from scout.tool_output import ToolOutput

# Shared configuration
VENV_PYTHON = "/Users/vivariumenv1/GITHUBS/scout/venv/bin/python"
REPO_ROOT = Path("/Users/vivariumenv1/GITHUBS/scout")


def _run_command(
    cmd: list[str],
    timeout: int = 30,
    cwd: Path = None,
) -> subprocess.CompletedProcess:
    """Shared subprocess.run wrapper."""
    if cwd is None:
        cwd = REPO_ROOT
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=os.environ.copy(),
    )


# =============================================================================
# Scout Help Tool
# =============================================================================


@log_tool_invocation
def scout_help(ctx: Any = None) -> ToolOutput:
    """
    List all available Scout tools with their descriptions.

    Call this at the start of a session to discover what tools are available and what
    they do.

    Example: `scout_help()`

    Returns:
        ToolOutput with tool_name="help", content=help_text, cost_usd=0.0
    """
    # This is implemented directly in the MCP server for now
    # Could be extracted to CLI later
    content = """# Scout MCP Tools

## Git Tools
- scout_git_status: Show git status
- scout_git_diff: Show git diff
- scout_git_log: Show commit history
- scout_git_branch: List branches
- scout_git_show: Show commit details
- scout_git_add: Stage files
- scout_git_commit: Create commits

## GitHub Tools
- scout_pr_info: Fetch PR metadata
- scout_pr: Create GitHub PRs

## LLM Tools
- scout_plan: Generate implementation plans
- scout_nav: Search codebase
- scout_query: Natural language search
- scout_roast: Efficiency reports
- scout_brief: Code briefings
- scout_propose_tool: Propose new tools

## Validation Tools
- scout_lint: Run linter
- scout_validate_module: Type checking and tests
- scout_env: Check environment variables
- scout_function_info: Inspect functions

## File Operations
- scout_read_file: Read files
- scout_write_with_review: Write with LLM review
- scout_delete_with_review: Delete with LLM review

## Batch Tools
- scout_batch: Run multiple tasks
- scout_run: Run Python modules

## Admin Tools
- scout_help: This help
- scout_status: Show status
- scout_restart_server: Restart server
- scout_shell: Execute shell commands
- scout_grep: Search content
- scout_doc_sync: Documentation sync
- scout_index: Code search index
- scout_ci_guard: CI validation
"""
    return ToolOutput.from_content(
        tool_name="help",
        content=content,
        cost_usd=0.0,
        metadata={},
    )


# =============================================================================
# Scout Status Tool
# =============================================================================


@log_tool_invocation
def scout_status(
    json_output: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Show Scout status - doc-sync status, dependency graph, drafts, spend.

    Use this tool to check the current state of Scout.

    Example: `scout_status()`

    Returns:
        ToolOutput with tool_name="status", content=result_string, cost_usd=0.0
    """
    cmd = [VENV_PYTHON, "-m", "scout.cli.status"]
    if json_output:
        cmd.append("--json")

    try:
        result = _run_command(cmd, timeout=60)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="status",
            content=content,
            cost_usd=0.0,
            metadata={"json_output": json_output},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="status",
            content=f"Error getting status: {e}",
            cost_usd=0.0,
            metadata={"json_output": json_output, "error": True},
        )


# =============================================================================
# Scout Restart Server Tool
# =============================================================================


@log_tool_invocation
async def scout_restart_server(
    ctx: Any = None,
) -> ToolOutput:
    """
    Restart the Scout MCP server.

    Example: `scout_restart_server()`

    Returns:
        ToolOutput with tool_name="restart_server", content=result_string, cost_usd=0.0
    """
    # This would need to be implemented to restart the actual server process
    # For now, return a message
    content = json.dumps(
        {
            "status": "info",
            "message": "Server restart requested. This requires external process management.",
        }
    )
    return ToolOutput.from_content(
        tool_name="restart_server",
        content=content,
        cost_usd=0.0,
        metadata={},
    )


# =============================================================================
# Scout Shell Tool
# =============================================================================


@log_tool_invocation
async def scout_shell(
    command: str,
    timeout: int = 30,
    proposal_reason: str = "",
    ctx: Any = None,
) -> ToolOutput:
    """
    Execute a whitelisted shell command with audit logging.

    This tool provides a safe alternative to raw shell execution by:
    - Validating commands against a configurable whitelist
    - Blocking dangerous operations
    - Logging all executions to the audit trail

    Example: `scout_shell(command="git status")`

    Returns:
        ToolOutput with tool_name="shell", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "scout.cli.scout",
        "shell",
        "--command",
        command,
    ]

    if timeout:
        cmd.extend(["--timeout", str(timeout)])
    if proposal_reason:
        cmd.extend(["--proposal-reason", proposal_reason])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout + 5)

        output = stdout.decode()
        if stderr:
            output += "\n" + stderr.decode()

        content = json.dumps(
            {
                "status": "success" if proc.returncode == 0 else "error",
                "output": output,
                "returncode": proc.returncode,
            }
        )
        return ToolOutput.from_content(
            tool_name="shell",
            content=content,
            cost_usd=0.0,
            metadata={"command": command, "timeout": timeout},
        )
    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="shell",
            content=json.dumps({"status": "error", "message": f"Shell command timed out after {timeout}s"}),
            cost_usd=0.0,
            metadata={"command": command, "timeout": timeout, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="shell",
            content=json.dumps({"status": "error", "message": str(e)}),
            cost_usd=0.0,
            metadata={"command": command, "timeout": timeout, "error": True},
        )


# =============================================================================
# Scout Grep Tool
# =============================================================================


@log_tool_invocation
async def scout_grep(
    pattern: str,
    path: str = ".",
    glob: str = "**/*",
    context_lines: int = 0,
    verbosity: int = 1,
    include_binary: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Search file contents using regex with governed path access.

    Use this tool to find text patterns in code files.
    - Respects path scoping
    - Respects off-limits files
    - Logs all searches to audit

    Example: `scout_grep(pattern="def.*test", path="vivarium/scout", glob="*.py")`

    Returns:
        ToolOutput with tool_name="grep", content=grep_output, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "scout.cli.scout",
        "grep",
        "--pattern",
        pattern,
    ]

    if path and path != ".":
        cmd.extend(["--path", path])
    if glob and glob != "**/*":
        cmd.extend(["--glob", glob])
    if context_lines:
        cmd.extend(["--context", str(context_lines)])
    if verbosity != 1:
        cmd.extend(["--verbosity", str(verbosity)])
    if include_binary:
        cmd.append("--include-binary")

    try:
        result = _run_command(cmd, timeout=120)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no matches)"
        return ToolOutput.from_content(
            tool_name="grep",
            content=content,
            cost_usd=0.0,
            metadata={"pattern": pattern, "path": path, "glob": glob},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="grep",
            content=f"Error running grep: {e}",
            cost_usd=0.0,
            metadata={"pattern": pattern, "path": path, "glob": glob},
        )


# =============================================================================
# Scout Doc Sync Tool
# =============================================================================


@log_tool_invocation
async def scout_doc_sync(
    command: str = "status",
    target: str = None,
    json_output: bool = True,
    ctx: Any = None,
) -> ToolOutput:
    """
    Run Scout documentation sync commands.

    Use this tool to generate, repair, validate, and export documentation.
    - generate: Generate new documentation
    - repair: Repair stale docs
    - export: Export knowledge graph
    - validate: Validate docs
    - status: Show sync status

    Example: `scout_doc_sync(command="status")`

    Returns:
        ToolOutput with tool_name="doc_sync", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "scout.cli.doc_sync",
        command,
    ]

    if target:
        cmd.extend(["--target", target])
    if json_output:
        cmd.append("--json")

    try:
        result = _run_command(cmd, timeout=180)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="doc_sync",
            content=content,
            cost_usd=0.0,
            metadata={"command": command, "target": target},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="doc_sync",
            content=f"Error running doc_sync: {e}",
            cost_usd=0.0,
            metadata={"command": command, "target": target, "error": True},
        )


# =============================================================================
# Scout Index Tool
# =============================================================================


@log_tool_invocation
async def scout_index(
    command: str = "stats",
    target: str = None,
    json_output: bool = True,
    ctx: Any = None,
) -> ToolOutput:
    """
    Run Scout index commands - local code search (ctags + SQLite).

    Use this tool to build and query the code index.
    - build: Build index from scratch
    - update: Incremental update
    - query: Search symbols and files
    - stats: Show coverage

    Example: `scout_index(command="stats")`

    Returns:
        ToolOutput with tool_name="index", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "scout.cli.index",
        command,
    ]

    if target:
        cmd.extend(["--target", target])
    if json_output:
        cmd.append("--json")

    try:
        result = _run_command(cmd, timeout=180)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="index",
            content=content,
            cost_usd=0.0,
            metadata={"command": command, "target": target},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="index",
            content=f"Error running index: {e}",
            cost_usd=0.0,
            metadata={"command": command, "target": target, "error": True},
        )


# =============================================================================
# Scout CI Guard Tool
# =============================================================================


@log_tool_invocation
async def scout_ci_guard(
    base_branch: str = "origin/main",
    hourly_limit: float = 5.0,
    min_confidence: float = 0.7,
    require_draft_events: bool = False,
    draft_events_hours: int = 24,
    no_placeholders_check: bool = False,
    placeholders_target: str = "vivarium",
    json_output: bool = True,
    ctx: Any = None,
) -> ToolOutput:
    """
    Run CI validation: .tldr.md coverage, draft confidence, hourly spend.

    Use this tool to validate CI requirements before merging.

    Example: `scout_ci_guard()`

    Returns:
        ToolOutput with tool_name="ci_guard", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "vivarium.scout.cli.ci_guard",
    ]

    cmd.extend(["--base-branch", base_branch])
    cmd.extend(["--hourly-limit", str(hourly_limit)])
    cmd.extend(["--min-confidence", str(min_confidence)])

    if require_draft_events:
        cmd.append("--require-draft-events")
    if draft_events_hours != 24:
        cmd.extend(["--draft-events-hours", str(draft_events_hours)])
    if no_placeholders_check:
        cmd.append("--no-placeholders-check")
    if placeholders_target != "vivarium":
        cmd.extend(["--placeholders-target", placeholders_target])
    if json_output:
        cmd.append("--json")

    try:
        result = _run_command(cmd, timeout=120)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="ci_guard",
            content=content,
            cost_usd=0.0,
            metadata={"base_branch": base_branch, "hourly_limit": hourly_limit},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="ci_guard",
            content=f"Error running ci_guard: {e}",
            cost_usd=0.0,
            metadata={"base_branch": base_branch, "error": True},
        )


# =============================================================================
# Decorator Management Tools
# =============================================================================


@log_tool_invocation
async def scout_add_decorator(
    decorator: str,
    target_function: str,
    ctx: Any = None,
) -> ToolOutput:
    """
    Programmatically add a decorator to a Scout tool function.

    Example: `scout_add_decorator(decorator="@log_tool_invocation", target_function="scout_lint")`

    Returns:
        ToolOutput with tool_name="add_decorator", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "scout.cli.bootstrap",
        "add-decorator",
        "--decorator",
        decorator,
        "--target",
        target_function,
    ]

    try:
        result = _run_command(cmd, timeout=60)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="add_decorator",
            content=content,
            cost_usd=0.0,
            metadata={"decorator": decorator, "target_function": target_function},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="add_decorator",
            content=f"Error adding decorator: {e}",
            cost_usd=0.0,
            metadata={"decorator": decorator, "target_function": target_function, "error": True},
        )


@log_tool_invocation
async def scout_modify_decorator(
    decorator: str,
    target_function: str,
    ctx: Any = None,
) -> ToolOutput:
    """
    Programmatically modify an existing decorator on a Scout tool function.

    Example: `scout_modify_decorator(decorator="@simple_cache(ttl_seconds=10)", target_function="scout_lint")`

    Returns:
        ToolOutput with tool_name="modify_decorator", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "scout.cli.bootstrap",
        "modify-decorator",
        "--decorator",
        decorator,
        "--target",
        target_function,
    ]

    try:
        result = _run_command(cmd, timeout=60)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="modify_decorator",
            content=content,
            cost_usd=0.0,
            metadata={"decorator": decorator, "target_function": target_function},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="modify_decorator",
            content=f"Error modifying decorator: {e}",
            cost_usd=0.0,
            metadata={"decorator": decorator, "target_function": target_function, "error": True},
        )


@log_tool_invocation
async def scout_batch_add_decorator(
    decorator: str,
    targets: str,
    ctx: Any = None,
) -> ToolOutput:
    """
    Batch add a decorator to multiple Scout tool functions.

    Example: `scout_batch_add_decorator(decorator="@simple_cache(ttl_seconds=5)", targets="scout_git_*,scout_plan")`

    Returns:
        ToolOutput with tool_name="batch_add_decorator", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "scout.cli.bootstrap",
        "batch-add-decorator",
        "--decorator",
        decorator,
        "--targets",
        targets,
    ]

    try:
        result = _run_command(cmd, timeout=120)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="batch_add_decorator",
            content=content,
            cost_usd=0.0,
            metadata={"decorator": decorator, "targets": targets},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="batch_add_decorator",
            content=f"Error in batch add: {e}",
            cost_usd=0.0,
            metadata={"decorator": decorator, "targets": targets, "error": True},
        )


@log_tool_invocation
async def scout_batch_modify_decorator(
    decorator: str,
    targets: str,
    ctx: Any = None,
) -> ToolOutput:
    """
    Batch modify an existing decorator on multiple Scout tool functions.

    Example: `scout_batch_modify_decorator(decorator="@simple_cache(ttl_seconds=300)", targets="scout_git_*,scout_plan")`

    Returns:
        ToolOutput with tool_name="batch_modify_decorator", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "scout.cli.bootstrap",
        "batch-modify-decorator",
        "--decorator",
        decorator,
        "--targets",
        targets,
    ]

    try:
        result = _run_command(cmd, timeout=120)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="batch_modify_decorator",
            content=content,
            cost_usd=0.0,
            metadata={"decorator": decorator, "targets": targets},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="batch_modify_decorator",
            content=f"Error in batch modify: {e}",
            cost_usd=0.0,
            metadata={"decorator": decorator, "targets": targets, "error": True},
        )


# =============================================================================
# Scout Docs Search Tool
# =============================================================================


REPO_ROOT = Path("/Users/vivariumenv1/GITHUBS/scout")


@log_tool_invocation
async def scout_search_docs(
    query: str = "",
    limit: int = 5,
    ctx: Any = None,
) -> ToolOutput:
    """
    Search generated documentation using semantic full-text search.

    This tool searches the auto-generated docs in docs/ directory.
    Use this to find documentation about Scout modules, APIs, and concepts.

    Args:
        query: Search query (keywords, module name, or question)
        limit: Maximum number of results (default 5)

    Returns:
        ToolOutput with tool_name="search_docs", content=result_string, cost_usd=0.0

    Example:
        scout_search_docs(query="how does config work")
    """
    from scout.cli.index import index_docs, search_docs, _db_path
    from pathlib import Path

    repo_root = REPO_ROOT

    # First, reindex docs
    docs_indexed = index_docs(repo_root)

    # Then search
    results = search_docs(repo_root, query, limit)

    if not results:
        content = f"No docs found for '{query}'. {docs_indexed} docs indexed."
        return ToolOutput.from_content(
            tool_name="search_docs",
            content=content,
            cost_usd=0.0,
            metadata={"query": query, "limit": limit},
        )

    output = f"Found {len(results)} results (indexed {docs_indexed} docs):\n\n"
    for r in results:
        output += f"## {r['title']} ({r['module']})\n"
        output += f"```\n{r.get('snippet', 'No snippet')}\n```\n"
        output += f"Path: {r['path']}\n\n"

    return ToolOutput.from_content(
        tool_name="search_docs",
        content=output,
        cost_usd=0.0,
        metadata={"query": query, "limit": limit, "num_results": len(results)},
    )


@log_tool_invocation
async def scout_reindex_docs(
    ctx: Any = None,
) -> ToolOutput:
    """
    Reindex all generated documentation for search.

    Use this after generating new docs to make them searchable.

    Returns:
        ToolOutput with tool_name="reindex_docs", content=result_string, cost_usd=0.0

    Example:
        scout_reindex_docs()
    """
    from scout.cli.index import index_docs
    from pathlib import Path

    count = index_docs(REPO_ROOT)
    content = f"Indexed {count} documentation files."
    return ToolOutput.from_content(
        tool_name="reindex_docs",
        content=content,
        cost_usd=0.0,
        metadata={"count": count},
    )
