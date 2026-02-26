from __future__ import annotations
"""
LLM-Powered Tools for Scout MCP Server.

This module contains extracted LLM-powered tools:
- scout_plan: Generate implementation plans
- scout_nav: Search codebase using navigation
- scout_query: Natural language repo search
- scout_roast: Efficiency reports and code critique
- scout_brief: Generate code briefings
- scout_propose_tool: Propose new tools
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from scout.cache import simple_cache
from scout.tools import log_tool_invocation
from scout.tool_output import ToolOutput

# Shared configuration
VENV_PYTHON = "/Users/vivariumenv1/GITHUBS/scout/.venv/bin/python"
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
# Scout Plan Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@log_tool_invocation
async def scout_plan(
    request: str,
    output_dir: str = "docs/plans",
    json_output: bool = False,
    structured: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Generate an implementation plan using MiniMax.

    Always use this tool for planning new features or significant changes.
    The plan will be saved to `output_dir` (default docs/plans).

    Args:
        request: Natural language feature request.
        output_dir: Directory to save plan (if not JSON).
        json_output: If True, return JSON with plan and metadata.
        structured: If True, return parsed steps with commands/args for automation.

    Example: `scout_plan(request="add user auth")`

    Returns:
        ToolOutput with tool_name="plan", content=plan_string, cost_usd=estimated
    """
    # Estimated cost for plan generation
    estimated_cost = 0.02

    cmd = [
        VENV_PYTHON,
        "-m",
        "scout.cli.main", "plan",
        request,
        "--output-dir",
        output_dir,
    ]
    if json_output or structured:
        cmd.append("--json")
    # Always use --quiet when capturing output (JSON or structured)
    # Progress will be shown via REPL's live display instead
    cmd.append("--quiet")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await proc.communicate()
        plan_output = stdout.decode()

        # Auto-capture plan
        try:
            asyncio.create_task(
                _auto_capture_plan(request, plan_output, json_output or structured)
            )
        except Exception:
            pass

        if structured:
            try:
                plan_data = json.loads(plan_output)
                # Handle CLI's actual output format: {"text": "...", "data": {"steps": [...]}}
                # Also support legacy format: {"plan": "...", "steps": [...]}
                data_section = plan_data.get("data", {})
                steps = data_section.get("steps", []) or plan_data.get("steps", [])
                plan_text = plan_data.get("text", "") or data_section.get("plan", "") or plan_data.get("plan", "")
                metadata_section = data_section.get("metadata", {})
                tokens = metadata_section.get("tokens", 0)
                cost = metadata_section.get("cost", estimated_cost)
                model = metadata_section.get("model", "minimax")

                # Build structured result directly from JSON
                structured_result = {
                    "plan": plan_text,
                    "steps": steps,
                    "tokens": tokens,
                    "cost": cost,
                    "model": model,
                }
                content = json.dumps(structured_result, indent=2)
            except json.JSONDecodeError:
                content = plan_output
        elif json_output:
            content = plan_output
        else:
            content = plan_output

        return ToolOutput.from_content(
            tool_name="plan",
            content=content,
            cost_usd=estimated_cost,
            metadata={"request": request, "output_dir": output_dir, "json_output": json_output, "structured": structured},
        )

    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="plan",
            content="Error: Plan generation timed out",
            cost_usd=estimated_cost,
            metadata={"request": request, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="plan",
            content=f"Error generating plan: {e}",
            cost_usd=estimated_cost,
            metadata={"request": request, "error": True},
        )


# =============================================================================
# Scout Nav Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@log_tool_invocation
async def scout_nav(
    query: str = None,
    task: str = None,
    entry: str = None,
    file: str = None,
    question: str = None,
    json_output: bool = True,
    output: str = None,
    ctx: Any = None,
) -> ToolOutput:
    """
    Search the codebase using Scout's navigation.

    Use this tool to find code in the codebase quickly.
    - Use --task for navigation task (e.g., 'fix auth timeout bug')
    - Use --entry for entry point hint
    - Use --file and --question for Q&A mode

    Example: `scout_nav(task="fix auth timeout bug")`

    Returns:
        ToolOutput with tool_name="nav", content=result_string, cost_usd=estimated
    """
    # Estimated cost for navigation
    estimated_cost = 0.015

    cmd = [VENV_PYTHON, "-m", "scout.cli.main", "nav"]

    if query:
        cmd.append(query)
    if task:
        cmd.extend(["--task", task])
    if entry:
        cmd.extend(["--entry", entry])
    if file:
        cmd.extend(["--file", file])
    if question:
        cmd.extend(["--question", question])
    if json_output:
        cmd.append("--json")
    if output:
        cmd.extend(["--output", output])

    try:
        result = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=120)
        if result.returncode != 0:
            return ToolOutput.from_content(
                tool_name="nav",
                content=f"Error: {stderr.decode()}",
                cost_usd=estimated_cost,
                metadata={"query": query, "task": task, "error": True},
            )
        return ToolOutput.from_content(
            tool_name="nav",
            content=stdout.decode(),
            cost_usd=estimated_cost,
            metadata={"query": query, "task": task},
        )
    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="nav",
            content="Error: Navigation timed out",
            cost_usd=estimated_cost,
            metadata={"query": query, "task": task, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="nav",
            content=f"Error running nav: {e}",
            cost_usd=estimated_cost,
            metadata={"query": query, "task": task, "error": True},
        )


# =============================================================================
# Scout Query Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@log_tool_invocation
async def scout_query(
    query: str,
    scope: str = None,
    json_output: bool = True,
    ctx: Any = None,
) -> ToolOutput:
    """
    Run natural language repo search using Scout.

    Use this tool to search the codebase with natural language queries.
    - Outputs to clipboard and docs/temp
    - Use --scope to limit to specific module/directory

    Example: `scout_query("where is auth middleware defined")`

    Returns:
        ToolOutput with tool_name="query", content=result_string, cost_usd=estimated
    """
    # Estimated cost for query
    estimated_cost = 0.01

    cmd = [VENV_PYTHON, "-m", "scout.cli.main", "search", query]

    if scope:
        cmd.extend(["--scope", scope])
    if json_output:
        cmd.append("--json")

    try:
        result = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=120)
        if result.returncode != 0:
            return ToolOutput.from_content(
                tool_name="query",
                content=f"Error: {stderr.decode()}",
                cost_usd=estimated_cost,
                metadata={"query": query, "scope": scope, "error": True},
            )
        return ToolOutput.from_content(
            tool_name="query",
            content=stdout.decode(),
            cost_usd=estimated_cost,
            metadata={"query": query, "scope": scope},
        )
    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="query",
            content="Error: Query timed out",
            cost_usd=estimated_cost,
            metadata={"query": query, "scope": scope, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="query",
            content=f"Error running query: {e}",
            cost_usd=estimated_cost,
            metadata={"query": query, "scope": scope, "error": True},
        )


# =============================================================================
# Scout Roast Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@log_tool_invocation
async def scout_roast(
    target: str = None,
    today: bool = False,
    week: bool = False,
    month: bool = False,
    use_docs: bool = True,
    compare: str = None,
    full: bool = False,
    json_output: bool = True,
    ctx: Any = None,
) -> ToolOutput:
    """
    Run Scout roast - efficiency reports and impact-aware code critique.

    Use this tool to get code critique and efficiency reports.
    - Use --today/--week/--month for time range
    - Use --target for specific file(s) to critique

    Example: `scout_roast(today=True)`

    Returns:
        ToolOutput with tool_name="roast", content=result_string, cost_usd=estimated
    """
    # Estimated cost for roast
    estimated_cost = 0.02

    cmd = [VENV_PYTHON, "-m", "scout.cli.main", "improve"]

    if target:
        cmd.extend(["--target", target])
    if today:
        cmd.append("--today")
    if week:
        cmd.append("--week")
    if month:
        cmd.append("--month")
    if not use_docs:
        cmd.append("--no-use-docs")
    if compare:
        cmd.extend(["--compare", compare])
    if full:
        cmd.append("--full")
    if json_output:
        cmd.append("--json")

    try:
        result = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=120)
        if result.returncode != 0:
            return ToolOutput.from_content(
                tool_name="roast",
                content=f"Error: {stderr.decode()}",
                cost_usd=estimated_cost,
                metadata={"target": target, "today": today, "week": week, "month": month, "error": True},
            )
        return ToolOutput.from_content(
            tool_name="roast",
            content=stdout.decode(),
            cost_usd=estimated_cost,
            metadata={"target": target, "today": today, "week": week, "month": month},
        )
    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="roast",
            content="Error: Roast timed out",
            cost_usd=estimated_cost,
            metadata={"target": target, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="roast",
            content=f"Error running roast: {e}",
            cost_usd=estimated_cost,
            metadata={"target": target, "error": True},
        )


# =============================================================================
# Scout Brief Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@log_tool_invocation
async def scout_brief(
    query: str = None,
    json_output: bool = True,
    ctx: Any = None,
) -> ToolOutput:
    """
    Run Scout brief - generate code briefings.

    Example: `scout_brief(query="authentication flow")`

    Returns:
        ToolOutput with tool_name="brief", content=result_string, cost_usd=estimated
    """
    # Estimated cost for brief
    estimated_cost = 0.015

    # brief command is not available in scout CLI - map to docs
    cmd = [VENV_PYTHON, "-m", "scout.cli.main", "docs"]

    if query:
        cmd.append(query)
    if json_output:
        cmd.append("--json")

    try:
        result = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=120)
        if result.returncode != 0:
            return ToolOutput.from_content(
                tool_name="brief",
                content=f"Error: {stderr.decode()}",
                cost_usd=estimated_cost,
                metadata={"query": query, "error": True},
            )
        return ToolOutput.from_content(
            tool_name="brief",
            content=stdout.decode(),
            cost_usd=estimated_cost,
            metadata={"query": query},
        )
    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="brief",
            content="Error: Brief timed out",
            cost_usd=estimated_cost,
            metadata={"query": query, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="brief",
            content=f"Error running brief: {e}",
            cost_usd=estimated_cost,
            metadata={"query": query, "error": True},
        )


# =============================================================================
# Scout Propose Tool
# =============================================================================


@log_tool_invocation
async def scout_propose_tool(
    tool_name: str,
    description: str,
    justification: str,
    security_considerations: str = "",
    session_id: str = "default",
    json_output: bool = True,
    ctx: Any = None,
) -> ToolOutput:
    """
    Propose a new tool for Scout MCP server.

    Use this tool when you need a capability that is not available through existing governed tools.
    This creates a GitHub issue for human review.

    Example: `scout_propose_tool(tool_name="scout_regex_search", description="Search code using regex", justification="Current tools don't support regex")`

    Returns:
        ToolOutput with tool_name="propose_tool", content=result_string, cost_usd=estimated
    """
    # Estimated cost for propose_tool
    estimated_cost = 0.005

    # Rate limiting check
    allowed, remaining = _check_rate_limit(session_id)
    if not allowed:
        return ToolOutput.from_content(
            tool_name="propose_tool",
            content=json.dumps(
                {
                    "status": "error",
                    "message": f"Rate limit exceeded. {remaining} proposals remaining.",
                    "hint": "Wait before proposing more tools or use a different session.",
                }
            ),
            cost_usd=estimated_cost,
            metadata={"tool_name": tool_name, "error": True, "rate_limited": True},
        )

    # Propose tool is not available in scout CLI - return an error
    return ToolOutput.from_content(
        tool_name="propose_tool",
        content=json.dumps(
            {
                "status": "not_implemented",
                "message": "The propose_tool command is not yet available in the scout CLI.",
                "hint": "Please use GitHub issues directly to propose new tools.",
            }
        ),
        cost_usd=estimated_cost,
        metadata={"tool_name": tool_name, "error": False, "not_implemented": True},
    )


# =============================================================================
# Helper Functions
# =============================================================================


# Rate limiting for tool proposals
_proposal_rate_limit: dict[str, list[float]] = {}
PROPOSAL_RATE_LIMIT = 5
PROPOSAL_RATE_WINDOW = 3600


def _check_rate_limit(session_id: str) -> tuple[bool, int]:
    """Check if session has exceeded rate limit."""
    import time

    now = time.time()
    if session_id not in _proposal_rate_limit:
        _proposal_rate_limit[session_id] = []

    _proposal_rate_limit[session_id] = [
        ts for ts in _proposal_rate_limit[session_id] if now - ts < PROPOSAL_RATE_WINDOW
    ]

    remaining = PROPOSAL_RATE_LIMIT - len(_proposal_rate_limit[session_id])
    return remaining > 0, remaining


def _record_proposal(session_id: str) -> None:
    """Record a proposal for rate limiting."""
    import time

    now = time.time()
    if session_id not in _proposal_rate_limit:
        _proposal_rate_limit[session_id] = []
    _proposal_rate_limit[session_id].append(now)


async def _auto_capture_plan(request: str, plan_output: str, is_json: bool) -> None:
    """Auto-capture plan to storage."""
    try:
        from scout.plan_capture import capture_plan

        await capture_plan(request, plan_output, is_json)
    except Exception:
        pass
