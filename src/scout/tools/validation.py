from __future__ import annotations
"""
Validation Tools for Scout MCP Server.

This module contains extracted validation tools:
- scout_lint: Run Scout linter on Python files
- scout_validate_module: Run type checking and tests
- scout_env: Check environment variables
- scout_function_info: Inspect function signatures
"""

import json
import subprocess
from pathlib import Path
from typing import Any


from scout.tool_output import ToolOutput

# Simple passthrough decorator (replaces @simple_cache and @log_tool_invocation)
def simple_cache(ttl_seconds: int = 60, dependencies: list = None):
    """Passthrough decorator - cache functionality not yet implemented."""
    def decorator(func):
        return func
    return decorator

def log_tool_invocation(func):
    """Passthrough decorator - logging handled by audit.py."""
    return func

# Shared configuration
VENV_PYTHON = "/Users/vivariumenv1/Vivarium/.venv/bin/python"
REPO_ROOT = Path("/Users/vivariumenv1/Vivarium")


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
# Scout Lint Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@log_tool_invocation
def scout_lint(
    path: str = ".",
    fix: bool = False,
    check: bool = False,
    all_files: bool = False,
    verbose: bool = False,
    json_output: bool = False,
    full: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Run Scout linter on Python files.

    Use this tool to check and fix lint errors.
    - Use --fix to automatically correct issues
    - Use --check to report only, no fixes
    - Use --all to lint all tracked Python files

    Example: `scout_lint(path="src/main.py", fix=True)`

    Returns:
        ToolOutput with tool_name="lint", content=result_string, cost_usd=0.0
    """
    cmd = [VENV_PYTHON, "-m", "vivarium.scout.cli.lint"]

    if path and path != ".":
        cmd.append(path)
    if fix:
        cmd.append("--fix")
    if check:
        cmd.append("--check")
    if all_files:
        cmd.append("--all")
    if verbose:
        cmd.append("--verbose")
    if json_output:
        cmd.append("--json")
    if full:
        cmd.append("--full")

    try:
        result = _run_command(cmd, timeout=120)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="lint",
            content=content,
            cost_usd=0.0,
            metadata={"path": path, "fix": fix, "check": check, "all_files": all_files},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="lint",
            content=f"Error running lint: {e}",
            cost_usd=0.0,
            metadata={"path": path, "fix": fix, "check": check, "all_files": all_files},
        )


# =============================================================================
# Scout Validate Module Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@log_tool_invocation
def scout_validate_module(
    module_path: str,
    type_check: bool = True,
    test: bool = True,
    json_output: bool = False,
    full: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Run type checking (mypy) and tests (pytest) on a module.

    Example: `scout_validate_module(module_path="vivarium/scout/cli/audit.py")`

    Returns:
        ToolOutput with tool_name="validate_module", content=result_string, cost_usd=0.0
    """
    cmd = [VENV_PYTHON, "-m", "vivarium.scout.cli.validate", "--module", module_path]

    if not type_check:
        cmd.append("--no-type-check")
    if not test:
        cmd.append("--no-test")
    if json_output:
        cmd.append("--json")
    if full:
        cmd.append("--full")

    try:
        result = _run_command(cmd, timeout=180)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="validate_module",
            content=content,
            cost_usd=0.0,
            metadata={"module_path": module_path, "type_check": type_check, "test": test},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="validate_module",
            content=f"Error validating module: {e}",
            cost_usd=0.0,
            metadata={"module_path": module_path, "type_check": type_check, "test": test},
        )


# =============================================================================
# Scout Env Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@log_tool_invocation
def scout_env(
    vars: list[str] = None,
    json_output: bool = False,
    full: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Check environment variables and report which are set.

    Use this tool to verify that required API keys and environment variables are configured.
    - Checks MINIMAX_API_KEY, GROQ_API_KEY, and GEMINI_API_KEY by default
    - Or check specific variables by passing them in the `vars` list

    Example: `scout_env(vars=["MINIMAX_API_KEY", "GROQ_API_KEY"])`

    Returns:
        ToolOutput with tool_name="env", content=result_string, cost_usd=0.0
    """
    cmd = [VENV_PYTHON, "-m", "vivarium.scout.cli.env"]

    if vars:
        cmd.extend(vars)
    if json_output:
        cmd.append("--json")
    if full:
        cmd.append("--full")

    try:
        result = _run_command(cmd)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="env",
            content=content,
            cost_usd=0.0,
            metadata={"vars": vars},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="env",
            content=f"Error checking env: {e}",
            cost_usd=0.0,
            metadata={"vars": vars},
        )


# =============================================================================
# Scout Function Info Tool
# =============================================================================


@simple_cache(ttl_seconds=5, dependencies=["**/*.py"])
@log_tool_invocation
def scout_function_info(
    module: str,
    function: str,
    json_output: bool = False,
    full: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Inspect function signatures, docstrings, and source locations.

    Use this tool to understand how a function works without reading the file manually.
    - Returns signature, docstring, source file, and line number
    - Includes truncated source code for quick reference

    Example: `scout_function_info(module="vivarium.scout.llm.minimax", function="call_minimax_async")`

    Returns:
        ToolOutput with tool_name="function_info", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "vivarium.scout.cli.function_info",
        "--module",
        module,
        "--function",
        function,
    ]

    if json_output:
        cmd.append("--json")
    if full:
        cmd.append("--full")

    try:
        result = _run_command(cmd)
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr
        content = output or "(no output)"
        return ToolOutput.from_content(
            tool_name="function_info",
            content=content,
            cost_usd=0.0,
            metadata={"module": module, "function": function},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="function_info",
            content=f"Error getting function info: {e}",
            cost_usd=0.0,
            metadata={"module": module, "function": function},
        )
