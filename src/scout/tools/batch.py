from __future__ import annotations
"""
Batch Processing Tools for Scout MCP Server.

This module contains extracted batch processing tools:
- scout_batch: Run multiple Scout tasks in parallel or sequentially
- scout_run: Run a Python module with arguments
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from scout.tool_output import ToolOutput
from scout.tools import log_tool_invocation

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
# Scout Batch Tool
# =============================================================================


@log_tool_invocation
async def scout_batch(
    tasks_json: str,
    auto_execute: bool = True,
    ctx: Any = None,
) -> ToolOutput:
    """
    Run multiple Scout tasks in parallel or sequentially with state.

    Use this tool to execute multiple Scout commands in a batch.
    - Supports sequential and parallel execution modes
    - Can extract steps from plan output and execute them
    - Maintains state between tasks

    Example:
    ```
    scout_batch(tasks_json='[
      {"command": "lint", "args": {"paths": ["file1.py"]}},
      {"command": "git_status", "args": {}}
    ]')
    ```

    Returns:
        ToolOutput with tool_name="batch", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "vivarium.scout.cli.pipeline",
        "--tasks",
        tasks_json,
    ]

    if not auto_execute:
        cmd.append("--no-auto-execute")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)

        if proc.returncode != 0:
            return ToolOutput.from_content(
                tool_name="batch",
                content=json.dumps(
                    {
                        "status": "error",
                        "message": stderr.decode() if stderr else "Batch failed",
                    }
                ),
                cost_usd=0.0,
                metadata={"auto_execute": auto_execute, "error": True},
            )
        return ToolOutput.from_content(
            tool_name="batch",
            content=stdout.decode(),
            cost_usd=0.0,
            metadata={"auto_execute": auto_execute},
        )
    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="batch",
            content=json.dumps({"status": "error", "message": "Batch operation timed out"}),
            cost_usd=0.0,
            metadata={"auto_execute": auto_execute, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="batch",
            content=json.dumps({"status": "error", "message": str(e)}),
            cost_usd=0.0,
            metadata={"auto_execute": auto_execute, "error": True},
        )


# =============================================================================
# Scout Run Tool
# =============================================================================


@log_tool_invocation
async def scout_run(
    module: str,
    args: list[str] = None,
    timeout: int = 60,
    full: bool = False,
    json_output: bool = True,
    ctx: Any = None,
) -> ToolOutput:
    """
    Run a Python module with arguments, timeout, and capture output.

    Use this tool to:
    - Execute Python modules (unittest, mypy, pytest, etc.) with arguments
    - Run commands with a timeout to prevent hanging
    - Capture stdout/stderr for analysis

    Example: `scout_run(module="unittest", args=["discover", "tests/"])`

    Returns:
        ToolOutput with tool_name="run", content=result_string, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "vivarium.scout.cli.run",
        "--module",
        module,
    ]

    if args:
        cmd.extend(args)
    if timeout:
        cmd.extend(["--timeout", str(timeout)])
    if full:
        cmd.append("--full")
    if json_output:
        cmd.append("--json")

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout + 10
        )

        output = stdout.decode()
        if not full and stderr:
            output += "\n" + stderr.decode()

        if proc.returncode != 0:
            return ToolOutput.from_content(
                tool_name="run",
                content=json.dumps({"status": "error", "message": output, "returncode": proc.returncode}),
                cost_usd=0.0,
                metadata={"module": module, "args": args, "error": True},
            )
        return ToolOutput.from_content(
            tool_name="run",
            content=output,
            cost_usd=0.0,
            metadata={"module": module, "args": args},
        )
    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="run",
            content=json.dumps({"status": "error", "message": f"Run operation timed out after {timeout}s"}),
            cost_usd=0.0,
            metadata={"module": module, "args": args, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="run",
            content=json.dumps({"status": "error", "message": str(e)}),
            cost_usd=0.0,
            metadata={"module": module, "args": args, "error": True},
        )
