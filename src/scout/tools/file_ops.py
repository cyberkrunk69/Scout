from __future__ import annotations
"""
File Operations Tools for Scout MCP Server.

This module contains extracted file operation tools:
- scout_read_file: Read a file's content
- scout_write_with_review: Write new content after LLM review
- scout_delete_with_review: Delete a file after LLM review
"""

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from scout.tool_output import ToolOutput
from scout.tools import log_tool_invocation
from scout.config.defaults import (
    FILE_READ_TIMEOUT,
    FILE_WRITE_TIMEOUT,
    FILE_DELETE_TIMEOUT,
    FILE_EDIT_TIMEOUT,
)
from scout.config.paths import REPO_ROOT, VENV_PYTHON


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
# Scout Read File Tool
# =============================================================================


@log_tool_invocation
def scout_read_file(
    file_path: str,
    ctx: Any = None,
) -> ToolOutput:
    """
    Read a file's content without review (but enforce path restrictions).

    Use this tool to read files within allowed directories.
    - Validates path is within allowed directories
    - Returns file content directly
    - No LLM call - deterministic output

    Example: `scout_read_file(file_path="vivarium/scout/cli/plan.py")`

    Returns:
        ToolOutput with tool_name="read_file", content=file_content_or_error, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "vivarium.scout.cli.scout",
        "read-file",
        "--file-path",
        file_path,
    ]

    try:
        result = _run_command(cmd, timeout=FILE_READ_TIMEOUT)
        if result.returncode != 0:
            content = json.dumps(
                {"status": "error", "message": result.stderr or "Failed to read file"}
            )
            return ToolOutput.from_content(
                tool_name="read_file",
                content=content,
                cost_usd=0.0,
                metadata={"file_path": file_path, "error": True},
            )
        return ToolOutput.from_content(
            tool_name="read_file",
            content=result.stdout,
            cost_usd=0.0,
            metadata={"file_path": file_path},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="read_file",
            content=json.dumps({"status": "error", "message": str(e)}),
            cost_usd=0.0,
            metadata={"file_path": file_path, "error": True},
        )


# =============================================================================
# Scout Write With Review Tool
# =============================================================================


@log_tool_invocation
async def scout_write_with_review(
    content: str,
    file_path: str,
    user_request_history: str = None,
    ctx: Any = None,
) -> ToolOutput:
    """
    Write new content to a file after obtaining approval from a senior engineer (MiniMax).

    Use this tool to create or overwrite files with LLM-powered review.
    - Validates path is within allowed directories
    - Generates diff between old and new content
    - Sends diff to MiniMax for senior engineer review
    - Only writes if approved

    Example: `scout_write_with_review(file_path="vivarium/scout/new_tool.py", content="# New file...")`

    Returns:
        ToolOutput with tool_name="write_with_review", content=result, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "vivarium.scout.cli.scout",
        "write-with-review",
        "--content",
        content,
        "--file-path",
        file_path,
    ]

    if user_request_history:
        cmd.extend(["--user-request-history", user_request_history])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=FILE_WRITE_TIMEOUT)

        if proc.returncode != 0:
            content = json.dumps(
                {
                    "status": "error",
                    "message": stderr.decode() if stderr else "Failed to write file",
                }
            )
            return ToolOutput.from_content(
                tool_name="write_with_review",
                content=content,
                cost_usd=0.0,
                metadata={"file_path": file_path, "error": True},
            )
        return ToolOutput.from_content(
            tool_name="write_with_review",
            content=stdout.decode(),
            cost_usd=0.0,
            metadata={"file_path": file_path},
        )
    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="write_with_review",
            content=json.dumps({"status": "error", "message": "Write operation timed out"}),
            cost_usd=0.0,
            metadata={"file_path": file_path, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="write_with_review",
            content=json.dumps({"status": "error", "message": str(e)}),
            cost_usd=0.0,
            metadata={"file_path": file_path, "error": True},
        )


# =============================================================================
# Scout Delete With Review Tool
# =============================================================================


@log_tool_invocation
async def scout_delete_with_review(
    file_path: str,
    reason: str = "",
    user_request_history: str = None,
    ctx: Any = None,
) -> ToolOutput:
    """
    Delete a file after senior engineer review.

    Use this tool to delete files with LLM-powered safety review.
    - Validates path is within allowed directories
    - Reads file content to provide context for review
    - Sends deletion request to MiniMax for approval
    - Only deletes if approved

    Example: `scout_delete_with_review(file_path="vivarium/scout/old_tool.py", reason="Obsolete utility")`

    Returns:
        ToolOutput with tool_name="delete_with_review", content=result, cost_usd=0.0
    """
    cmd = [
        VENV_PYTHON,
        "-m",
        "vivarium.scout.cli.scout",
        "delete-with-review",
        "--file-path",
        file_path,
    ]

    if reason:
        cmd.extend(["--reason", reason])
    if user_request_history:
        cmd.extend(["--user-request-history", user_request_history])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=FILE_DELETE_TIMEOUT)

        if proc.returncode != 0:
            content = json.dumps(
                {
                    "status": "error",
                    "message": stderr.decode() if stderr else "Failed to delete file",
                }
            )
            return ToolOutput.from_content(
                tool_name="delete_with_review",
                content=content,
                cost_usd=0.0,
                metadata={"file_path": file_path, "error": True},
            )
        return ToolOutput.from_content(
            tool_name="delete_with_review",
            content=stdout.decode(),
            cost_usd=0.0,
            metadata={"file_path": file_path},
        )
    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="delete_with_review",
            content=json.dumps({"status": "error", "message": "Delete operation timed out"}),
            cost_usd=0.0,
            metadata={"file_path": file_path, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="delete_with_review",
            content=json.dumps({"status": "error", "message": str(e)}),
            cost_usd=0.0,
            metadata={"file_path": file_path, "error": True},
        )


# Need asyncio for async functions
import asyncio


# =============================================================================
# Scout Edit Tool
# =============================================================================


@log_tool_invocation
async def scout_edit(
    file_path: str,
    instruction: str,
    dry_run: bool = False,
    ctx: Any = None,
) -> ToolOutput:
    """
    Edit a file based on an instruction with optional dry-run mode.

    This tool provides AI-assisted file editing with safety features:
    - Validates path is within allowed directories
    - If dry_run=True, returns the diff without applying changes
    - If dry_run=False, applies the edit and returns the result

    Example (dry-run): `scout_edit(file_path="path/to/file.py", instruction="add docstring", dry_run=True)`
    Example (apply): `scout_edit(file_path="path/to/file.py", instruction="add docstring", dry_run=False)`

    Args:
        file_path: Path to the file to edit
        instruction: Natural language instruction for the edit
        dry_run: If True, return diff without applying changes (default: False)
        ctx: Optional context

    Returns:
        ToolOutput with tool_name="scout_edit", content=result, cost_usd=0.0
        For dry_run=True: includes 'diff' in the result
        For dry_run=False: includes 'status' and 'applied' info
    """
    path = Path(file_path)

    # Validate path exists
    if not path.exists():
        return ToolOutput.from_content(
            tool_name="scout_edit",
            content=json.dumps({"status": "error", "error": f"File not found: {file_path}"}),
            cost_usd=0.0,
            metadata={"file_path": file_path, "error": True},
        )

    try:
        # Read current content
        current_content = path.read_text(encoding="utf-8")

        # For dry-run, we simulate what would happen by calling the LLM
        # but not applying the changes. We use the edit CLI for this.
        cmd = [
            VENV_PYTHON,
            "-m",
            "vivarium.scout.cli_enhanced.commands.edit",
            str(path),
            "--prompt",
            instruction,
        ]

        if dry_run:
            cmd.append("--dry-run")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=REPO_ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=os.environ.copy(),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=FILE_EDIT_TIMEOUT)

        output = stdout.decode() if stdout else ""
        error = stderr.decode() if stderr else ""

        if proc.returncode != 0:
            return ToolOutput.from_content(
                tool_name="scout_edit",
                content=json.dumps({"status": "error", "error": error or "Edit failed"}),
                cost_usd=0.0,
                metadata={"file_path": file_path, "error": True},
            )

        # Parse output - could be diff (dry-run) or success (apply)
        result = {
            "status": "dry_run" if dry_run else "applied",
            "file_path": str(path.absolute()),
        }

        if dry_run:
            result["diff"] = output
        else:
            result["message"] = output

        return ToolOutput.from_content(
            tool_name="scout_edit",
            content=json.dumps(result),
            cost_usd=0.0,
            metadata={"file_path": file_path, "dry_run": dry_run},
        )

    except asyncio.TimeoutError:
        return ToolOutput.from_content(
            tool_name="scout_edit",
            content=json.dumps({"status": "error", "error": "Edit operation timed out"}),
            cost_usd=0.0,
            metadata={"file_path": file_path, "error": True},
        )
    except Exception as e:
        return ToolOutput.from_content(
            tool_name="scout_edit",
            content=json.dumps({"status": "error", "error": str(e)}),
            cost_usd=0.0,
            metadata={"file_path": file_path, "error": True},
        )
