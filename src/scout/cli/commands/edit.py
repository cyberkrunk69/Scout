#!/usr/bin/env python
"""
Edit command - Edit files with AI assistance.
"""

from __future__ import annotations

import asyncio
import difflib
import json
from pathlib import Path
from typing import Optional

from scout.cli.mcp_bridge.client import get_mcp_client
from scout.cli.formatting.output import ConsoleOutput


async def run(file_path: str, prompt: str, dry_run: bool = False) -> int:
    """Run the edit command."""
    console = ConsoleOutput()

    if not file_path or not prompt:
        console.print("[yellow]Usage: scout edit <file> --prompt \"<instruction>\"[/yellow]")
        return 1

    target = Path(file_path)
    if not target.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        return 1

    console.print(f"[dim]Editing: {file_path}[/dim]")
    console.print(f"[dim]Instruction: {prompt}[/dim]")

    if dry_run:
        console.print("[dim]Mode: DRY RUN (no changes will be made)[/dim]")

    client = get_mcp_client()
    result = await client.call_tool("scout_edit", {
        "file_path": str(target),
        "instruction": prompt,
        "dry_run": dry_run,
    })

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return 1

    # Handle the result
    tool_result = result.get("result", {})
    if isinstance(tool_result, str):
        try:
            tool_result = json.loads(tool_result)
        except json.JSONDecodeError:
            pass

    if dry_run:
        # Show diff for dry-run mode
        console.print("\n[bold]Diff preview:[/bold]")
        diff = tool_result.get("diff", "No diff available")
        console.print(f"[cyan]{diff}[/cyan]")
    else:
        console.print("\n[bold]Edit result:[/bold]")
        console.print(tool_result.get("message", "No result"))

    return 0


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("--prompt", "-p", required=True)
    parser.add_argument("--dry-run", action="store_true", help="Show diff without applying changes")
    args = parser.parse_args()
    asyncio.run(run(args.file, args.prompt, args.dry_run))
