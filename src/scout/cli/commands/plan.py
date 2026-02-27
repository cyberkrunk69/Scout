#!/usr/bin/env python
"""
Plan command - Generate implementation plans.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from scout.cli.mcp_bridge.client import get_mcp_client
from scout.cli.formatting.output import ConsoleOutput


async def run(query: Optional[str] = None, files: Optional[list] = None, json_output: bool = False) -> int:
    """Run the plan command."""
    console = ConsoleOutput()

    if not query:
        console.print("[yellow]Usage: scout plan \"<query>\" [--files file1 file2] [--json][/yellow]")
        return 1

    # Add file context if provided
    context = ""
    if files:
        from scout.cli.context.files import FileContext
        fc = FileContext()
        context = fc.read_files(files)
        query = f"{query}\n\nContext:\n{context}"

    console.print(f"[dim]Generating plan for: {query[:50]}...[/dim]")

    client = get_mcp_client()
    result = await client.call_tool("scout_plan", {
        "request": query,
        "json_output": True,
        "structured": json_output
    })

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return 1

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        console.print("\n[bold]Plan:[/bold]")
        console.print_markdown(result.get("plan", "No plan generated"))

    return 0


if __name__ == "__main__":
    asyncio.run(run())
