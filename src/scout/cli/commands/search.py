#!/usr/bin/env python
"""
Search command - Search the codebase.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from scout.cli.mcp_bridge.client import get_mcp_client
from scout.cli.formatting.output import ConsoleOutput


async def run(pattern: str, path: Optional[str] = None) -> int:
    """Run the search command."""
    console = ConsoleOutput()

    if not pattern:
        console.print("[yellow]Usage: scout search <pattern> [--path path][/yellow]")
        return 1

    console.print(f"[dim]Searching for: {pattern}[/dim]")

    client = get_mcp_client()
    result = await client.call_tool("scout_grep", {
        "pattern": pattern,
        "path": path
    })

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return 1

    console.print("\n[bold]Results:[/bold]")
    console.print(result.get("result", "No results"))

    return 0


if __name__ == "__main__":
    import sys
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else None))
