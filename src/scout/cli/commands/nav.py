#!/usr/bin/env python
"""
Nav command - Navigate to symbol.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from scout.cli.mcp_bridge.client import get_mcp_client
from scout.cli.formatting.output import ConsoleOutput


async def run(symbol: str, trust_level: str = "normal") -> int:
    """Run the nav command."""
    console = ConsoleOutput()

    if not symbol:
        console.print("[yellow]Usage: scout nav <symbol>[/yellow]")
        return 1

    console.print(f"[dim]Navigating to: {symbol}[/dim]")

    client = get_mcp_client()
    result = await client.call_tool("scout_nav", {
        "task": symbol,
        "trust_level": trust_level,
    })

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return 1

    console.print("\n[bold]Navigation result:[/bold]")
    console.print(result.get("result", "No result"))

    return 0


if __name__ == "__main__":
    import sys
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else None))
