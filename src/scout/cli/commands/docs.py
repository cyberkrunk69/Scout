#!/usr/bin/env python
"""
Docs command - Generate documentation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from scout.cli.mcp_bridge.client import get_mcp_client
from scout.cli.formatting.output import ConsoleOutput


async def run(files: Optional[list] = None, output: Optional[str] = None, recursive: bool = False) -> int:
    """Run the docs command."""
    console = ConsoleOutput()

    if not files:
        console.print("[yellow]Usage: scout docs <files>... [--output dir] [--recursive][/yellow]")
        return 1

    for file in files:
        target = Path(file)
        if not target.exists():
            console.print(f"[red]File not found: {file}[/red]")
            continue

        console.print(f"[dim]Generating docs for: {file}[/dim]")

        client = get_mcp_client()
        result = await client.call_tool("scout_generate_docs", {
            "request": f"Generate comprehensive documentation for {file}",
            "target_files": str(target),
            "audience": "developer",
            "output_format": "markdown",
        })

        if "error" in result:
            console.print(f"[red]Error: {result['error']}[/red]")
            continue

        console.print(f"\n[bold]Docs for {file}:[/bold]")
        console.print(result.get("result", "No docs generated"))

    return 0


if __name__ == "__main__":
    import sys
    asyncio.run(run(files=sys.argv[1:] if len(sys.argv) > 1 else None))
