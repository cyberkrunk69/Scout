#!/usr/bin/env python
"""
Execute command - Execute a plan.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from scout.cli.mcp_bridge.client import get_mcp_client
from scout.cli.formatting.output import ConsoleOutput


async def run(plan_id: Optional[str] = None) -> int:
    """Run the execute command."""
    console = ConsoleOutput()

    if not plan_id:
        console.print("[yellow]Usage: scout execute <plan_id>[/yellow]")
        return 1

    # Check if it's a file path
    plan_path = Path(plan_id)
    if plan_path.exists():
        with open(plan_path) as f:
            plan_json = f.read()
    else:
        plan_json = plan_id

    console.print(f"[dim]Executing plan...[/dim]")

    client = get_mcp_client()
    result = await client.call_tool("scout_execute_plan", {
        "plan_json": plan_json
    })

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return 1

    console.print("\n[bold]Execution result:[/bold]")
    console.print(result.get("result", "No result"))

    return 0


if __name__ == "__main__":
    asyncio.run(run())
