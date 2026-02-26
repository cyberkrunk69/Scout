#!/usr/bin/env python
"""
Git command - Git helpers.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from scout.cli.mcp_bridge.client import get_mcp_client
from scout.cli.formatting.output import ConsoleOutput


async def run(subcommand: Optional[str] = None, auto: bool = False) -> int:
    """Run the git command."""
    console = ConsoleOutput()

    if not subcommand:
        console.print("[yellow]Usage: scout git <commit|pr|status> [--auto][/yellow]")
        return 1

    client = get_mcp_client()

    if subcommand == "status":
        result = await client.call_tool("scout_git_status", {})

    elif subcommand == "commit":
        result = await client.call_tool("scout_git_commit", {
            "auto": auto
        })

    elif subcommand == "pr":
        result = await client.call_tool("scout_pr", {})

    elif subcommand == "log":
        result = await client.call_tool("scout_git_log", {})

    elif subcommand == "diff":
        result = await client.call_tool("scout_git_diff", {})

    else:
        console.print(f"[red]Unknown git subcommand: {subcommand}[/red]")
        console.print("Available: commit, pr, status, log, diff")
        return 1

    if "error" in result:
        console.print(f"[red]Error: {result['error']}[/red]")
        return 1

    console.print(result.get("result", "No result"))

    return 0


if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("subcommand", nargs="?")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.subcommand, args.auto))
