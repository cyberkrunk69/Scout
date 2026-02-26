#!/usr/bin/env python
"""
Config command - View/edit configuration.
"""

from __future__ import annotations

from scout.cli.formatting.output import ConsoleOutput
from scout.config import ScoutConfig


def run(action: str = "show", key: str = None, value: str = None) -> int:
    """Run the config command."""
    console = ConsoleOutput()
    config = ScoutConfig()

    if action == "show":
        console.print("[bold]Configuration:[/bold]")
        console.print(f"""
Scout Configuration (from .scout/config.yaml):

  Raw config: {list(config._raw.keys())}
        """)
        return 0

    elif action == "get":
        if not key:
            console.print("[yellow]Usage: scout config get <key>[/yellow]")
            return 1
        # Try to get nested key
        parts = key.split(".")
        val = config._raw
        for part in parts:
            val = val.get(part)
            if val is None:
                break
        console.print(f"{key} = {val}")
        return 0

    elif action == "set":
        if not key or value is None:
            console.print("[yellow]Usage: scout config set <key> <value>[/yellow]")
            return 1
        console.print(f"[yellow]Config modification not implemented yet[/yellow]")
        console.print("[dim]Edit .scout/config.yaml directly to modify config[/dim]")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "show"
    key = sys.argv[2] if len(sys.argv) > 2 else None
    value = sys.argv[3] if len(sys.argv) > 3 else None
    run(action, key, value)
