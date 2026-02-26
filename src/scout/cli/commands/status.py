#!/usr/bin/env python
"""
Status command - Show session info.
"""

from __future__ import annotations

from scout.cli.formatting.output import ConsoleOutput
from scout.cli.context.session import load_session


def run() -> int:
    """Run the status command."""
    console = ConsoleOutput()
    session = load_session()

    console.print(f"""
[bold]Session Status[/bold]

Session ID: {session.id}
Created: {session.created_at}
Messages: {len(session.messages)}
Total Cost: ${session.total_cost:.4f}
Total Tokens: {session.total_tokens}

[bold]Recent Messages:[/bold]
""")

    for msg in session.messages[-5:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")[:50]
        console.print(f"  [{role}]: {content}...")

    return 0


if __name__ == "__main__":
    run()
