#!/usr/bin/env python
"""
Output formatting with Rich console.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.theme import Theme


# Custom theme for Scout CLI
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "red",
    "success": "green",
})


class ConsoleOutput:
    """Console output with Rich formatting."""

    def __init__(self):
        self.console = Console(theme=custom_theme)

    def print(self, text: str = "", **kwargs):
        """Print text to console."""
        self.console.print(text, **kwargs)

    def print_markdown(self, text: str):
        """Print markdown-formatted text."""
        md = Markdown(text)
        self.console.print(md)

    def print_error(self, text: str):
        """Print error text."""
        self.console.print(f"[red]Error:[/red] {text}")

    def print_success(self, text: str):
        """Print success text."""
        self.console.print(f"[green]Success:[/green] {text}")

    def print_warning(self, text: str):
        """Print warning text."""
        self.console.print(f"[yellow]Warning:[/yellow] {text}")

    def print_info(self, text: str):
        """Print info text."""
        self.console.print(f"[cyan]Info:[/cyan] {text}")

    def print_dim(self, text: str):
        """Print dimmed text."""
        self.console.print(f"[dim]{text}[/dim]")

    def clear(self):
        """Clear the console."""
        self.console.clear()


if __name__ == "__main__":
    out = ConsoleOutput()
    out.print("[bold]Hello World![/bold]")
    out.print_markdown("# Markdown Heading\n\nSome **bold** text.")
