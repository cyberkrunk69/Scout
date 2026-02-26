"""Minimal PlanProgress stub for enhanced CLI commands.

This is a lightweight version for commands that need progress tracking
but don't want to import from legacy cli/plan.py.
"""

from typing import Optional


class PlanProgress:
    """Minimal progress tracker for CLI commands."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.total_cost = 0.0
        self.total_tokens = 0
        self.task_name = ""

    def start(self, task_name: str):
        """Start a new task."""
        self.task_name = task_name
        if not self.quiet:
            print(f"▶ {task_name}...")

    def info(self, message: str):
        """Print info message."""
        if not self.quiet:
            print(f"  → {message}")

    def warning(self, message: str):
        """Print warning message."""
        if not self.quiet:
            print(f"  ⚠ {message}")

    def complete(self, message: str):
        """Mark task complete."""
        if not self.quiet:
            print(f"✓ {message}")

    def spin(self, message: str = ""):
        """Spinner (no-op in minimal version)."""
        pass
