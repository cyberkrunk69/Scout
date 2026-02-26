#!/usr/bin/env python
"""Prepare-commit-msg command - Git prepare-commit-msg hook."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from scout.cli.formatting.output import ConsoleOutput
from scout.router import BudgetExhaustedError, TriggerRouter

logger = logging.getLogger(__name__)


def run(
    message_file: str,
) -> int:
    """Run the prepare-commit-msg command - populate commit message from drafts."""
    
    console = ConsoleOutput()

    try:
        message_path = Path(message_file).resolve()
        
        if not message_path.exists():
            return 0
        
        # Use TriggerRouter to prepare commit message
        router = TriggerRouter()
        router.prepare_commit_msg(message_path)
        
        console.print("[green]Commit message prepared[/green]")
        return 0
        
    except BudgetExhaustedError:
        # TICKET-86: distinct exit code for budget exhaustion
        console.print("[yellow]Budget exhausted, skipping commit message[/yellow]")
        return 2
        
    except Exception as e:
        logger.exception(f"prepare-commit-msg hook failed: {e}")
        # Don't fail the commit on error - just log
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Git prepare-commit-msg hook")
    parser.add_argument("message_file", help="Path to commit message file (from git)")
    
    args = parser.parse_args()
    exit(run(message_file=args.message_file))
