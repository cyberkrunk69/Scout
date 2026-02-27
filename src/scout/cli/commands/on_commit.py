#!/usr/bin/env python
"""On-commit command - Git post-commit hook."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from scout.cli.formatting.output import ConsoleOutput

logger = logging.getLogger(__name__)


def run(
    files: Optional[List[str]] = None,
) -> int:
    """Run the on-commit command - git hook for processing changed files."""
    
    console = ConsoleOutput()

    try:
        # Get files - from args or stdin
        if not files and not sys.stdin.isatty():
            # Read from stdin if no args (e.g. from git hook piping)
            files = [line.strip() for line in sys.stdin if line.strip()]
        
        if not files:
            console.print("[yellow]No files to process[/yellow]")
            return 0
        
        # Convert to Path objects
        changed_files = [Path(f) for f in files]
        
        # Import router module
        from scout.router import on_git_commit
        
        # Run the hook
        on_git_commit(changed_files)
        
        console.print(f"[green]Processed {len(changed_files)} file(s)[/green]")
        return 0
        
    except Exception as e:
        logger.exception(f"on-commit hook failed: {e}")
        console.print(f"[red]Error: {e}[/red]")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Git post-commit hook")
    parser.add_argument("files", nargs="*", help="Changed files from git diff-tree")
    
    args = parser.parse_args()
    exit(run(files=args.files))
