#!/usr/bin/env python
"""Improve command - Autonomous code improvement pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from scout.cli.formatting.output import ConsoleOutput
from scout.cli.context.session import load_session, save_session, Session
from scout.audit import AuditLog

logger = logging.getLogger(__name__)


def _load_session_safe() -> Session:
    """Load session with fallback on error."""
    try:
        return load_session()
    except Exception as e:
        logger.warning(f"Session load failed: {e}. Creating new session.")
        return Session()


async def run(
    target: Optional[str] = None,
    goal: str = "improve code quality",
    apply: bool = False,
    dry_run: bool = False,
    json_output: bool = False,
    rollback: Optional[str] = None,
) -> int:
    """Run the improve command - MCP-free, full integration."""
    
    console = ConsoleOutput()
    session = _load_session_safe()
    audit = AuditLog()

    # Handle rollback separately
    if rollback:
        from scout.improve import rollback_from_backup
        result = rollback_from_backup(rollback)
        if json_output:
            print(json.dumps(result, indent=2))
        else:
            print(f"Rollback {result['status']}: {result.get('message', '')}")
        return 0 if result['status'] == 'success' else 1

    # Validate
    if not target and not goal:
        console.print("[red]Error: target or goal required[/red]")
        return 1
    
    # Log start
    audit.log(
        "command_start",
        command="improve",
        target=target,
        goal=goal,
    )

    try:
        if not json_output:
            console.print(f"[dim]Running improve: {goal}[/dim]")
        
        # Call core pipeline module
        from scout.improve import run_improve_pipeline
        
        result = await run_improve_pipeline(
            goal=goal,
            target=target,
            apply_fixes=apply,
            dry_run=dry_run,
            repo_root=Path.cwd(),
        )
        
        # Cost tracking - extract from result
        cost = getattr(result, 'cost', 0.0) or 0.0
        session.total_cost += cost
        
        try:
            save_session(session)
        except Exception as e:
            logger.warning(f"Session save failed: {e}")
        
        # Format output - branch cleanly
        if json_output:
            output = {
                "status": result.status if hasattr(result, 'status') else "unknown",
                "cost": cost,
                "result": result.to_dict() if hasattr(result, 'to_dict') else str(result),
            }
            # JSON ONLY - no mixed output
            print(json.dumps(output, indent=2))
            return 0
        
        # Human-readable output
        console.print(f"[green]Done[/green] Cost: ${cost:.4f}")
        if hasattr(result, 'status'):
            console.print(f"Status: {result.status}")
        
        if result.hotspots:
            console.print(f"\n[bold]Hotspots found:[/bold] {len(result.hotspots)}")
            for h in result.hotspots[:5]:
                console.print(f"  - {h.get('file', 'unknown')}")
        
        if result.applied:
            console.print(f"[green]Applied fixes:[/green] {len(result.applied)}")
        
        # Log complete
        audit.log(
            "command_complete",
            command="improve",
            status="success",
            cost=cost,
        )
        
        return 0
        
    except Exception as e:
        # Error handling - log and return non-zero
        logger.exception(f"Command failed: {e}")
        
        audit.log(
            "command_error",
            command="improve",
            error=str(e),
        )
        
        if json_output:
            print(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"[red]Error: {e}[/red]")
        
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous code improvement pipeline")
    parser.add_argument("target", nargs="?", help="File or directory to improve")
    parser.add_argument("--goal", default="improve code quality", help="Improvement goal")
    parser.add_argument("--apply", action="store_true", help="Apply fixes automatically")
    parser.add_argument("--dry-run", action="store_true", help="Preview without applying")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--rollback", metavar="FILE", help="Rollback a file from .bak backup")
    
    args = parser.parse_args()
    exit(asyncio.run(run(
        target=args.target,
        goal=args.goal,
        apply=args.apply,
        dry_run=args.dry_run,
        json_output=args.json,
        rollback=args.rollback,
    )))
