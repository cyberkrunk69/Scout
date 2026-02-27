#!/usr/bin/env python
"""Graph command - Query code relationship graph."""

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
    subcommand: str,
    symbol: Optional[str] = None,
    from_symbol: Optional[str] = None,
    to_symbol: Optional[str] = None,
    file: Optional[str] = None,
    repo: Optional[Path] = None,
    max_depth: int = 10,
    json_output: bool = False,
) -> int:
    """Run the graph command - MCP-free, full integration."""
    
    console = ConsoleOutput()
    session = _load_session_safe()
    audit = AuditLog()
    
    if repo is None:
        repo = Path.cwd()

    # Log start
    audit.log(
        "command_start",
        command="graph",
        subcommand=subcommand,
        symbol=symbol,
    )

    try:
        # Import graph module
        from scout import graph as graph_module
        
        result = None
        
        if subcommand == "callers":
            if not symbol:
                console.print("[red]Error: symbol required for callers[/red]")
                return 1
            callers = graph_module.get_callers(symbol, repo)
            
            if json_output:
                result = {
                    "symbol": symbol,
                    "ambiguous": any(c.ambiguous for c in callers),
                    "callers": [
                        {
                            "symbol": c.symbol,
                            "file": str(c.file),
                            "line": c.line,
                        }
                        for c in callers
                    ],
                }
            else:
                if not callers:
                    console.print("No results found")
                else:
                    console.print(f"[bold]Callers of {symbol}:[/bold]")
                    for c in callers:
                        ambiguity_note = " (AMBIGUOUS)" if c.ambiguous else ""
                        console.print(f"  - {c.symbol} (line {c.line}){ambiguity_note}")
                return 0
                
        elif subcommand == "callees":
            if not symbol:
                console.print("[red]Error: symbol required for callees[/red]")
                return 1
            callees = graph_module.get_callees(symbol, repo)
            
            if json_output:
                result = {
                    "symbol": symbol,
                    "ambiguous": any(c.ambiguous for c in callees),
                    "callees": [
                        {
                            "symbol": c.symbol,
                            "file": str(c.file),
                            "line": c.line,
                        }
                        for c in callees
                    ],
                }
            else:
                if not callees:
                    console.print("No results found")
                else:
                    console.print(f"[bold]Callees of {symbol}:[/bold]")
                    for c in callees:
                        ambiguity_note = " (AMBIGUOUS)" if c.ambiguous else ""
                        console.print(f"  - {c.symbol} (line {c.line}){ambiguity_note}")
                return 0
                
        elif subcommand == "trace":
            if not from_symbol or not to_symbol:
                console.print("[red]Error: from_symbol and to_symbol required for trace[/red]")
                return 1
            path = graph_module.trace_path(from_symbol, to_symbol, repo, max_depth)
            
            if json_output:
                result = {
                    "from": from_symbol,
                    "to": to_symbol,
                    "path": [
                        {
                            "from": step.from_symbol,
                            "to": step.to_symbol,
                            "edge_type": step.edge_type,
                        }
                        for step in path
                    ],
                }
            else:
                if not path:
                    console.print("No path found")
                else:
                    console.print(f"[bold]Path from {from_symbol} to {to_symbol}:[/bold]")
                    for i, step in enumerate(path):
                        console.print(f"  {i + 1}. {step.from_symbol} --> {step.to_symbol} ({step.edge_type})")
                return 0
                
        elif subcommand == "impact":
            if not file:
                console.print("[red]Error: file required for impact[/red]")
                return 1
            dependents = graph_module.impact_analysis(file, repo)
            
            if json_output:
                result = {
                    "file": file,
                    "impacted_symbols": sorted(list(dependents)),
                }
            else:
                if not dependents:
                    console.print("No impacted symbols found")
                else:
                    console.print(f"[bold]Symbols impacted by changes to {file}:[/bold]")
                    for sym in sorted(dependents):
                        console.print(f"  - {sym}")
                return 0
                
        elif subcommand == "usages":
            if not symbol:
                console.print("[red]Error: symbol required for usages[/red]")
                return 1
            usages = graph_module.find_usages(symbol, repo)
            
            if json_output:
                result = {
                    "symbol": symbol,
                    "usages": [
                        {
                            "symbol": u.symbol,
                            "file": str(u.file),
                            "line": u.line,
                            "usage_type": u.usage_type,
                        }
                        for u in usages
                    ],
                }
            else:
                if not usages:
                    console.print("No results found")
                else:
                    console.print(f"[bold]Usages of {symbol}:[/bold]")
                    for u in usages:
                        console.print(f"  - {u.symbol} (line {u.line}, {u.usage_type})")
                return 0
        else:
            console.print(f"[red]Unknown subcommand: {subcommand}[/red]")
            return 1
        
        # Output JSON if requested
        if json_output and result:
            print(json.dumps(result, indent=2))
        
        # Log complete
        audit.log(
            "command_complete",
            command="graph",
            subcommand=subcommand,
            status="success",
        )
        
        return 0
        
    except Exception as e:
        # Error handling - log and return non-zero
        logger.exception(f"Command failed: {e}")
        
        audit.log(
            "command_error",
            command="graph",
            error=str(e),
        )
        
        if json_output:
            print(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"[red]Error: {e}[/red]")
        
        return 1


def main():
    """Entry point for graph command."""
    parser = argparse.ArgumentParser(description="Query code relationship graph")
    parser.add_argument("subcommand", choices=["callers", "callees", "trace", "impact", "usages"],
                       help="Graph subcommand")
    parser.add_argument("args", nargs="*", help="Subcommand arguments")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum path depth for trace")
    
    args = parser.parse_args()
    
    # Parse subcommand-specific args
    symbol = None
    from_symbol = None
    to_symbol = None
    file = None
    
    if args.subcommand in ["callers", "callees", "usages"]:
        if args.args:
            symbol = args.args[0]
    elif args.subcommand == "trace":
        if len(args.args) >= 2:
            from_symbol = args.args[0]
            to_symbol = args.args[1]
    elif args.subcommand == "impact":
        if args.args:
            file = args.args[0]
    
    exit(asyncio.run(run(
        subcommand=args.subcommand,
        symbol=symbol,
        from_symbol=from_symbol,
        to_symbol=to_symbol,
        file=file,
        repo=args.repo,
        max_depth=args.max_depth,
        json_output=args.json,
    )))


if __name__ == "__main__":
    main()
