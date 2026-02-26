#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def _repo_root() -> Path:
    return Path.cwd().resolve()

def main() -> int:
    # Configure logging BEFORE any Scout modules are imported
    # Suppress verbose debug logs from third-party libraries
    logging.getLogger('watchdog').setLevel(logging.WARNING)
    logging.getLogger('fsevents').setLevel(logging.WARNING)
    logging.getLogger('markdown_it').setLevel(logging.WARNING)
    # Keep scout at INFO level for application progress
    logging.getLogger('scout').setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        prog="scout",
        description="Scout CLI - Independent AI Codebase Intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--trust-level",
        choices=["permissive", "normal", "strict"],
        default="normal",
        help="Trust verification strictness (default: normal)",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # --- Command Definitions ---
    subparsers.add_parser("chat", help="Start interactive REPL session")

    plan_p = subparsers.add_parser("plan", help="Generate implementation plan")
    plan_p.add_argument("query", nargs="?", help="Planning query")
    plan_p.add_argument("--files", nargs="*", help="Context files")
    plan_p.add_argument("--json", action="store_true", help="Output JSON")

    exec_p = subparsers.add_parser("execute", help="Execute a plan")
    exec_p.add_argument("plan_id", nargs="?", help="Plan ID or path to plan JSON")

    docs_p = subparsers.add_parser("docs", help="Generate documentation")
    docs_p.add_argument("files", nargs="*", help="Files to document")
    docs_p.add_argument("--output", "-o", help="Output directory")
    docs_p.add_argument("--recursive", "-r", action="store_true", help="Recursive")

    edit_p = subparsers.add_parser("edit", help="Edit file with AI assistance")
    edit_p.add_argument("file", help="File to edit")
    edit_p.add_argument("--prompt", "-p", required=True, help="Edit instruction")

    search_p = subparsers.add_parser("search", help="Search codebase")
    search_p.add_argument("pattern", help="Search pattern")
    search_p.add_argument("--path", help="Search path")

    nav_p = subparsers.add_parser("nav", help="Navigate to symbol")
    nav_p.add_argument("symbol", help="Symbol to navigate to")

    git_p = subparsers.add_parser("git", help="Git helpers")
    git_p.add_argument("subcommand", nargs="?", help="Git subcommand (commit, pr, status)")
    git_p.add_argument("--auto", action="store_true", help="Auto-generate commit message")

    config_p = subparsers.add_parser("config", help="View/edit configuration")
    config_p.add_argument("action", choices=["show", "get", "set"], default="show", nargs="?")
    config_p.add_argument("key", nargs="?")
    config_p.add_argument("value", nargs="?")

    subparsers.add_parser("status", help="Show session info")

    # Self-improvement pilot command
    self_improve_p = subparsers.add_parser("self-improve", help="Self-improvement pilot: analyze validation data and propose improvements")
    self_improve_p.add_argument("--tool", default="scout_plan", help="Tool to analyze (default: scout_plan)")
    self_improve_p.add_argument("--days", type=int, default=7, help="Analysis window in days (default: 7)")
    self_improve_p.add_argument("--dry-run", action="store_true", help="Generate suggestion without submitting")
    self_improve_p.add_argument("--budget-threshold", type=float, default=10.0, help="Max $/hour before skipping (default: 10.0)")

    # doc-sync command group
    doc_sync_p = subparsers.add_parser("doc-sync", help="Documentation sync commands")
    doc_sync_sub = doc_sync_p.add_subparsers(dest="doc_sync_subcommand")

    # query subcommand
    query_p = doc_sync_sub.add_parser("query", help="Generate docs for files matching a query")
    query_p.add_argument("query", nargs="?", help="Query to find relevant files")
    query_p.add_argument("--top-k", default=10, type=int, help="Number of files from BM25")
    query_p.add_argument("--confidence", default=50, type=int, help="Min confidence threshold")
    query_p.add_argument("--deep", action="store_true", help="Use LLM to refine file list")

    subparsers.add_parser("clear", help="Clear screen")
    subparsers.add_parser("version", help="Show version")

    # Improve command
    improve_p = subparsers.add_parser("improve", help="Autonomous code improvement pipeline")
    improve_p.add_argument("target", nargs="?", help="File or directory to improve")
    improve_p.add_argument("--goal", default="improve code quality", help="Improvement goal")
    improve_p.add_argument("--apply", action="store_true", help="Apply fixes automatically")
    improve_p.add_argument("--dry-run", action="store_true", help="Preview without applying")
    improve_p.add_argument("--json", action="store_true", help="JSON output")
    improve_p.add_argument("--rollback", metavar="FILE", help="Rollback from .bak backup")

    # Web command
    web_p = subparsers.add_parser("web", help="Web automation using browser agent")
    web_p.add_argument("goal", help="Natural language goal description")
    web_p.add_argument("--json", action="store_true", help="JSON output")
    web_p.add_argument("--refresh", action="store_true", help="Force replan (bypass cache)")
    web_p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    # Graph command
    graph_p = subparsers.add_parser("graph", help="Query code relationship graph")
    graph_sub = graph_p.add_subparsers(dest="graph_subcommand")
    
    callers_p = graph_sub.add_parser("callers", help="Find who calls a symbol")
    callers_p.add_argument("symbol", help="Symbol name")
    callers_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root")
    callers_p.add_argument("--json", action="store_true", help="JSON output")
    
    callees_p = graph_sub.add_parser("callees", help="Find what a symbol calls")
    callees_p.add_argument("symbol", help="Symbol name")
    callees_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root")
    callees_p.add_argument("--json", action="store_true", help="JSON output")
    
    trace_p = graph_sub.add_parser("trace", help="Find path between symbols")
    trace_p.add_argument("from_symbol", help="Source symbol")
    trace_p.add_argument("to_symbol", help="Target symbol")
    trace_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root")
    trace_p.add_argument("--max-depth", type=int, default=10, help="Maximum path depth")
    trace_p.add_argument("--json", action="store_true", help="JSON output")
    
    impact_p = graph_sub.add_parser("impact", help="Find affected symbols for a file")
    impact_p.add_argument("file", help="File path")
    impact_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root")
    impact_p.add_argument("--json", action="store_true", help="JSON output")
    
    usages_p = graph_sub.add_parser("usages", help="Find all usages of a symbol")
    usages_p.add_argument("symbol", help="Symbol name")
    usages_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root")
    usages_p.add_argument("--json", action="store_true", help="JSON output")

    # Git hooks
    on_commit_p = subparsers.add_parser("on-commit", help="Git post-commit hook")
    on_commit_p.add_argument("files", nargs="*", help="Changed files")
    
    prepare_p = subparsers.add_parser("prepare-commit-msg", help="Git prepare-commit-msg hook")
    prepare_p.add_argument("message_file", help="Path to commit message file")

    # Custom Help Command to handle "scout help <command>"
    help_p = subparsers.add_parser("help", help="Show help for a command")
    help_p.add_argument("subcommand", nargs="?", help="The command to get help for")

    # --- Logic ---
    args = parser.parse_args()
    load_dotenv(_repo_root() / ".env")

    # 1. Handle 'scout help <subcommand>'
    if args.command == "help":
        if args.subcommand:
            # Re-parse with the subcommand and --help flag
            parser.parse_args([args.subcommand, "--help"])
        else:
            parser.print_help()
        return 0

    # 2. Handle Default behavior (No command = chat)
    if args.command is None or args.command == "chat":
        # Check if stdin is a pipe (e.g. cat log.txt | scout)
        if not sys.stdin.isatty() and args.command is None:
            # If piped data exists, we might want to default to 'plan' or 'explain'
            # For now, we still enter REPL or you could route to a specific one-shot
            pass
        
        from scout.cli.repl import run_repl
        return asyncio.run(run_repl())

    # 3. Handle One-Shot Commands
    from scout.cli.commands import (
        plan, execute, docs, edit, search, nav, git, config, status, doc_sync_query, 
        improve, web, graph, on_commit, prepare_commit_msg
    )

    try:
        if args.command == "plan":
            return asyncio.run(plan.run(args.query, args.files, args.json))
        elif args.command == "execute":
            return asyncio.run(execute.run(args.plan_id))
        elif args.command == "docs":
            return asyncio.run(docs.run(args.files, args.output, args.recursive))
        elif args.command == "edit":
            return asyncio.run(edit.run(args.file, args.prompt))
        elif args.command == "search":
            return asyncio.run(search.run(args.pattern, args.path))
        elif args.command == "nav":
            return asyncio.run(nav.run(args.symbol, trust_level=args.trust_level))
        elif args.command == "git":
            return asyncio.run(git.run(args.subcommand, args.auto))
        elif args.command == "config":
            return config.run(args.action, args.key, args.value)
        elif args.command == "status":
            return status.run()
        elif args.command == "self-improve":
            from scout.cli.commands.self_improve import main as self_improve_main
            return self_improve_main()
        elif args.command == "doc-sync":
            if args.doc_sync_subcommand == "query":
                return asyncio.run(
                    doc_sync_query.run(
                        query=args.query,
                        top_k=args.top_k,
                        confidence=args.confidence,
                        deep=args.deep,
                    )
                )
            else:
                doc_sync_p.print_help()
                return 1
        elif args.command == "clear":
            os.system("clear" if os.name == "posix" else "cls")
            return 0
        elif args.command == "version":
            print("Scout CLI v1.0.0-jailbreak")
            return 0
        elif args.command == "improve":
            return asyncio.run(improve.run(
                target=args.target,
                goal=args.goal,
                apply=args.apply,
                dry_run=args.dry_run,
                json_output=args.json,
                rollback=args.rollback,
            ))
        elif args.command == "web":
            return asyncio.run(web.run(
                goal=args.goal,
                json_output=args.json,
                refresh=args.refresh,
                verbose=args.verbose,
            ))
        elif args.command == "graph":
            return asyncio.run(graph.run(
                subcommand=args.graph_subcommand,
                symbol=getattr(args, 'symbol', None),
                from_symbol=getattr(args, 'from_symbol', None),
                to_symbol=getattr(args, 'to_symbol', None),
                file=getattr(args, 'file', None),
                repo=getattr(args, 'repo', Path.cwd()),
                max_depth=getattr(args, 'max_depth', 10),
                json_output=args.json,
            ))
        elif args.command == "on-commit":
            return on_commit.run(files=args.files)
        elif args.command == "prepare-commit-msg":
            return prepare_commit_msg.run(message_file=args.message_file)
    except Exception as e:
        print(f"[bold red]Error:[/bold red] {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())