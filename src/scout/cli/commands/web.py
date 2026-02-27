#!/usr/bin/env python
"""Web command - Web automation using browser agent."""

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
    goal: str,
    json_output: bool = False,
    refresh: bool = False,
    verbose: bool = False,
) -> int:
    """Run the web automation command - MCP-free, full integration."""
    
    console = ConsoleOutput()
    session = _load_session_safe()
    audit = AuditLog()

    if not goal:
        console.print("[red]Error: goal required[/red]")
        return 1
    
    # Log start
    audit.log(
        "command_start",
        command="web",
        goal=goal,
    )

    try:
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        if not json_output:
            console.print(f"[dim]Executing web automation: {goal}[/dim]")
        
        # Import core modules
        from scout.batch_plan_parser import parse_plan_steps_with_json_first
        from scout.plan_executor import PlanExecutor, plan_result_to_dict
        from scout.plan_store import PlanStore
        from scout.tools.browser_agent import browser_act
        
        # Initialize stores
        repo_root = Path.cwd()
        plan_store = PlanStore(repo_root=repo_root)
        await plan_store.initialize()
        
        # Check plan cache first (unless refresh requested)
        plan_id = None
        steps = None
        
        if not refresh:
            cached = await plan_store.get_plan(goal)
            if cached:
                plan_id, steps = cached
                if not json_output:
                    console.print(f"[dim]Plan cache hit (plan_id: {plan_id})[/dim]")
        
        # Generate new plan if needed
        if steps is None:
            if not json_output:
                console.print(f"[dim]Generating plan...[/dim]")
            
            # Use scout_plan from tools (no legacy import)
            from scout.tools.llm import scout_plan
            
            try:
                plan_result = await scout_plan(request=goal)
                plan_text = plan_result.content if hasattr(plan_result, 'content') else str(plan_result)
                tokens = 0
                cost = getattr(plan_result, 'cost_usd', 0.02) or 0.02
                
                steps = parse_plan_steps_with_json_first(plan_text)
                
                # Filter to only browser_act steps
                browser_steps = [
                    s for s in steps 
                    if isinstance(s, dict) and s.get("command") == "browser_act"
                ]
                
                if not browser_steps:
                    console.print("[red]Error: No browser_act steps found in plan[/red]")
                    return 1
                
                steps = browser_steps
                
                # Store in cache
                plan_id = await plan_store.store_plan(goal, steps)
                
                if not json_output:
                    console.print(f"[dim]Plan generated and cached (plan_id: {plan_id})[/dim]")
                    
            except Exception as e:
                console.print(f"[red]Error generating plan: {e}[/red]")
                if verbose:
                    import traceback
                    traceback.print_exc()
                return 1
        
        # Execute plan
        if not json_output:
            console.print(f"[dim]Executing plan ({len(steps)} steps)...[/dim]")
        
        executor = PlanExecutor(browser_agent=browser_act, max_retries=1)
        
        result = await executor.execute_plan(steps, plan_id=plan_id)
        
        # Update plan stats
        await plan_store.update_plan_stats(plan_id, result.success)
        
        # Cost tracking
        cost = result.total_cost
        session.total_cost += cost
        
        try:
            save_session(session)
        except Exception as e:
            logger.warning(f"Session save failed: {e}")
        
        # Format output - branch cleanly
        if json_output:
            output = plan_result_to_dict(result)
            print(json.dumps(output, indent=2))
            return 0
        
        # Human-readable output
        if result.success:
            console.print(f"[green]Plan executed successfully[/green]")
        else:
            console.print(f"[red]Plan failed: {result.steps_failed} step(s) failed[/red]")
        
        console.print(f"Steps: {result.steps_executed}/{result.steps_executed + result.steps_failed} completed")
        console.print(f"Duration: {result.total_duration_ms / 1000:.1f}s")
        console.print(f"Cost: ${cost:.4f}")
        
        if result.extracted_data:
            console.print(f"Extracted: {len(result.extracted_data)} items")
            for key, value in result.extracted_data.items():
                console.print(f"  {key}: {value}")
        
        # Show step results
        console.print("\n[bold]Step Results:[/bold]")
        for sr in result.step_results:
            status = "[green]OK[/green]" if sr.success else "[red]FAIL[/red]"
            action = sr.action
            target = sr.target or ""
            
            if sr.success:
                console.print(f"  {sr.step_index}. {status}: {action} {target}")
                if sr.output and "extracted" in sr.output:
                    console.print(f"      -> {sr.output['extracted']}")
            else:
                console.print(f"  {sr.step_index}. {status}: {action} {target}")
                if sr.error:
                    console.print(f"      Error: {sr.error}")
                if sr.failure_reason:
                    console.print(f"      Reason: {sr.failure_reason}")
        
        # Log complete
        audit.log(
            "command_complete",
            command="web",
            status="success" if result.success else "failed",
            cost=cost,
            steps=result.steps_executed,
        )
        
        return 0 if result.success else 1
        
    except Exception as e:
        # Error handling - log and return non-zero
        logger.exception(f"Command failed: {e}")
        
        audit.log(
            "command_error",
            command="web",
            error=str(e),
        )
        
        if json_output:
            print(json.dumps({"status": "error", "error": str(e)}))
        else:
            console.print(f"[red]Error: {e}[/red]")
        
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Web automation using browser agent")
    parser.add_argument("goal", help="Natural language goal description")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--refresh", action="store_true", help="Force replan (bypass cache)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    exit(asyncio.run(run(
        goal=args.goal,
        json_output=args.json,
        refresh=args.refresh,
        verbose=args.verbose,
    )))
