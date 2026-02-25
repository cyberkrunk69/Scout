# TODO: This module is not yet integrated into the main application.
# It is planned for future use as part of the Batch Pipeline framework.
# See ADR-008 for design context.
from __future__ import annotations
"""Auto-spawn sub-batch functionality for executing plan steps."""
import asyncio
import os
import re
from typing import Any, Optional
from scout.batch_context import BatchContext
from scout.batch_pipeline import PipelineExecutor
from scout.batch_plan_parser import parse_plan_steps
from scout.batch_cli_discovery import (
    DISCOVERABLE_COMMANDS,
    CLI_MODULE_MAP,
    discover_cli_interface,
)

# Command inference keywords - maps intent keywords to Scout commands
COMMAND_INFERENCE_MAP = {
    "lint": ["lint", "linter", "flake8", "mypy", "check", "style"],
    "audit": ["audit", "logs", "history", "events"],
    "run": ["run", "execute", "test", "pytest", "unittest"],
    "nav": ["navigate", "find", "search", "where", "locate", "file"],
    "roast": ["roast", "critique", "review", "assess"],
    "query": ["query", "ask", "question", "explain"],
    "validate": ["validate", "verify", "check"],
    "index": ["index", "ctags", "symbols"],
    "doc_sync": ["doc", "documentation", "sync", "docs"],
    "env": ["env", "environment", "variables"],
    "git_status": ["git status", "status"],
    "git_branch": ["branch", "branches"],
    "git_diff": ["diff", "changes"],
    "git_log": ["log", "history", "commit"],
    "brief": ["brief", "summary", "overview"],
    "ci_guard": ["ci", "guard", "checks"],
    "status": ["status", "health"],
    "plan": ["plan", "plan generate", "implementation"],
    "git_show": ["show", "commit details"],
}


def _infer_command_from_description(description: str) -> str:
    """
    Infer the correct Scout command from a step description.
    
    Uses keyword matching against the COMMAND_INFERENCE_MAP.
    Returns "nav" as fallback if no match found.
    """
    if not description:
        return "nav"
    
    desc_lower = description.lower()
    best_match = "nav"
    best_score = 0
    
    for command, keywords in COMMAND_INFERENCE_MAP.items():
        score = sum(1 for kw in keywords if kw in desc_lower)
        if score > best_score:
            best_score = score
            best_match = command
    
    return best_match


class SubBatchOrchestrator:
    """
    Orchestrate sub-batch execution from plan steps.
    
    When extract_steps=true is set on a task that returns plan output,
    this orchestrator will:
    1. Parse steps from the plan output
    2. Convert steps to batch tasks (with command inference)
    3. Execute them as a sub-batch
    4. Aggregate results
    """
    
    MAX_DEPTH = 3  # Prevent infinite recursion
    
    def __init__(self, max_depth: int = MAX_DEPTH):
        self.max_depth = max_depth
        self._depth = 0
        self._discovered_interfaces: dict = {}
    
    async def _ensure_interface(self, command: str) -> dict:
        """Ensure CLI interface is discovered for a command."""
        if command in self._discovered_interfaces:
            return self._discovered_interfaces[command]
        
        # Get venv python path
        venv_python = os.environ.get("VIVARIUM_PYTHON", ".venv/bin/python")
        
        interface = await discover_cli_interface(command, venv_python)
        self._discovered_interfaces[command] = interface
        return interface
    
    async def execute_plan_steps(
        self,
        plan_output: str,
        task_runner,
        context: BatchContext,
        mode: str = "parallel"
    ) -> dict:
        """
        Execute steps from a plan output.
        
        Args:
            plan_output: The raw plan text or JSON
            task_runner: Function to execute single task
            context: BatchContext for state
            mode: Execution mode (parallel/sequential)
        
        Returns:
            Dict with sub_batch_results and summary
        """
        self._depth += 1
        
        if self._depth > self.max_depth:
            return {
                "error": f"Max recursion depth {self.max_depth} exceeded",
                "sub_batch_results": []
            }
        
        # Try to parse as JSON first
        steps = []
        try:
            import json
            # Handle both string and dict inputs (auto-json returns dict)
            if isinstance(plan_output, dict):
                parsed = plan_output
            else:
                parsed = json.loads(plan_output)
            
            if isinstance(parsed, dict) and "steps" in parsed:
                steps = parsed["steps"]
            elif isinstance(parsed, list):
                steps = parsed
            elif isinstance(parsed, dict) and "plan" in parsed:
                # Handle structured plan output with "plan" key
                plan_text = parsed.get("plan", "")
                steps = parse_plan_steps(plan_text)
            else:
                # Treat as markdown
                steps = parse_plan_steps(str(parsed))
        except (json.JSONDecodeError, ValueError, TypeError):
            # Parse as markdown
            steps = parse_plan_steps(str(plan_output))
        
        if not steps:
            return {
                "sub_batch_results": [],
                "summary": {"total": 0, "successes": 0, "failures": 0},
                "note": "No executable steps found"
            }
        
        # Convert steps to batch tasks (with command inference)
        tasks = await self._steps_to_tasks(steps)
        
        # Execute as sub-batch
        pipeline = PipelineExecutor(context, task_runner)
        results = await pipeline.run(tasks, mode=mode)
        
        self._depth -= 1
        
        # Aggregate results
        successes = sum(1 for r in results if r.get("status") == "success")
        
        return {
            "sub_batch_results": results,
            "summary": {
                "total": len(tasks),
                "successes": successes,
                "failures": len(tasks) - successes
            },
            "depth": self._depth
        }
    
    async def _steps_to_tasks(self, steps: list[dict]) -> list[dict]:
        """Convert parsed steps to batch tasks with command inference."""
        tasks = []
        
        for step in steps:
            # Extract command - infer from description if not specified
            explicit_command = step.get("command")
            description = step.get("description", "")
            
            if explicit_command:
                command = explicit_command
            else:
                # Infer command from description using keyword matching
                command = _infer_command_from_description(description)
            
            args = step.get("args", {})
            
            # Add description/query as context if not present
            if description and "query" not in args:
                args["query"] = description
            
            tasks.append({
                "command": command,
                "args": args,
                "step_id": step.get("id"),
                "_inferred": not explicit_command  # Track if command was inferred
            })
        
        return tasks
    
    def reset_depth(self):
        """Reset recursion depth for fresh execution."""
        self._depth = 0
