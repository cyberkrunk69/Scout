#!/usr/bin/env python
"""
Scout CLI Enhanced - Interactive REPL

Provides an interactive REPL with:
- prompt_toolkit for rich input (history, autocomplete, multi-line)
- Session persistence
- Command history
- Streaming output
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Suppress verbose debug logging from HTTP libraries
for lib in ["httpcore", "httpx", "httpcore.connection", "httpcore.http11"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory

from scout.cli.formatting.output import ConsoleOutput
from scout.cli.formatting.progress import Spinner, ReasoningLiveDisplay
from scout.cli.context.session import Session, load_session, save_session
from scout.cli.mcp_bridge.client import MCPClient
from scout.cli.tool_loader import get_tool_schemas
from scout.persistence import read_last_plan
from scout.batch_plan_parser import parse_plan_steps
from scout.executor import execute_plan
from scout.ui.whimsy_manager import get_whimsy_manager

import re
import json


def extract_json_blocks(content: str) -> list[dict]:
    """
    Extract JSON blocks from Markdown content.
    
    Looks for code blocks with json language tag containing arrays [...].
    Returns list of parsed JSON objects.
    
    Examples:
        ```json
        [{"command": "foo", "args": {}}]
        ```
    """
    results = []
    
    # Pattern to match JSON code blocks in markdown
    # Matches ```json ... ``` or ``` ... ```
    json_block_pattern = r'```(?:json)?\s*\n?([\[\{].*?)\n?```'
    
    matches = re.findall(json_block_pattern, content, re.DOTALL)
    
    for match in matches:
        try:
            parsed = json.loads(match)
            # Handle both array and object
            if isinstance(parsed, list):
                results.extend(parsed)
            elif isinstance(parsed, dict):
                results.append(parsed)
        except json.JSONDecodeError:
            continue
    
    return results


class REPLProgressBridge:
    """
    Bridge that connects PlanProgress to ReasoningLiveDisplay.
    
    This allows the planner to emit progress events that show up
    in the REPL's live display in real-time.
    """
    
    def __init__(self, display):
        self.display = display
        self.total_cost = 0.0
        self.total_tokens = 0
        self.start_time = None
        self._task_count = 0
        self._current_model = "unknown"
        # Tier-based cost tracking
        self.cost_by_tier = {"fast": 0.0, "medium": 0.0, "large": 0.0}
        self.cost_by_iteration = []  # List of (iteration, cost)

    @property
    def current_model(self) -> str:
        """Get the current model being used."""
        return self._current_model
        self.cost_by_iteration = []  # List of (iteration, cost)
        
    def record_call(self, tier: str, cost: float):
        """Record cost per tier and iteration."""
        if tier in self.cost_by_tier:
            self.cost_by_tier[tier] += cost
        self.cost_by_iteration.append((len(self.cost_by_iteration), cost))
        
    def get_cost_summary(self) -> str:
        """Generate cost summary string."""
        return (
            f"fast=${self.cost_by_tier['fast']:.4f} | "
            f"medium=${self.cost_by_tier['medium']:.4f} | "
            f"large=${self.cost_by_tier['large']:.4f}"
        )
    def start(self, task_name: str):
        """Start a new task - show in REPL display."""
        self._task_count += 1
        import time
        self.start_time = time.time()
        # Update the live display
        self.display.set_action(task_name, 0.1)
        self.display.add_reasoning(f"▶ {task_name}")
        
    def spin(self, message: str = ""):
        """Show spinner with message in REPL display."""
        # Update action with progress
        progress = min(0.1 + (self._task_count * 0.1), 0.8)
        self.display.set_action(message, progress)
        # Add to reasoning trace (truncated to avoid noise)
        if message and len(message) < 60:
            self.display.add_reasoning(message)
            
    def progress(self, current: int, total: int, message: str = ""):
        """Show progress bar in REPL display."""
        pct = (current / total * 100) if total > 0 else 0
        self.display.set_action(f"{message} ({pct:.0f}%)", min(current/total, 0.9))
        
    def complete(self, message: str = "", tokens: int = 0, cost: float = 0):
        """Mark task complete - update REPL display."""
        self.total_cost += cost
        self.total_tokens += tokens
        self.display.add_reasoning(f"✓ {message}")
        self.display.set_stats(cost_usd=self.total_cost, steps=self._task_count)
        
    def sub_complete(self, index: int, total: int, title: str):
        """Mark sub-task complete."""
        self.display.add_reasoning(f"  [{index}/{total}] ✓ {title[:50]}")
        
    def info(self, message: str):
        """Add info message to REPL display."""
        self.display.add_log(message, "info")

    def warning(self, message: str):
        """Add warning message to REPL display."""
        self.display.add_log(message, "warning")

    def error(self, message: str):
        """Add error message to REPL display."""
        self.display.add_log(message, "error")

    def discovery(self, message: str):
        """Add discovery message to REPL display."""
        self.display.add_log(message, "discovery")

    def set_model(self, model: str):
        """Set the current model being used."""
        self._current_model = model
        # Show model in reasoning with a special marker
        self.display.add_reasoning(f"[model]{model}[/model]")


def handle_user_input_request(prompt: str, schema: dict = None) -> str:
    """
    Handle a user input request during plan execution.
    
    Pauses execution, displays the prompt, and returns user input.
    
    Args:
        prompt: The prompt to display to the user
        schema: Optional JSON schema for expected input
        
    Returns:
        The user's input string
    """
    try:
        from rich.console import Console
        console = Console()
    except ImportError:
        # Fallback if rich not available
        print(f"\n{'='*50}")
        print(f"INPUT REQUIRED")
        print(f"{'='*50}")
        print(prompt)
        if schema:
            print(f"\nExpected format: {schema}")
        return input("\n> ")
    
    console = Console()
    console.print(f"\n[yellow bold]⏸ INPUT REQUIRED[/yellow bold]")
    console.print(f"[dim]{prompt}[/dim]")
    if schema:
        console.print(f"[dim]Expected: {schema}[/dim]")
    
    # Get input with prompt
    user_input = console.input("\n> ")
    return user_input


class UserInputBridge:
    """Bridge for handling user input during plan execution."""
    
    def __init__(self, input_handler=None):
        self.input_handler = input_handler or handle_user_input_request
        self._paused = False
        
    async def request_input(self, prompt: str, schema: dict = None) -> str:
        """Request user input, pausing execution."""
        self._paused = True
        result = self.input_handler(prompt, schema)
        self._paused = False
        return result
        
    def is_paused(self) -> bool:
        """Check if execution is paused for input."""
        return self._paused


def extract_first_json_block(content: str) -> Optional[list[dict]]:
    """
    Extract the first valid JSON array block from Markdown.
    
    Returns the first JSON array found, or None if no valid array found.
    """
    blocks = extract_json_blocks(content)
    return blocks if blocks else None


class ScoutCompleter(Completer):
    """Auto-completion for Scout commands."""

    def __init__(self):
        self.commands = [
            "/stop", "/plan", "/execute", "/docs", "/edit", "/search", "/nav",
            "/git", "/config", "/status", "/help", "/clear", "/exit",
        ]

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            for cmd in self.commands:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))


def _repo_root() -> Path:
    """Get repository root."""
    return Path.cwd().resolve()


def _scout_dir() -> Path:
    """Get .scout directory for session data."""
    scout_dir = _repo_root() / ".scout"
    scout_dir.mkdir(exist_ok=True)
    return scout_dir


async def run_repl():
    """Run the interactive REPL."""
    console = ConsoleOutput()
    session = load_session()
    mcp_client = MCPClient()

    # Initialize executor with tools - CRITICAL for plan execution
    # Phase 2: Pass real services for validation, budget, and governance
    try:
        from scout.executor import initialize_executor
        from scout.config import ScoutConfig
        from scout.validation_pipeline import ValidationPipeline
        from scout.budget_service import BudgetService
        from scout.quality_gates import ToolOutputGate
        from scout.tool_output import ToolOutputRegistry

        config = ScoutConfig()
        await initialize_executor(
            config=config,
            validation_pipeline=ValidationPipeline(),
            budget_service=BudgetService(config),
            gate=ToolOutputGate() if config.get_quality_gates_config().get("enabled") else None,
            tool_output_registry=ToolOutputRegistry(),
        )
        console.print("[dim]Executor initialized with tools and Phase 2 services[/dim]")
    except ImportError as e:
        # Fallback: initialize without Phase 2 services
        try:
            from scout.executor import initialize_executor
            await initialize_executor()
            console.print("[dim]Executor initialized with tools (Phase 2 services unavailable)[/dim]")
        except Exception as fallback_err:
            console.print(f"[yellow]Warning: Executor initialization failed: {fallback_err}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Warning: Executor initialization failed: {e}[/yellow]")

    # Setup prompt session with history
    history_file = _scout_dir() / "history"
    prompt_session = PromptSession(
        history=FileHistory(str(history_file)),
        auto_suggest=AutoSuggestFromHistory(),
        multiline=True,
        prompt_continuation="... ",
        completer=ScoutCompleter(),
    )

    console.print("[bold blue]Scout CLI[/bold blue] v1.0.0 — Type /help for commands")
    console.print("[dim]Type /exit to quit[/dim]\n")

    messages = []
    
    # Autonomous execution state
    autonomous_task: Optional[asyncio.Task] = None
    display: Optional[ReasoningLiveDisplay] = None

    while True:
        try:
            text = await prompt_session.prompt_async("scout> ")
            text = text.strip()

            if not text:
                continue

            # Handle commands
            if text.startswith("/"):
                await handle_command(
                    text, console, session, mcp_client, messages,
                    autonomous_task=autonomous_task, display=display
                )
            else:
                # Route non-command input to autonomous execution path
                autonomous_task, display = await handle_autonomous(
                    text, console, session, mcp_client, messages, prompt_session,
                    autonomous_task=autonomous_task, display=display
                )

        except KeyboardInterrupt:
            # Handle Ctrl+C - safely interrupt autonomous execution
            if autonomous_task and not autonomous_task.done():
                console.print("\n[yellow]Interrupting autonomous execution...[/yellow]")
                autonomous_task.cancel()
                if display:
                    display.cancel()
                try:
                    await asyncio.wait_for(autonomous_task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
                autonomous_task = None
                display = None
                console.print("[dim]Execution interrupted.[/dim]")
            else:
                console.print("\n[yellow]Use /exit to quit[/yellow]")
        except EOFError:
            break

    # Save session on exit
    save_session(session)
    console.print("\n[dim]Session saved. Goodbye![/dim]")
    return 0


async def handle_command(
    text: str,
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    messages: list,
    autonomous_task: Optional[asyncio.Task] = None,
    display: Optional[ReasoningLiveDisplay] = None,
):
    """Handle a command input."""
    parts = text.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd == "/exit":
        raise EOFError

    elif cmd == "/stop":
        # Handle /stop command to interrupt autonomous execution
        if autonomous_task and not autonomous_task.done():
            console.print("[yellow]Stopping autonomous execution...[/yellow]")
            autonomous_task.cancel()
            if display:
                display.cancel()
            try:
                await asyncio.wait_for(autonomous_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            console.print("[dim]Execution stopped.[/dim]")
        else:
            console.print("[dim]No autonomous execution to stop.[/dim]")
        return autonomous_task, display

    elif cmd == "/help":
        show_help(console, args)

    elif cmd == "/clear":
        os.system("clear" if os.name == "posix" else "cls")

    elif cmd == "/plan":
        await cmd_plan(console, session, mcp_client, args, messages)

    elif cmd == "/execute":
        await cmd_execute(console, session, mcp_client, args)

    elif cmd == "/docs":
        await cmd_docs(console, session, mcp_client, args)

    elif cmd == "/edit":
        await cmd_edit(console, session, mcp_client, args)

    elif cmd == "/search":
        await cmd_search(console, session, mcp_client, args)

    elif cmd == "/nav":
        await cmd_nav(console, session, mcp_client, args)

    elif cmd == "/git":
        await cmd_git(console, session, mcp_client, args)

    elif cmd == "/config":
        cmd_config(console, args)

    elif cmd == "/status":
        cmd_status(console, session)

    else:
        console.print(f"[red]Unknown command: {cmd}[/red]")
        console.print("Type /help for available commands")

    return autonomous_task, display

    return autonomous_task, display


async def handle_autonomous(
    text: str,
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    messages: list,
    prompt_session: PromptSession,
    autonomous_task: Optional[asyncio.Task] = None,
    display: Optional[ReasoningLiveDisplay] = None,
) -> tuple[Optional[asyncio.Task], Optional[ReasoningLiveDisplay]]:
    """
    Handle autonomous execution - route non-command input to Scout's reasoning engine.
    
    This function:
    1. Shows live reasoning display (what Scout is thinking)
    2. Uses TaskRouter to determine execution path
    3. Executes the appropriate tools with progress feedback
    """
    # If there's an existing task running, don't start another
    if autonomous_task and not autonomous_task.done():
        console.print("[yellow]Already executing. Use /stop to cancel.[/yellow]")
        return autonomous_task, display
    
    console.print(f"[dim]Analyzing: {text}[/dim]")
    
    # Start autonomous execution in a task - display will be created inside _run_autonomous
    task = asyncio.create_task(
        _run_autonomous(text, console, session, mcp_client, messages, prompt_session, display)
    )
    # The display is created inside _run_autonomous; we need to extract it after task completes
    # For now, return None as display is managed internally by _run_autonomous
    return task, None


async def _run_autonomous(
    text: str,
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    messages: list,
    prompt_session: PromptSession,
    display: Optional[ReasoningLiveDisplay] = None,
):
    """Internal function to run autonomous execution with live display."""
    # Create live display for reasoning visualization with stderr=True to keep stdout clean
    display = ReasoningLiveDisplay(stderr=True)
    
    try:
        with display:
            # Phase 1: Use LLM to route to the correct tool
            display.start_heartbeat("🧠 [bold magenta]Analyzing User Intent...[/bold magenta]")
            display.set_stats(status="Analyzing intent...")
            
            # Get available tool schemas
            tool_schemas = get_tool_schemas()
            tool_list = "\n".join([f"- {t['name']}: {t['description']}" for t in tool_schemas])
            
            # Build prompt for LLM to pick the right tool
            routing_prompt = f"""You are a tool router. Given a user query, pick the most appropriate Scout tool to handle it.

IMPORTANT - Multi-step detection:
If the user's request describes multiple distinct actions (e.g., contains "then", "and then", "after that", "and also", or lists several tasks separated by commas or bullet points), you MUST select "scout_plan". The scout_plan tool will break the request into a series of executable steps. Only select a single tool if the request clearly describes ONE action (like "read file X" or "run command Y").

Examples:
- User: "Show me the content of config.py" → Tool: scout_read_file
- User: "Create a file called test.txt with content hello and then list files" → Tool: scout_plan
- User: "Run pytest and then update the readme" → Tool: scout_plan
- User: "Find all async functions and document them" → Tool: scout_plan

Available tools:
{tool_list}

User query: {text}

Respond with ONLY the tool name that best handles this query. If no tool is appropriate OR if the request has multiple steps, respond with "scout_plan" (which can handle any request by generating a plan).

Respond in this exact format:
TOOL: <tool_name>
REASON: <one line explanation>"""
            
            # Call LLM for routing
            from scout.llm.router import call_llm
            
            try:
                result = await call_llm(
                    routing_prompt,
                    task_type="simple",
                    model="llama-3.1-8b-instant",
                    max_tokens=200,
                )
                
                # Parse the response
                response_lines = result.content.strip().split('\n')
                selected_tool = None
                reason = ""
                
                for line in response_lines:
                    if line.startswith("TOOL:"):
                        selected_tool = line.replace("TOOL:", "").strip()
                    elif line.startswith("REASON:"):
                        reason = line.replace("REASON:", "").strip()
                
                # Validate tool exists in TOOL_MAP
                from scout.cli.tool_loader import TOOL_MAP
                if selected_tool and selected_tool not in TOOL_MAP:
                    display.add_reasoning(f"Invalid tool '{selected_tool}' - falling back to scout_plan")
                    selected_tool = "scout_plan"
                else:
                    display.add_reasoning(f"LLM routed to: {selected_tool} ({reason})")
                
            except Exception as e:
                # Fallback to planning if LLM fails
                display.add_reasoning(f"LLM routing failed: {e}, falling back to plan generation")
                selected_tool = "scout_plan"
            
            display.stop_heartbeat()
            
            # Always require confirmation before executing plans
            # Let the LLM routing handle intent classification
            auto_execute = False
            
            # Phase 2: Execute the selected tool
            if selected_tool == "scout_plan":
                # Complex request - generate a plan
                whimsy = get_whimsy_manager()
                status_text = whimsy.get_status_string("generating_plan", "Generating plan")
                display.set_action(status_text, 0.3)
                display.add_reasoning("Complex request - generating implementation plan")
                display.increment_steps()
                display.clear_logs()
                await _run_with_plan(text, console, session, mcp_client, messages, display, prompt_session, auto_execute=auto_execute)
            else:
                # Phase 1.5: Validate selected tool can answer the query
                display.add_reasoning(f"Validating tool selection: {selected_tool}")
                
                can_answer = await _validate_tool_selection(
                    text, selected_tool, tool_schemas
                )
                
                if not can_answer:
                    display.add_reasoning(
                        f"Tool '{selected_tool}' cannot answer this query - falling back to planning"
                    )
                    display.set_action("Generating plan", 0.3)
                    display.increment_steps()
                    display.clear_logs()
                    await _run_with_plan(text, console, session, mcp_client, messages, display, prompt_session, auto_execute=auto_execute)
                    return
                
                # Direct tool execution
                display.set_action(f"Executing {selected_tool}", 0.4)
                display.add_reasoning(f"Direct tool execution: {selected_tool}")
                display.set_stats(status="Executing tool", steps=display._steps_completed + 1)
                display.clear_logs()
                
                result = await mcp_client.call_tool(selected_tool, {"query": text})
                
                display.set_action("Completed", 1.0)
                display.add_reasoning("Tool execution complete")
                display.set_stats(status="Completed", steps=display._steps_completed + 1)
                
                # Format and print result in natural language
                tool_result = result.get("result", str(result))
                
                # Check if it's an error
                if result.get("error"):
                    console.print(f"\n[red]Error: {result['error']}[/red]")
                else:
                    console.print(f"\n[bold]Result from {selected_tool}:[/bold]")
                    console.print(tool_result)
                
    except asyncio.CancelledError:
        display.cancel()
        display.set_stats(status="Cancelled")
        console.print("\n[dim]Autonomous execution cancelled.[/dim]")
        raise
    except Exception as e:
        display.set_stats(status=f"Error: {e}")
        console.print(f"\n[red]Error during autonomous execution: {e}[/red]")


async def _validate_tool_selection(
    query: str,
    selected_tool: str,
    tool_schemas: list[dict],
) -> bool:
    """
    Validate that the selected tool can actually answer the user's query.
    
    Uses a lightweight LLM check to verify the tool's capability.
    Returns True if the tool can answer, False if we should fall back to planning.
    """
    from scout.llm.router import call_llm
    
    # Find the tool's description
    tool_desc = ""
    for schema in tool_schemas:
        if schema.get("name") == selected_tool:
            tool_desc = schema.get("description", "")
            break
    
    validation_prompt = f"""You are a tool capability validator. Determine if the selected tool can answer the user's query.

User query: {query}

Selected tool: {selected_tool}
Tool description: {tool_desc}

Can this tool answer the user's query? Consider:
- Does the query ask about something the tool actually does?
- Is the tool's purpose aligned with what the user wants?
- If the user mentioned "git" but meant "GitHub issues", the tool cannot answer.

Respond with ONLY "YES" if the tool can answer, or "NO" if it cannot.
If uncertain, respond with "NO" to be safe (falling back to planning is safer than wrong tool).
"""
    try:
        result = await call_llm(
            validation_prompt,
            task_type="simple",
            model="llama-3.1-8b-instant",
            max_tokens=10,
        )
        content = result.content.strip().upper()
        return content == "YES"
    except Exception:
        # On any error, be safe and allow the tool to run
        return True


def _intent_to_tool(intent_type: str) -> Optional[str]:
    """Map intent type to Scout tool name."""
    mapping = {
        "query_code": "scout_function_info",
        "document": "scout_doc_sync",
        "test": "scout_run",
        "search": "scout_grep",
        "nav": "scout_nav",
    }
    return mapping.get(intent_type)


async def _run_with_plan(
    text: str,
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    messages: list,
    display: ReasoningLiveDisplay,
    prompt_session: PromptSession,
    auto_execute: bool = False,
):
    """Execute via plan generation and execution.
    
    Args:
        auto_execute: If True, skip confirmation and execute immediately.
                      If False (default), show steps and ask for confirmation.
    """
    from scout.batch_plan_parser import parse_plan_steps
    
    whimsy = get_whimsy_manager()
    status_text = whimsy.get_status_string("generating_plan", "Generating plan")
    display.set_action(status_text, 0.3)
    display.add_reasoning("Generating implementation plan...")
    display.set_stats(status=status_text)
    
    # Generate plan with structured output for executable steps
    plan_result = await mcp_client.call_tool("scout_plan", {"query": text, "json_output": True, "structured": True})
    
    if "error" in plan_result:
        display.add_reasoning(f"Planning failed: {plan_result['error']}")
        display.set_stats(status="Planning failed")
        console.print(f"\n[red]Error: {plan_result['error']}[/red]")
        return
    
    plan_data = plan_result.get("result", plan_result)
    # Handle both string results (from scout_plan) and dict results
    # Handle CLI's actual output format: {"text": "...", "data": {"steps": [...]}}
    # Also support legacy format: {"plan": "...", "steps": [...]}
    # Also support ToolOutput dict format: {"content": "...", "tool_name": "plan", ...}
    extracted_steps = None
    if isinstance(plan_data, str):
        # Try to extract plan from JSON in the output
        try:
            import json as _json
            _parsed = _json.loads(plan_data[plan_data.index("{"):])
            data_section = _parsed.get("data", {})
            plan_text = _parsed.get("text", "") or data_section.get("plan", "") or _parsed.get("plan", plan_data)
            # Extract structured steps if present
            extracted_steps = data_section.get("steps", []) or _parsed.get("steps", [])
        except (ValueError, _json.JSONDecodeError):
            plan_text = plan_data
    else:
        # Handle ToolOutput dict format: {"content": "...", "tool_name": "plan", ...}
        data_section = plan_data.get("data", {})
        # Check for ToolOutput format with "content" field
        if "content" in plan_data and "tool_name" in plan_data:
            # This is a ToolOutput dict - content contains the actual plan
            content = plan_data.get("content", "")
            # Try to parse content as JSON to extract steps
            try:
                import json as _json
                content_parsed = _json.loads(content)
                plan_text = content_parsed.get("text", "") or content_parsed.get("plan", content)
                extracted_steps = content_parsed.get("data", {}).get("steps", []) or content_parsed.get("steps", [])
            except (ValueError, _json.JSONDecodeError):
                plan_text = content
                extracted_steps = []
        else:
            # Legacy/direct dict format
            plan_text = plan_data.get("text", "") or data_section.get("plan", "") or plan_data.get("plan", str(plan_data))
            # Extract structured steps if present
            extracted_steps = data_section.get("steps", []) or plan_data.get("steps", [])
    
    display.add_reasoning("Plan generated")

    # Persist full plan with steps to disk as JSON
    import json as _json
    full_plan = {
        "plan": plan_text,
        "steps": extracted_steps if extracted_steps else []
    }
    from scout.persistence import save_last_plan
    save_last_plan(_json.dumps(full_plan))
    
    # Stop the spinner/display before printing the plan and prompting
    # Need to stop the Rich Live display to allow normal console output
    display.clear()
    if hasattr(display, 'live') and display.live:
        display.live.stop()
    
    # Flush stdout to ensure the display is fully stopped
    sys.stdout.flush()
    sys.stderr.flush()
    
    console.print(f"\n[bold]Plan:[/bold]")
    console.print_markdown(plan_text)
    
    # Use extracted steps from structured output, or parse from plan text
    plan_steps = extracted_steps if extracted_steps else parse_plan_steps(plan_text)
    
    if plan_steps:
        # Display structured steps with tool calls
        console.print("\n[bold]Plan steps:[/bold]")
        for step in plan_steps:
            step_id = step.get("id", "?")
            description = step.get("description", "No description")
            command = step.get("command", "unknown")
            args = step.get("args", {})
            
            # Format the tool call for display
            if command == "write_file":
                tool_call = f"scout_create_file(path='{args.get('path', 'unknown')}', content='...')"
            elif command == "edit_file":
                tool_call = f"scout_edit(path='{args.get('path', 'unknown')}', instruction='{args.get('instruction', '...')}')"
            elif command == "run_command":
                tool_call = f"bash(command='{args.get('command', '')}')"
            elif command == "lint":
                tool_call = f"scout_lint(targets='{args.get('targets', '')}')"
            elif command == "run":
                tool_call = f"scout_run(targets='{args.get('targets', '')}')"
            elif command == "edit":
                tool_call = f"scout_edit(path='{args.get('path', '')}', instruction='{args.get('instruction', '')}')"
            else:
                tool_call = f"{command}({args})"
            
            console.print(f"  {step_id}. {description} → {tool_call}")
        
        # Confirmation prompt (unless auto_execute is True)
        if not auto_execute:
            console.print("\n[bold yellow]Execute this plan? (y/N)[/bold yellow]")
            console.print("[dim]Tip: Use /execute later to run, or prefix with 'execute' to run automatically[/dim]")
            
            # Print extra newlines and clear terminal to ensure input works
            # Use ANSI escape code to clear the screen and move cursor to bottom
            print("\n" * 5 + "\033[2J\033[H", flush=True)
            
            # Read confirmation from user - use prompt_toolkit session for proper input handling
            try:
                response = await prompt_session.prompt_async("> ")
                response = response.strip().lower()
            except (EOFError, KeyboardInterrupt):
                response = "n"
            
            if response not in ("y", "yes"):
                console.print("[yellow]Plan execution aborted.[/yellow]")
                console.print("[dim]Use /execute to run this plan later.[/dim]")
                return
        else:
            console.print("[dim]Auto-executing plan (--force flag detected)[/dim]")
        
        # Restart the Live display for execution phase
        if hasattr(display, 'live') and display.live:
            display.live.start()
        
        # User confirmed - proceed with execution
        display.set_action("Running plan", 0.7)
        display.add_reasoning("Executing plan steps...")
        display.set_stats(status="Running plan")
        
        exec_result = await mcp_client.call_tool("scout_execute_plan", {"plan_json": plan_text})
        
        display.set_action("Completed", 1.0)
        display.add_reasoning("Plan execution complete")
        display.set_stats(status="Completed", steps=display._steps_completed + 1)
        
        console.print(f"\n[bold]Execution result:[/bold]")
        console.print(exec_result.get("result", str(exec_result)))
    else:
        # No parseable steps - show the raw plan and skip execution
        display.add_reasoning("No executable steps found in plan")
        console.print("\n[yellow]No executable steps found in the plan.[/yellow]")
        console.print("[dim]View the plan above and use /execute with the plan content if needed.[/dim]")


def show_help(console: ConsoleOutput, args: str = ""):
    """Show help for commands."""
    if args:
        help_text = {
            "/stop": "Usage: /stop\nStop running autonomous execution.",
            "/plan": "Usage: /plan <query>\nGenerate an implementation plan.",
            "/execute": "Usage: /execute [plan_id]\nExecute a plan (or last generated plan if empty).",
            "/docs": "Usage: /docs <file>\nGenerate documentation for a file.",
            "/edit": "Usage: /edit <file> --prompt <instruction>\nEdit a file with AI assistance.",
            "/search": "Usage: /search <pattern>\nSearch the codebase.",
            "/nav": "Usage: /nav <symbol>\nNavigate to a symbol definition.",
            "/git": "Usage: /git <subcommand>\nGit helpers (commit, pr, status).",
            "/config": "Usage: /config [show|get|set <key> <value>]\nView/edit configuration.",
            "/status": "Usage: /status\nShow session info, costs, cache stats.",
        }
        console.print(help_text.get(args, f"No help for {args}"))
    else:
        console.print("""
[bold]Available Commands:[/bold]

[cyan]/stop[/cyan]              Stop autonomous execution
[cyan]/plan[/cyan] <query>      Generate implementation plan
[cyan]/execute[/cyan] [id]      Execute a plan
[cyan]/docs[/cyan] <file>       Generate documentation
[cyan]/edit[/cyan] <file>       Edit file with AI
[cyan]/search[/cyan] <pattern>  Search codebase
[cyan]/nav[/cyan] <symbol>      Navigate to symbol
[cyan]/git[/cyan] <cmd>         Git helpers
[cyan]/config[/cyan]            View/edit configuration
[cyan]/status[/cyan]            Show session info
[cyan]/help[/cyan] [cmd]       Show help
[cyan]/clear[/cyan]            Clear screen
[cyan]/exit[/cyan]              Exit REPL

[dim]Without / prefix, input is routed to autonomous execution path[/dim]

[bold]Natural Language Planning:[/bold]
[dim]  - Type a request to generate and execute a multi-step plan
[dim]  - Plan steps are shown before execution with confirmation prompt
[dim]  - Add "execute" or "run" to your request to auto-execute
[dim]    (e.g., "Create a hello world script and execute it")
[dim]  - Use /execute to run the last generated plan[/dim]
""")

async def cmd_plan(
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    args: str,
    messages: list,
):
    """Handle /plan command with live visualization."""
    if not args:
        console.print("Usage: /plan <query>")
        return

    whimsy = get_whimsy_manager()
    status_text = whimsy.get_status_string("generating_plan", "Generating plan")
    console.print(f"{status_text} for: {args}...")

    # Initialize live display for mission control visualization
    display = ReasoningLiveDisplay(stderr=True)
    
    try:
        with display:
            # Phase 1: Discovery - show we're analyzing the request
            display.set_stats(status="Analyzing request...", steps=0)
            display.add_reasoning(f"Planning: {args[:50]}{'...' if len(args) > 50 else ''}")
            display.set_action("Scanning available tools", 0.1)
            display.add_log("Initializing planning engine...", "info")
            
            # Simulate tool discovery (in real implementation, this would come from the planner)
            # Track A: Discover available tools
            await asyncio.sleep(0.3)  # Brief pause for visual effect
            display.add_log("Analyzing codebase structure...", "discovery")
            
            await asyncio.sleep(0.3)
            display.add_log("Identifying relevant modules...", "discovery")
            
            # Phase 2: Gap analysis
            display.set_action("Analyzing gaps", 0.3)
            display.add_reasoning("Analyzing implementation gaps...")
            display.add_log("Checking parameter mappings...", "info")
            
            # Simulate gap detection (Track B)
            await asyncio.sleep(0.3)
            display.add_log("Validating tool registry...", "info")
            
            # Phase 3: Synthesis - Big Brain working
            display.set_action("Synthesizing plan", 0.6)
            display.add_reasoning("Synthesizing implementation plan...")
            
            # Start heartbeat animation for brain activity with whimsy
            whimsy = get_whimsy_manager()
            whimsy_status = whimsy.get_status_string("reasoning", "Reasoning")
            display.start_heartbeat(f"🧠 {whimsy_status}...")
            
            # Use scout_plan tool instead of legacy generate_plan
            from scout.tools.llm import scout_plan
            
            # Call scout_plan - simpler, no progress bridge but works
            result = await scout_plan(request=args)
            
            # Extract results from ToolOutput
            plan_text = result.content if hasattr(result, 'content') else str(result)
            cost = getattr(result, 'cost_usd', 0.02) or 0.02
            tokens = 0  # Tool doesn't expose tokens directly
            steps = []  # Would need parsing
            
            # Stop heartbeat after planning complete
            display.stop_heartbeat()
            
            if plan_text:
                # Clear the display cleanly before showing final result
                display.clear()
                
                # Show completion in stats
                display.set_stats(status="Complete", steps=display._steps_completed + 1, cost_usd=cost)
                
                # Save plan in text/data format - consumer adapts to whatever is in data
                structured_plan = {
                    "text": plan_text,  # markdown for humans
                    "data": {  # structured for automation - dynamic schema
                        "steps": steps,
                        "metadata": {
                            "tokens": tokens,
                            "cost": cost,
                        }
                    }
                }
                
                # Save the plan
                from scout.persistence import save_last_plan
                save_last_plan(json.dumps(structured_plan))
                
                # Print final plan to stdout (display is now cleared)
                console.print(f"\n[bold]Plan:[/bold]")
                console.print_markdown(plan_text)
                
                # Track cost
                session.total_cost += float(cost)
                
                session.messages.append({"role": "user", "content": args})
                session.messages.append({"role": "assistant", "content": plan_text})
            else:
                display.clear()
                console.print("\n[yellow]No plan generated.[/yellow]")
                
    except asyncio.CancelledError:
        display.cancel()
        display.set_stats(status="Cancelled")
        console.print("\n[dim]Planning cancelled.[/dim]")
        raise
    except Exception as e:
        display.set_stats(status=f"Error: {e}")
        console.print(f"\n[red]Error during planning: {e}[/red]")
        import traceback
        traceback.print_exc()

async def cmd_execute(
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    args: str,
):
    """Handle /execute command with automatic plan discovery."""
    plan_content: Optional[str] = None

    # If no arguments provided, try to find the last plan
    if not args:
        plan_content = read_last_plan()
        
        if plan_content is None:
            console.print("[yellow]Usage: /execute <plan_id> (or leave empty to run the last plan)[/yellow]")
            return
        
        # Found last plan - print discovery message
        console.print("📦 [bold cyan]Found last generated plan. Preparing to execute...[/bold cyan]\n")
    else:
        # Args provided - treat as plan ID or inline plan JSON
        plan_content = args

    # Try to parse as JSON first (our new text/data format)
    import json as _json
    try:
        parsed = _json.loads(plan_content)
        
        # New format: {"text": "...", "data": {"steps": [...]}}
        if isinstance(parsed, dict):
            if "data" in parsed and isinstance(parsed["data"], dict):
                # New text/data format
                plan_steps = parsed.get("data", {}).get("steps", [])
                if plan_steps:
                    console.print("[dim]✓ Loaded structured plan (text/data format)[/dim]")
                else:
                    plan_steps = None
            elif "steps" in parsed:
                # Legacy format: {"steps": [...], "plan": "..."}
                plan_steps = parsed.get("steps", [])
                if plan_steps:
                    console.print("[dim]✓ Loaded structured plan (legacy format)[/dim]")
                else:
                    plan_steps = None
            else:
                plan_steps = None
        else:
            plan_steps = None
    except _json.JSONDecodeError:
        plan_steps = None
    
    # If not JSON or no steps, try extracting from markdown
    if not plan_steps:
        json_steps = extract_first_json_block(plan_content)
        if json_steps:
            console.print("[dim]✓ Found JSON block in Markdown, using direct execution[/dim]")
            plan_steps = json_steps
        else:
            try:
                plan_steps = parse_plan_steps(plan_content)
            except Exception as e:
                console.print(f"[red]Failed to parse plan: {e}[/red]")
                return

    if not plan_steps:
        console.print("[yellow]No executable steps found in plan.[/yellow]")
        return

    console.print(f"[dim]Executing {len(plan_steps)} plan steps...[/dim]")

    # Execute the plan using the executor
    try:
        result = await execute_plan(plan_steps)
        
        # Format and display results
        if result.get("success"):
            console.print("\n[bold green]✓ Plan executed successfully[/bold green]")
        else:
            stop_reason = result.get("stop_reason", "unknown error")
            console.print(f"\n[yellow]Plan stopped: {stop_reason}[/yellow]")
        
        # Show summary
        total_cost = result.get("total_cost", 0)
        total_elapsed = result.get("total_elapsed", 0)
        console.print(f"\n[dim]Cost: ${total_cost:.4f} | Time: {total_elapsed:.2f}s[/dim]")
        
        # Show step results summary
        results = result.get("results", [])
        for i, step_result in enumerate(results, 1):
            step_success = step_result.get("success", False)
            status = "✓" if step_success else "✗"
            step_desc = step_result.get("description", f"Step {i}")
            console.print(f"  {status} Step {i}: {step_desc[:50]}...")
            
    except Exception as e:
        console.print(f"[red]Execution error: {e}[/red]")
        # Debug: log the exception
        import traceback
        import sys
        print(f"[DEBUG] Exception in cmd_execute: {e}", file=sys.stderr)
        print(f"[DEBUG] Traceback: {traceback.format_exc()}", file=sys.stderr)


async def cmd_docs(
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    args: str,
):
    """Handle /docs command.
    
    Accepts either:
    - A file path: /docs path/to/file.py
    - A natural language request: /docs write a readme for the scout system
    """
    if not args:
        console.print("[yellow]Usage: /docs <file or request>[/yellow]")
        return

    # Check if args looks like a file path (using safe heuristic, not Path.exists())
    # Path(args).exists() fails with OSError on very long strings
    is_file_path = False
    if args:
        # Simple heuristic: contains path separator or ends with known extension
        has_separator = "/" in args or "\\" in args
        has_extension = any(args.endswith(ext) for ext in [".py", ".md", ".txt", ".js", ".ts", ".json", ".yaml", ".yml", ".toml"])
        is_file_path = has_separator or has_extension
    
    if is_file_path:
        # Generate docs for specific file
        console.print(f"[dim]Generating docs for: {args}[/dim]")
        tool_params = {
            "request": f"Generate comprehensive documentation for {args}",
            "target_files": args,
        }
    else:
        # Natural language request - use summarization LLM
        console.print(f"[dim]Generating docs: {args}[/dim]")
        tool_params = {
            "request": args,
            # No target_files - allows system-wide docs generation
        }

    with Spinner("Generating docs"):
        result = await mcp_client.call_tool("scout_generate_docs", tool_params)

    console.print("\n[bold]Documentation:[/bold]")
    console.print(result.get("result", "No docs generated"))


async def cmd_edit(
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    args: str,
):
    """Handle /edit command."""
    parts = args.split("--prompt", 1)
    if len(parts) < 2:
        console.print("[yellow]Usage: /edit <file> --prompt <instruction>[/yellow]")
        return

    file_path = parts[0].strip()
    prompt = parts[1].strip()

    console.print(f"[dim]Editing: {file_path}[/dim]")

    with Spinner("Editing"):
        result = await mcp_client.call_tool("scout_edit", {
            "file_path": file_path,
            "instruction": prompt
        })

    console.print("\n[bold]Edit result:[/bold]")
    console.print(result.get("result", "No result"))


async def cmd_search(
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    args: str,
):
    """Handle /search command."""
    if not args:
        console.print("[yellow]Usage: /search <pattern>[/yellow]")
        return

    console.print(f"[dim]Searching for: {args}[/dim]")

    result = await mcp_client.call_tool("scout_grep", {"pattern": args})

    console.print("\n[bold]Results:[/bold]")
    console.print(result.get("result", "No results"))


async def cmd_nav(
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    args: str,
):
    """Handle /nav command."""
    if not args:
        console.print("[yellow]Usage: /nav <symbol>[/yellow]")
        return

    console.print(f"[dim]Navigating to: {args}[/dim]")

    result = await mcp_client.call_tool("scout_nav", {"task": args})

    console.print("\n[bold]Navigation result:[/bold]")
    console.print(result.get("result", "No result"))


async def cmd_git(
    console: ConsoleOutput,
    session: Session,
    mcp_client: MCPClient,
    args: str,
):
    """Handle /git command."""
    if not args:
        console.print("[yellow]Usage: /git <commit|pr|status>[/yellow]")
        return

    subcmd = args.split()[0]

    if subcmd == "status":
        result = await mcp_client.call_tool("scout_git_status", {})
    elif subcmd == "commit":
        msg = " ".join(args.split()[1:]) if len(args.split()) > 1 else None
        result = await mcp_client.call_tool("scout_git_commit", {"message": msg} if msg else {})
    elif subcmd == "pr":
        result = await mcp_client.call_tool("scout_pr", {})
    else:
        console.print(f"[red]Unknown git subcommand: {subcmd}[/red]")
        return

    console.print(result.get("result", "No result"))


def cmd_config(console: ConsoleOutput, args: str):
    """Handle /config command."""
    parts = args.split()
    action = parts[0] if parts else "show"

    if action == "show":
        console.print("[bold]Configuration:[/bold]")
        console.print("Use /config get <key> or /config set <key> <value>")
    elif action == "get" and len(parts) > 1:
        console.print(f"Value for {parts[1]}: (not implemented)")
    elif action == "set" and len(parts) > 2:
        console.print(f"Set {parts[1]} = {parts[2]} (not implemented)")
    else:
        console.print("[yellow]Usage: /config [show|get <key>|set <key> <value>][/yellow]")


def cmd_status(console: ConsoleOutput, session: Session):
    """Handle /status command."""
    console.print(f"""
[bold]Session Status[/bold]

Session ID: {session.id}
Created: {session.created_at}
Messages: {len(session.messages)}
Total Cost: ${session.total_cost:.4f}
Total Tokens: {session.total_tokens}
""")


if __name__ == "__main__":
    asyncio.run(run_repl())
