#!/usr/bin/env python
"""
Progress indicators and spinners.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner as RichSpinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.columns import Columns
from rich.panel import Panel

class Spinner:
    """Context manager for showing a spinner while running async operations."""

    def __init__(self, message: str = "Loading...", console: Optional[Console] = None):
        self.message = message
        self.console = console or Console()
        self.live: Optional[Live] = None

    def __enter__(self):
        # FIX: Added "dots" as the first positional argument
        spinner = RichSpinner("dots", text=self.message, style="cyan")
        self.live = Live(spinner, console=self.console, transient=True)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.live:
            self.live.stop()
        return False

    def update(self, message: str):
        """Update the spinner message."""
        if self.live:
            # FIX: Added "dots" here as well
            spinner = RichSpinner("dots", text=message, style="cyan")
            self.live.update(spinner)

class ProgressBar:
    """Progress bar for long-running operations."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.progress: Optional[Progress] = None

    def __enter__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
            transient=True,
        )
        self.progress.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.__exit__(exc_type, exc_val, exc_tb)
        return False

    def add_task(self, description: str, total: Optional[float] = None):
        """Add a task to the progress bar."""
        if self.progress:
            return self.progress.add_task(description, total=total)
        return None

    def update(self, task_id, advance: float = 1, **kwargs):
        """Update a task."""
        if self.progress:
            self.progress.update(task_id, advance=advance, **kwargs)

@contextmanager
def spinner(message: str = "Loading..."):
    """Simple spinner context manager."""
    # We use the class-based Spinner here to ensure consistency
    with Spinner(message) as s:
        yield s


class ParallelProcessDisplay:
    """
    Display for parallel processes with live-updating whimsy status.
    
    Shows multiple processes running in parallel, each with:
    - What they're doing (user-facing language)
    - Why they're doing it
    - When/current stage
    - How complete (progress bar)
    - Meta humor from 8B model
    """

    STAGE_ICONS = {
        "pending": ("⏳", "yellow"),
        "running": ("⚡", "cyan"),
        "completed": ("✅", "green"),
        "failed": ("❌", "red"),
    }

    def __init__(self, max_display: int = 5):
        self.max_display = max_display
        self._processes: dict[str, dict] = {}

    def add_process(self, process_id: str, name: str) -> None:
        """Add a new process to display."""
        self._processes[process_id] = {
            "name": name,
            "stage": "pending",
            "whimsy": None,
            "progress": 0.0,
        }

    def update_process(
        self,
        process_id: str,
        stage: str = None,
        whimsy: str = None,
        progress: float = None,
    ) -> None:
        """Update process status."""
        if process_id not in self._processes:
            return
        if stage is not None:
            self._processes[process_id]["stage"] = stage
        if whimsy is not None:
            self._processes[process_id]["whimsy"] = whimsy
        if progress is not None:
            self._processes[process_id]["progress"] = progress

    def remove_process(self, process_id: str) -> None:
        """Remove a process from display."""
        self._processes.pop(process_id, None)

    def get_display_data(self) -> list[dict]:
        """Get processes for display (limited to max_display)."""
        processes = list(self._processes.values())[: self.max_display]
        result = []
        for p in processes:
            icon, color = self.STAGE_ICONS.get(p["stage"], ("❓", "white"))
            whimsy = p.get("whimsy") or p["name"]
            progress = p.get("progress", 0.0)
            bar_width = int(progress * 10)
            bar = "█" * bar_width + "░" * (10 - bar_width)
            result.append(
                {
                    "icon": icon,
                    "color": color,
                    "name": p["name"],
                    "whimsy": whimsy,
                    "progress": progress,
                    "bar": bar,
                    "stage": p["stage"],
                }
            )
        return result

    def clear(self) -> None:
        """Clear all processes."""
        self._processes.clear()


class ReasoningLiveDisplay:
    """
    Live display showing Scout's reasoning and current action.
    
    Left pane: What Scout is thinking (Reasoning)
    Right pane: Progress bar/spinner for current Action (Tool call)
    Far Right pane: Parallel Processes with live whimsy (Brain Activity)
    Footer: Real-time stats (USD cost, steps completed, current status)
    
    Supports Ghost Panels - panels with 0 width when empty via WhimsyManager.
    """

    # Log type to (emoji, color) mapping - inspired by whimsy.py aesthetics
    LOG_ICONS = {
        "info": ("ℹ️", "blue"),
        "warning": ("⚠️", "yellow"),
        "discovery": ("🔍", "cyan"),
    }

    def __init__(self, console: Optional[Console] = None, stderr: bool = True):
        # Use stderr=True by default to keep stdout clean for final Markdown
        self.console = console or Console(stderr=stderr)
        self.live: Optional[Live] = None
        self._reasoning: list[str] = []
        self._logs: list[tuple[str, str, str]] = []  # (message, emoji, color)
        self._current_action: str = ""
        self._action_progress: float = 0.0
        self._cancelled = False
        # Heartbeat animation state
        self._heartbeat_active: bool = False
        self._heartbeat_text: str = ""
        self._heartbeat_frame: int = 0
        # Stats tracking
        self._total_cost_usd: float = 0.0
        self._steps_completed: int = 0
        self._current_status: str = "Initializing..."
        # Parallel processes with whimsy
        self._parallel_display: ParallelProcessDisplay = ParallelProcessDisplay(max_display=4)
        
        # Ghost Panel integration
        try:
            from scout.ui.whimsy_manager import get_whimsy_manager
            self._whimsy = get_whimsy_manager()
        except ImportError:
            self._whimsy = None
    
    def _get_panel_width(self, panel_id: str, content: str, default_width: int = 30) -> int:
        """
        Get panel width - returns 0 for Ghost Panels (hidden) when content is empty.
        Uses WhimsyManager to determine visibility.
        """
        if not content and self._whimsy:
            # Check if this panel should be hidden when empty
            if not self._whimsy.is_panel_visible(f"ghost_{panel_id}"):
                return 0  # Ghost panel - hidden
        return default_width

    def set_stats(self, cost_usd: float = None, steps: int = None, status: str = None):
        """Update real-time stats."""
        if cost_usd is not None:
            self._total_cost_usd = cost_usd
        if steps is not None:
            self._steps_completed = steps
        if status is not None:
            self._current_status = status
        self._update_display()

    def add_cost(self, cost_delta: float):
        """Add to the total cost."""
        self._total_cost_usd += cost_delta
        self._update_display()

    def increment_steps(self):
        """Increment the steps completed counter."""
        self._steps_completed += 1
        self._update_display()

    def get_stats(self) -> dict:
        """Get current stats as a dict."""
        return {
            "cost_usd": self._total_cost_usd,
            "steps": self._steps_completed,
            "status": self._current_status,
        }

    # --- Parallel Process Management ---

    def add_parallel_process(self, process_id: str, name: str) -> None:
        """Add a parallel process to track."""
        self._parallel_display.add_process(process_id, name)
        self._update_display()

    def update_parallel_process(
        self,
        process_id: str,
        stage: str = None,
        whimsy: str = None,
        progress: float = None,
    ) -> None:
        """Update a parallel process status."""
        self._parallel_display.update_process(process_id, stage, whimsy, progress)
        self._update_display()

    def remove_parallel_process(self, process_id: str) -> None:
        """Remove a parallel process."""
        self._parallel_display.remove_process(process_id)
        self._update_display()

    def get_parallel_processes(self) -> list[dict]:
        """Get parallel processes for display."""
        return self._parallel_display.get_display_data()

    def feed_reasoning_trace(self, trace_event: dict):
        """
        Feed a ReasoningTrace event to the display.
        
        Args:
            trace_event: A dict containing reasoning trace information.
                Expected keys:
                - 'thought': The reasoning thought (str)
                - 'action': Current action being performed (str, optional)
                - 'progress': Progress value 0.0-1.0 (float, optional)
                - 'cost_delta': Cost increment (float, optional)
                - 'status': Status message (str, optional)
        """
        # Extract thought and add to reasoning
        if "thought" in trace_event:
            self.add_reasoning(trace_event["thought"])
        
        # Update action if provided
        if "action" in trace_event:
            progress = trace_event.get("progress", 0.0)
            self.set_action(trace_event["action"], progress)
        
        # Update cost if provided
        if "cost_delta" in trace_event:
            self.add_cost(trace_event["cost_delta"])
        
        # Update status if provided
        if "status" in trace_event:
            self.set_stats(status=trace_event["status"])
        
        # Increment steps if this is a step completion event
        if trace_event.get("is_step", False):
            self.increment_steps()

    def add_reasoning(self, thought: str):
        """Add a reasoning step to the display."""
        self._reasoning.append(thought)
        # Keep only last 10 reasoning steps
        if len(self._reasoning) > 10:
            self._reasoning = self._reasoning[-10:]
        self._update_display()

    def add_log(self, message: str, type: str = "info"):
        """
        Add a log message with icon based on type.
        
        Args:
            message: The log message to display
            type: Log type - 'info' (blue ℹ️), 'warning' (yellow ⚠️), 'discovery' (cyan 🔍)
        """
        emoji, color = self.LOG_ICONS.get(type, self.LOG_ICONS["info"])
        self._logs.append((message, emoji, color))
        # Keep only last 15 logs
        if len(self._logs) > 15:
            self._logs = self._logs[-15:]
        self._update_display()

    def clear_logs(self):
        """Clear log history."""
        self._logs = []
        self._update_display()

    def clear(self):
        """Clear all display state (reasoning, logs, action, parallel processes)."""
        self._reasoning = []
        self._logs = []
        self._current_action = ""
        self._action_progress = 0.0
        self._heartbeat_active = False
        self._heartbeat_text = ""
        self._parallel_display.clear()
        self._update_display()

    def start_heartbeat(self, text: str = "Reasoning about Tool Chain..."):
        """Start the heartbeat animation in the action pane."""
        self._heartbeat_active = True
        self._heartbeat_text = text
        self._heartbeat_frame = 0
        self._update_display()

    def stop_heartbeat(self):
        """Stop the heartbeat animation."""
        self._heartbeat_active = False
        self._heartbeat_text = ""
        self._update_display()

    def _get_heartbeat_frame(self) -> str:
        """Get the current heartbeat animation frame."""
        # Blinking heart pattern: 💓 💖 💗
        hearts = ["💓", "💖", "💗", "💗"]
        heart = hearts[self._heartbeat_frame % len(hearts)]
        self._heartbeat_frame += 1
        return f"{heart} {self._heartbeat_text}"

    def set_action(self, action: str, progress: float = 0.0):
        """Set the current action and optional progress (0.0-1.0)."""
        self._current_action = action
        self._action_progress = progress
        self._update_display()

    def update_progress(self, progress: float):
        """Update action progress (0.0-1.0)."""
        self._action_progress = progress
        self._update_display()

    def cancel(self):
        """Signal cancellation."""
        self._cancelled = True
        self.set_action("[yellow]Cancelled by user[/yellow]", 0.0)

    def is_cancelled(self) -> bool:
        """Check if the operation was cancelled."""
        return self._cancelled

    def clear_reasoning(self):
        """Clear reasoning history."""
        self._reasoning = []
        self._update_display()

    def _build_layout(self) -> "Panel":
        """Build the dual-pane layout with stats footer."""
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
        from rich.text import Text

        # Left pane: Reasoning + Logs combined
        left_parts = []
        
        # Add reasoning section
        if self._reasoning:
            reasoning_text = "\n".join(f"• {line}" for line in self._reasoning)
            left_parts.append("[bold cyan]Thoughts:[/bold cyan]\n" + reasoning_text)
        
        # Add logs section
        if self._logs:
            log_lines = []
            for msg, emoji, color in self._logs:
                log_lines.append(f"[{color}]{emoji}[/{color}] {msg}")
            logs_text = "\n".join(log_lines)
            if left_parts:
                left_parts.append("")
            left_parts.append("[bold magenta]Live Log:[/bold magenta]\n" + logs_text)
        
        left_content = "\n".join(left_parts) if left_parts else "[dim]Waiting for thoughts...[/dim]"

        reasoning_panel = Panel(
            left_content,
            title="[bold cyan]Reasoning[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )

        # Right pane: Action with progress or heartbeat
        if self._heartbeat_active:
            # Show heartbeat animation with brain text
            heartbeat_text = self._get_heartbeat_frame()
            # Use a custom spinner that shows the heartbeat
            heartbeat_display = f"[bold magenta]{heartbeat_text}[/bold magenta]"
            spinner = RichSpinner("dots", text=heartbeat_display, style="magenta")
            action_panel = Panel(
                spinner,
                title="[bold magenta]🧠 Brain Activity[/bold magenta]",
                border_style="magenta",
                padding=(0, 1),
            )
        elif self._action_progress > 0:
            # Create a progress bar for the action
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=20),
                TaskProgressColumn(),
                console=self.console,
                transient=True,
            )
            progress.start()
            task_id = progress.add_task(self._current_action, total=100)
            progress.update(task_id, completed=int(self._action_progress * 100))
            
            action_panel = Panel(
                progress,
                title="[bold green]Action/Spinners[/bold green]",
                border_style="green",
                padding=(0, 1),
            )
            progress.stop()
        else:
            # Just show spinner and action text
            spinner = RichSpinner("dots", text=self._current_action or "Idle", style="green")
            action_panel = Panel(
                spinner,
                title="[bold green]Action/Spinners[/bold green]",
                border_style="green",
                padding=(0, 1),
            )

        # Far Right pane: Parallel Processes with whimsy
        parallel_processes = self.get_parallel_processes()
        if parallel_processes:
            parallel_lines = []
            for p in parallel_processes:
                icon = p["icon"]
                color = p["color"]
                name = p["name"][:15]
                whimsy = p["whimsy"][:30] if p["whimsy"] else ""
                bar = p["bar"]
                parallel_lines.append(
                    f"[{color}]{icon}[/{color}] [bold]{name}[/bold]\n"
                    f"  {whimsy}\n"
                    f"  [{color}]{bar}[/{color}]"
                )
            parallel_content = "\n\n".join(parallel_lines)
        else:
            parallel_content = "[dim]No parallel processes...[/dim]"

        parallel_panel = Panel(
            parallel_content,
            title="[bold yellow]⚡ Parallel Processes[/bold yellow]",
            border_style="yellow",
            padding=(0, 1),
            width=45,
        )

        # Combine panels: Reasoning | Action | Parallel Processes
        layout = Columns([reasoning_panel, action_panel, parallel_panel], padding=1)
        
        # Build footer with stats
        cost_str = f"${self._total_cost_usd:.4f}" if self._total_cost_usd > 0 else "$0.0000"
        steps_str = str(self._steps_completed)
        
        footer_text = f"[bold]Stats:[/bold] [cyan]Cost:[/cyan] {cost_str}  [green]Steps:[/green] {steps_str}  [yellow]Status:[/yellow] {self._current_status}"
        
        # Return the complete layout with footer
        return Panel(
            layout,
            title="[bold blue]Mission Control[/bold blue]",
            border_style="blue",
            padding=(0, 1),
            subtitle=footer_text,
            subtitle_align="left",
        )

    def _update_display(self):
        """Update the live display."""
        if self.live:
            self.live.update(self._build_layout())

    def __enter__(self):
        """Start the live display."""
        self.live = Live(self._build_layout(), console=self.console, refresh_per_second=4)
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the live display."""
        if self.live:
            self.live.stop()
        return False