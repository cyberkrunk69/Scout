#!/usr/bin/env python
"""
Reasoning display component - shows Scout's thinking process.

Refactored from progress.py with:
- Async-first design
- Theme integration
- Non-blocking updates
- Panel visibility API for ghost panels
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn
from rich.spinner import Spinner as RichSpinner
from rich.text import Text

from scout.ui.components.base import BaseComponent, ComponentConfig, RefreshStrategy
from scout.ui.theme.manager import get_theme_manager


@dataclass
class ReasoningState:
    """State for reasoning display."""
    reasoning: list[str] = field(default_factory=list)
    logs: list[tuple[str, str, str]] = field(default_factory=list)  # (message, emoji, color)
    current_action: str = ""
    action_progress: float = 0.0
    heartbeat_active: bool = False
    heartbeat_text: str = ""
    heartbeat_frame: int = 0
    total_cost_usd: float = 0.0
    steps_completed: int = 0
    current_status: str = "Initializing..."
    cancelled: bool = False


class ReasoningDisplay(BaseComponent):
    """
    Live display showing Scout's reasoning and current action.
    
    Layout:
    - Left pane: What Scout is thinking (Reasoning)
    - Right pane: Progress bar/spinner for current Action (Tool call)
    - Far Right pane: Parallel Processes with live whimsy (Brain Activity)
    - Footer: Real-time stats (USD cost, steps completed, current status)
    
    Supports Ghost Panels - panels with 0 width when empty.
    """
    
    LOG_ICONS = {
        "info": ("ℹ", "blue"),
        "warning": ("⚠", "yellow"),
        "discovery": ("🔍", "cyan"),
    }
    
    HEARTBEAT_FRAMES = ["💓", "💖", "💗", "💗"]
    
    def __init__(self, console: Optional[Console] = None, stderr: bool = True, **kwargs):
        config = ComponentConfig(
            refresh_rate=4.0,
            refresh_strategy=RefreshStrategy.NORMAL,
        )
        super().__init__(config=config, **kwargs)
        
        self._console = console or Console(stderr=stderr)
        self._state = ReasoningState()
        self._live: Optional[Live] = None
        self._update_queue: asyncio.Queue = asyncio.Queue()
        self._update_task: Optional[asyncio.Task] = None
        self._parallel_display = ParallelProcessDisplay()
        self._theme = get_theme_manager().current_theme
        
        # Register for theme changes
        get_theme_manager().add_listener(self._on_theme_change)
        
    def _on_theme_change(self, theme):
        """Handle theme change."""
        self._theme = theme
        self._schedule_refresh()
        
    @property
    def state(self) -> ReasoningState:
        return self._state
    
    # --- State mutation methods ---
    
    def set_stats(self, cost_usd: Optional[float] = None, steps: Optional[int] = None, status: Optional[str] = None):
        """Update real-time stats."""
        if cost_usd is not None:
            self._state.total_cost_usd = cost_usd
        if steps is not None:
            self._state.steps_completed = steps
        if status is not None:
            self._state.current_status = status
        self._schedule_refresh()
        
    def add_cost(self, cost_delta: float):
        """Add to the total cost."""
        self._state.total_cost_usd += cost_delta
        self._schedule_refresh()
        
    def increment_steps(self):
        """Increment the steps completed counter."""
        self._state.steps_completed += 1
        self._schedule_refresh()
        
    def add_reasoning(self, thought: str):
        """Add a reasoning step to the display."""
        self._state.reasoning.append(thought)
        # Keep only last 10 reasoning steps
        if len(self._state.reasoning) > 10:
            self._state.reasoning = self._state.reasoning[-10:]
        self._schedule_refresh()
        
    def add_log(self, message: str, log_type: str = "info"):
        """Add a log message with icon based on type."""
        emoji, color = self.LOG_ICONS.get(log_type, self.LOG_ICONS["info"])
        self._state.logs.append((message, emoji, color))
        # Keep only last 15 logs
        if len(self._state.logs) > 15:
            self._state.logs = self._state.logs[-15:]
        self._schedule_refresh()
        
    def clear_logs(self):
        """Clear log history."""
        self._state.logs = []
        self._schedule_refresh()
        
    def clear(self):
        """Clear all display state."""
        self._state.reasoning = []
        self._state.logs = []
        self._state.current_action = ""
        self._state.action_progress = 0.0
        self._state.heartbeat_active = False
        self._state.heartbeat_text = ""
        self._parallel_display.clear()
        self._schedule_refresh()
        
    def start_heartbeat(self, text: str = "Reasoning about Tool Chain..."):
        """Start the heartbeat animation in the action pane."""
        self._state.heartbeat_active = True
        self._state.heartbeat_text = text
        self._state.heartbeat_frame = 0
        self._schedule_refresh()
        
    def stop_heartbeat(self):
        """Stop the heartbeat animation."""
        self._state.heartbeat_active = False
        self._state.heartbeat_text = ""
        self._schedule_refresh()
        
    def set_action(self, action: str, progress: float = 0.0):
        """Set the current action and optional progress (0.0-1.0)."""
        self._state.current_action = action
        self._state.action_progress = progress
        self._schedule_refresh()
        
    def update_progress(self, progress: float):
        """Update action progress (0.0-1.0)."""
        self._state.action_progress = progress
        self._schedule_refresh()
        
    def cancel(self):
        """Signal cancellation."""
        self._state.cancelled = True
        self.set_action("[yellow]Cancelled by user[/yellow]", 0.0)
        
    def is_cancelled(self) -> bool:
        """Check if the operation was cancelled."""
        return self._state.cancelled
        
    # --- Parallel process management ---
    
    def add_parallel_process(self, process_id: str, name: str):
        """Add a parallel process to track."""
        self._parallel_display.add_process(process_id, name)
        self._schedule_refresh()
        
    def update_parallel_process(
        self,
        process_id: str,
        stage: Optional[str] = None,
        whimsy: Optional[str] = None,
        progress: Optional[float] = None,
    ):
        """Update a parallel process status."""
        self._parallel_display.update_process(process_id, stage, whimsy, progress)
        self._schedule_refresh()
        
    def remove_parallel_process(self, process_id: str):
        """Remove a parallel process."""
        self._parallel_display.remove_process(process_id)
        self._schedule_refresh()
        
    # --- Rendering ---
    
    def render(self) -> Panel:
        """Build the dual-pane layout with stats footer."""
        # Left pane: Reasoning + Logs combined
        left_parts = []
        
        # Add reasoning section
        if self._state.reasoning:
            reasoning_text = "\n".join(f"• {line}" for line in self._state.reasoning)
            left_parts.append(f"[bold {self._theme.colors.reasoning}]Thoughts:[/bold {self._theme.colors.reasoning}]\n{reasoning_text}")
        
        # Add logs section
        if self._state.logs:
            log_lines = []
            for msg, emoji, color in self._state.logs:
                log_lines.append(f"[{color}]{emoji}[/{color}] {msg}")
            logs_text = "\n".join(log_lines)
            if left_parts:
                left_parts.append("")
            left_parts.append(f"[bold {self._theme.colors.logs}]Live Log:[/bold {self._theme.colors.logs}]\n{logs_text}")
        
        left_content = "\n".join(left_parts) if left_parts else "[dim]Waiting for thoughts...[/dim]"

        reasoning_panel = Panel(
            left_content,
            title=f"[bold {self._theme.colors.reasoning}]Reasoning[/bold {self._theme.colors.reasoning}]",
            border_style=self._theme.colors.reasoning,
            padding=self._theme.spacing.panel_padding,
        )

        # Right pane: Action with progress or heartbeat
        action_panel = self._render_action_panel()
        
        # Far Right pane: Parallel Processes with whimsy
        parallel_panel = self._render_parallel_panel()

        # Combine panels
        layout = Columns([reasoning_panel, action_panel, parallel_panel], padding=1)
        
        # Build footer with stats
        cost_str = f"${self._state.total_cost_usd:.4f}" if self._state.total_cost_usd > 0 else "$0.0000"
        steps_str = str(self._state.steps_completed)
        
        footer_text = f"[bold]Stats:[/bold] [{self._theme.colors.action}]Cost:[/] {cost_str}  [{self._theme.colors.success}]Steps:[/] {steps_str}  [{self._theme.colors.warning}]Status:[/] {self._state.current_status}"
        
        return Panel(
            layout,
            title=f"[bold {self._theme.colors.primary}]Mission Control[/bold {self._theme.colors.primary}]",
            border_style=self._theme.colors.primary,
            padding=self._theme.spacing.panel_padding,
            subtitle=footer_text,
            subtitle_align="left",
        )
    
    def _render_action_panel(self) -> Panel:
        """Render the action/spinner panel."""
        if self._state.heartbeat_active:
            # Show heartbeat animation
            heartbeat_text = self._get_heartbeat_frame()
            heartbeat_display = f"[bold {self._theme.colors.brain_activity}]{heartbeat_text}[/]"
            spinner = RichSpinner("dots", text=heartbeat_display, style=self._theme.colors.brain_activity)
            return Panel(
                spinner,
                title=f"[bold {self._theme.colors.brain_activity}]Brain Activity[/bold {self._theme.colors.brain_activity}]",
                border_style=self._theme.colors.brain_activity,
                padding=self._theme.spacing.panel_padding,
            )
        elif self._state.action_progress > 0:
            # Show progress bar
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=20),
                TaskProgressColumn(),
                console=self._console,
                transient=True,
            )
            progress.start()
            task_id = progress.add_task(self._state.current_action, total=100)
            progress.update(task_id, completed=int(self._state.action_progress * 100))
            
            panel = Panel(
                progress,
                title=f"[bold {self._theme.colors.action}]Action[/bold {self._theme.colors.action}]",
                border_style=self._theme.colors.action,
                padding=self._theme.spacing.panel_padding,
            )
            progress.stop()
            return panel
        else:
            # Just show spinner
            spinner = RichSpinner("dots", text=self._state.current_action or "Idle", style=self._theme.colors.action)
            return Panel(
                spinner,
                title=f"[bold {self._theme.colors.action}]Action[/bold {self._theme.colors.action}]",
                border_style=self._theme.colors.action,
                padding=self._theme.spacing.panel_padding,
            )
    
    def _render_parallel_panel(self) -> Panel:
        """Render the parallel processes panel."""
        parallel_processes = self._parallel_display.get_display_data()
        if parallel_processes:
            parallel_lines = []
            for p in parallel_processes:
                icon = p["icon"]
                color = p["color"]
                name = p["name"][:15]
                whimsy = p["whimsy"][:30] if p["whimsy"] else ""
                bar = p["bar"]
                parallel_lines.append(
                    f"[{color}]{icon}[/] [bold]{name}[/]\n  {whimsy}\n  [{color}]{bar}[/{color}]"
                )
            parallel_content = "\n\n".join(parallel_lines)
        else:
            parallel_content = "[dim]No parallel processes...[/dim]"

        return Panel(
            parallel_content,
            title=f"[bold {self._theme.colors.parallel}]Parallel Processes[/bold {self._theme.colors.parallel}]",
            border_style=self._theme.colors.parallel,
            padding=self._theme.spacing.panel_padding,
            width=45,
        )
    
    def _get_heartbeat_frame(self) -> str:
        """Get the current heartbeat animation frame."""
        heart = self.HEARTBEAT_FRAMES[self._state.heartbeat_frame % len(self.HEARTBEAT_FRAMES)]
        self._state.heartbeat_frame += 1
        return f"{heart} {self._state.heartbeat_text}"
    
    # --- Lifecycle ---
    
    def mount(self, console: Optional[Console] = None, parent=None):
        """Start the live display."""
        if console:
            self._console = console
        super().mount(self._console, parent)
        
    def unmount(self):
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None
        get_theme_manager().remove_listener(self._on_theme_change)
        super().unmount()
        
    def __enter__(self):
        """Start the live display."""
        self._live = Live(self.render(), console=self._console, refresh_per_second=int(self._theme.animation.refresh_rate))
        self._live.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the live display."""
        if self._live:
            self._live.stop()
        return False


class ParallelProcessDisplay:
    """
    Display for parallel processes with live-updating whimsy status.
    """
    
    STAGE_ICONS = {
        "pending": ("⏳", "yellow"),
        "running": ("⚡", "cyan"),
        "completed": ("✅", "green"),
        "failed": ("❌", "red"),
    }
    
    def __init__(self, max_display: int = 4):
        self.max_display = max_display
        self._processes: dict[str, dict] = {}
        
    def add_process(self, process_id: str, name: str):
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
        stage: Optional[str] = None,
        whimsy: Optional[str] = None,
        progress: Optional[float] = None,
    ):
        """Update process status."""
        if process_id not in self._processes:
            return
        if stage is not None:
            self._processes[process_id]["stage"] = stage
        if whimsy is not None:
            self._processes[process_id]["whimsy"] = whimsy
        if progress is not None:
            self._processes[process_id]["progress"] = progress
            
    def remove_process(self, process_id: str):
        """Remove a process from display."""
        self._processes.pop(process_id, None)
        
    def get_display_data(self) -> list[dict]:
        """Get processes for display."""
        processes = list(self._processes.values())[:self.max_display]
        result = []
        for p in processes:
            icon, color = self.STAGE_ICONS.get(p["stage"], ("❓", "white"))
            whimsy = p.get("whimsy") or p["name"]
            progress = p.get("progress", 0.0)
            bar_width = int(progress * 10)
            bar = "█" * bar_width + "░" * (10 - bar_width)
            result.append({
                "icon": icon,
                "color": color,
                "name": p["name"],
                "whimsy": whimsy,
                "progress": progress,
                "bar": bar,
                "stage": p["stage"],
            })
        return result
        
    def clear(self):
        """Clear all processes."""
        self._processes.clear()
