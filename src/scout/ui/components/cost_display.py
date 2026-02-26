#!/usr/bin/env python
"""
Cost breakdown display component - shows real-time cost analytics.

Features:
- Per-tier costs (fast/medium/large)
- Per-step breakdown
- Running total with budget warnings
- Projected total for current plan
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scout.ui.components.base import BaseComponent, ComponentConfig, RefreshStrategy
from scout.ui.theme.manager import get_theme_manager


@dataclass
class CostEntry:
    """Single cost entry."""
    tier: str
    amount: float
    tokens: int
    timestamp: float
    step_name: str = ""


@dataclass
class CostState:
    """State for cost tracking."""
    total_cost: float = 0.0
    total_tokens: int = 0
    cost_by_tier: dict[str, float] = field(default_factory=lambda: {"fast": 0.0, "medium": 0.0, "large": 0.0})
    entries: list[CostEntry] = field(default_factory=list)
    budget_limit: Optional[float] = None
    warning_threshold: float = 0.8  # Warn at 80% of budget


class CostBreakdown(BaseComponent):
    """
    Real-time cost breakdown display.
    
    Shows:
    - Cost by tier (fast/medium/large)
    - Running total
    - Budget warnings
    - Per-step breakdown
    """
    
    TIER_COLORS = {
        "fast": "green",
        "medium": "yellow", 
        "large": "red",
    }
    
    def __init__(
        self,
        console: Optional[Console] = None,
        budget_limit: Optional[float] = None,
        **kwargs
    ):
        config = ComponentConfig(
            refresh_rate=2.0,
            refresh_strategy=RefreshStrategy.NORMAL,
        )
        super().__init__(config=config, **kwargs)
        
        self._console = console or Console()
        self._state = CostState(budget_limit=budget_limit)
        self._theme = get_theme_manager().current_theme
        self._start_time = time.time()
        
        get_theme_manager().add_listener(self._on_theme_change)
        
    def _on_theme_change(self, theme):
        self._theme = theme
        self._schedule_refresh()
        
    @property
    def total_cost(self) -> float:
        return self._state.total_cost
    
    @property
    def total_tokens(self) -> int:
        return self._state.total_tokens
    
    def record_call(self, tier: str, cost: float, tokens: int = 0, step_name: str = ""):
        """Record a cost entry."""
        if tier in self._state.cost_by_tier:
            self._state.cost_by_tier[tier] += cost
            
        entry = CostEntry(
            tier=tier,
            amount=cost,
            tokens=tokens,
            timestamp=time.time(),
            step_name=step_name,
        )
        self._state.entries.append(entry)
        self._state.total_cost += cost
        self._state.total_tokens += tokens
        
        # Check budget warning
        if self._state.budget_limit:
            usage = self._state.total_cost / self._state.budget_limit
            if usage >= self._state.warning_threshold:
                self._budget_warning_shown = True
                
        self._schedule_refresh()
        
    def set_budget_limit(self, limit: Optional[float]):
        """Set the budget limit for warnings."""
        self._state.budget_limit = limit
        
    def reset(self):
        """Reset all cost tracking."""
        self._state.total_cost = 0.0
        self._state.total_tokens = 0
        self._state.cost_by_tier = {"fast": 0.0, "medium": 0.0, "large": 0.0}
        self._state.entries = []
        self._start_time = time.time()
        self._schedule_refresh()
        
    def get_cost_summary(self) -> str:
        """Get cost summary as a string."""
        return (
            f"fast=${self._state.cost_by_tier['fast']:.4f} | "
            f"medium=${self._state.cost_by_tier['medium']:.4f} | "
            f"large=${self._state.cost_by_tier['large']:.4f}"
        )
        
    def render(self) -> Panel:
        """Render the cost breakdown panel."""
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Tier", style="cyan")
        table.add_column("Cost", justify="right", style="green")
        
        for tier, cost in self._state.cost_by_tier.items():
            color = self.TIER_COLORS.get(tier, "white")
            table.add_row(tier, f"[{color}]${cost:.4f}[/]")
            
        # Add total row
        table.add_row("---", "---")
        table.add_row(
            "[bold]Total[/bold]",
            f"[bold green]${self._state.total_cost:.4f}[/bold green]"
        )
        
        # Add budget warning if applicable
        subtitle = ""
        if self._state.budget_limit:
            usage = self._state.total_cost / self._state.budget_limit
            if usage >= 1.0:
                subtitle = f"[red bold]OVER BUDGET: ${self._state.total_cost:.4f}/${self._state.budget_limit:.2f}[/]"
            elif usage >= self._state.warning_threshold:
                subtitle = f"[yellow bold]Warning: ${self._state.total_cost:.4f}/${self._state.budget_limit:.2f} ({usage*100:.0f}%)[/]"
                
        return Panel(
            table,
            title=f"[bold {self._theme.colors.primary}]Cost Breakdown[/bold {self._theme.colors.primary}]",
            border_style=self._theme.colors.primary,
            subtitle=subtitle,
            subtitle_align="left",
        )


class CostTracker:
    """
    Lightweight cost tracker for integration with progress bridge.
    
    This is a non-UI class that tracks costs and can be passed
    to the CostBreakdown component for display.
    """
    
    def __init__(self):
        self._state = CostState()
        self._listeners: list[callable] = []
        
    def record(self, tier: str, cost: float, tokens: int = 0, step_name: str = ""):
        """Record a cost entry and notify listeners."""
        self.record_call(tier, cost, tokens, step_name)
        
    def record_call(self, tier: str, cost: float, tokens: int = 0, step_name: str = ""):
        """Record a cost entry."""
        if tier in self._state.cost_by_tier:
            self._state.cost_by_tier[tier] += cost
            
        entry = CostEntry(
            tier=tier,
            amount=cost,
            tokens=tokens,
            timestamp=time.time(),
            step_name=step_name,
        )
        self._state.entries.append(entry)
        self._state.total_cost += cost
        self._state.total_tokens += tokens
        
        # Notify listeners
        for listener in self._listeners:
            try:
                listener(tier, cost, tokens)
            except Exception:
                pass
                
    def add_listener(self, callback: callable):
        """Add a cost update listener."""
        self._listeners.append(callback)
        
    def remove_listener(self, callback: callable):
        """Remove a cost update listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)
            
    @property
    def state(self) -> CostState:
        return self._state
        
    def reset(self):
        """Reset all tracking."""
        self._state = CostState()
