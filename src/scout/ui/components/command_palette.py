#!/usr/bin/env python
"""
Command palette component - Ctrl+P fuzzy command search.

Features:
- Fuzzy search for commands and recent actions
- Keyboard navigation
- Recent commands history
- Integration with ScoutCompleter
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import has_focus
from prompt_toolkit.keys import Keys
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from scout.ui.components.base import BaseComponent, ComponentConfig
from scout.ui.theme.manager import get_theme_manager


@dataclass
class Command:
    """A command that can be executed."""
    name: str
    description: str
    handler: Optional[Callable] = None
    shortcut: str = ""
    category: str = ""


@dataclass
class CommandMatch:
    """A command matched against a query."""
    command: Command
    score: float = 0.0
    matched_indices: list[int] = field(default_factory=list)


class CommandPalette(BaseComponent):
    """
    Fuzzy search command palette (Ctrl+P).
    
    Features:
    - Fuzzy matching algorithm
    - Keyboard navigation (up/down arrows)
    - Execute on Enter
    - Recent commands tracking
    """
    
    def __init__(
        self,
        console: Optional[Console] = None,
        max_results: int = 10,
        **kwargs
    ):
        config = ComponentConfig(
            refresh_rate=0.0,  # Static unless searching
            visible=False,  # Hidden by default
        )
        super().__init__(config=config, **kwargs)
        
        self._console = console or Console()
        self._commands: list[Command] = []
        self._recent_commands: list[str] = []
        self._max_results = max_results
        self._query = ""
        self._selected_index = 0
        self._results: list[CommandMatch] = []
        self._theme = get_theme_manager().current_theme
        self._visible = False
        self._key_bindings = self._create_key_bindings()
        
        get_theme_manager().add_listener(self._on_theme_change)
        
    def _on_theme_change(self, theme):
        self._theme = theme
        self._schedule_refresh()
        
    def _create_key_bindings(self) -> KeyBindings:
        """Create key bindings for the palette."""
        kb = KeyBindings()
        
        @kb.add('c-p', filter=has_focus('palette'))
        def toggle_up(event):
            self._selected_index = max(0, self._selected_index - 1)
            
        @kb.add('c-n', filter=has_focus('palette'))
        def toggle_down(event):
            self._selected_index = min(len(self._results) - 1, self._selected_index + 1)
            
        @kb.add('enter', filter=has_focus('palette'))
        def execute_selected(event):
            if self._results:
                self.execute(self._results[self._selected_index].command)
                
        @kb.add('escape', filter=has_focus('palette'))
        def close(event):
            self.hide()
            
        return kb
        
    def register_command(
        self,
        name: str,
        description: str,
        handler: Optional[Callable] = None,
        shortcut: str = "",
        category: str = "",
    ):
        """Register a command in the palette."""
        cmd = Command(
            name=name,
            description=description,
            handler=handler,
            shortcut=shortcut,
            category=category,
        )
        self._commands.append(cmd)
        
    def register_commands(self, commands: list[Command]):
        """Register multiple commands."""
        self._commands.extend(commands)
        
    def unregister_command(self, name: str):
        """Unregister a command by name."""
        self._commands = [c for c in self._commands if c.name != name]
        
    def show(self):
        """Show the command palette."""
        self._visible = True
        self._config.visible = True
        self._query = ""
        self._selected_index = 0
        self._search()
        
    def hide(self):
        """Hide the command palette."""
        self._visible = False
        self._config.visible = False
        self._query = ""
        
    def toggle(self):
        """Toggle visibility."""
        if self._visible:
            self.hide()
        else:
            self.show()
            
    @property
    def is_visible(self) -> bool:
        return self._visible
        
    def search(self, query: str):
        """Update search query."""
        self._query = query
        self._selected_index = 0
        self._search()
        
    def _search(self):
        """Perform fuzzy search."""
        if not self._query:
            # Show recent commands if no query
            self._results = []
            for name in self._recent_commands[:self._max_results]:
                for cmd in self._commands:
                    if cmd.name == name:
                        self._results.append(CommandMatch(command=cmd, score=1.0))
                        break
        else:
            self._results = self._fuzzy_search(self._query)
            
    def _fuzzy_search(self, query: str) -> list[CommandMatch]:
        """Fuzzy search implementation."""
        results = []
        query_lower = query.lower()
        
        for cmd in self._commands:
            match = self._fuzzy_match(query_lower, cmd.name)
            if match is not None:
                results.append(CommandMatch(command=cmd, score=match[0], matched_indices=match[1]))
                
        # Sort by score (higher is better)
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:self._max_results]
        
    def _fuzzy_match(self, query: str, target: str) -> Optional[tuple[float, list[int]]]:
        """Fuzzy match single target."""
        target_lower = target.lower()
        query_idx = 0
        matched_indices = []
        
        for i, char in enumerate(target_lower):
            if query_idx < len(query) and char == query[query_idx]:
                matched_indices.append(i)
                query_idx += 1
                
        if query_idx != len(query):
            return None  # Not all query chars matched
            
        # Calculate score
        score = len(matched_indices) / len(target)
        
        # Bonus for consecutive matches
        consecutive = 0
        prev_idx = -2
        for idx in matched_indices:
            if idx == prev_idx + 1:
                consecutive += 1
            prev_idx = idx
        score += consecutive * 0.1
        
        return (score, matched_indices)
        
    def execute(self, command: Command):
        """Execute a command."""
        # Add to recent
        if command.name not in self._recent_commands:
            self._recent_commands.insert(0, command.name)
            if len(self._recent_commands) > 20:
                self._recent_commands.pop()
                
        # Execute handler
        if command.handler:
            command.handler()
            
        self.hide()
        
    def move_selection_up(self):
        """Move selection up."""
        self._selected_index = max(0, self._selected_index - 1)
        
    def move_selection_down(self):
        """Move selection down."""
        self._selected_index = min(len(self._results) - 1, self._selected_index + 1)
        
    def render(self) -> Optional[Panel]:
        """Render the command palette."""
        if not self._visible:
            return None
            
        # Build results table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("index", width=3)
        table.add_column("name", width=20)
        table.add_column("description")
        
        for i, match in enumerate(self._results):
            cmd = match.command
            is_selected = i == self._selected_index
            style = "reverse" if is_selected else ""
            
            # Highlight matched characters
            name = cmd.name
            if match.matched_indices:
                # Build highlighted name
                parts = []
                last_idx = 0
                for idx in match.matched_indices:
                    if idx > last_idx:
                        parts.append(name[last_idx:idx])
                    parts.append(f"[bold]{name[idx]}[/bold]")
                    last_idx = idx + 1
                if last_idx < len(name):
                    parts.append(name[last_idx:])
                name = "".join(parts)
                
            table.add_row(
                f"[dim]{i+1}[/dim]",
                name,
                f"[dim]{cmd.description}[/dim]",
                style=style,
            )
            
        # Show hint
        hint = "\n[dim]↑↓ Navigate | Enter Execute | Esc Close[/dim]"
        
        return Panel(
            table,
            title=f"[bold]Command Palette[/bold] [dim]{self._query}[/dim]",
            border_style=self._theme.colors.primary,
        )


class CommandPaletteCompleter(Completer):
    """Completer that wraps CommandPalette."""
    
    def __init__(self, palette: CommandPalette):
        self._palette = palette
        
    def get_completions(self, document, complete_event):
        query = document.text_before_cursor
        
        if not query.startswith("/"):
            return
            
        # Get matching commands
        self._palette.search(query[1:])
        results = self._palette._results
        
        for match in results:
            cmd = match.command
            yield Completion(
                cmd.name,
                start_position=-len(query) + 1,
                display_meta=cmd.description,
            )
