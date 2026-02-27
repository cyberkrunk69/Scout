#!/usr/bin/env python
"""
Toast notification system - non-blocking user feedback.

Features:
- Queue system for multiple notifications
- Auto-dismiss with configurable timeout
- Different notification types (info, success, warning, error)
- Positioned notifications that don't block main UI
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from scout.ui.components.base import BaseComponent, ComponentConfig
from scout.ui.theme.manager import get_theme_manager


class ToastType(Enum):
    """Types of toast notifications."""
    INFO = auto()
    SUCCESS = auto()
    WARNING = auto()
    ERROR = auto()


@dataclass
class Toast:
    """Single toast notification."""
    message: str
    toast_type: ToastType = ToastType.INFO
    timeout: float = 3.0  # seconds
    title: str = ""
    
    
class ToastManager(BaseComponent):
    """
    Manager for toast notifications.
    
    Handles:
    - Queueing multiple toasts
    - Auto-dismiss after timeout
    - Stacked display of active toasts
    """
    
    TYPE_STYLES = {
        ToastType.INFO: ("ℹ", "blue"),
        ToastType.SUCCESS: ("✓", "green"),
        ToastType.WARNING: ("⚠", "yellow"),
        ToastType.ERROR: ("✗", "red"),
    }
    
    def __init__(
        self,
        console: Optional[Console] = None,
        max_visible: int = 3,
        default_timeout: float = 3.0,
        **kwargs
    ):
        config = ComponentConfig(
            refresh_rate=2.0,
        )
        super().__init__(config=config, **kwargs)
        
        self._console = console or Console()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._active_toasts: list[Toast] = []
        self._max_visible = max_visible
        self._default_timeout = default_timeout
        self._theme = get_theme_manager().current_theme
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        get_theme_manager().add_listener(self._on_theme_change)
        
    def _on_theme_change(self, theme):
        self._theme = theme
        self._schedule_refresh()
        
    async def start(self):
        """Start the toast manager."""
        self._running = True
        self._task = asyncio.create_task(self._process_queue())
        
    async def stop(self):
        """Stop the toast manager."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
    def show(
        self,
        message: str,
        toast_type: ToastType = ToastType.INFO,
        timeout: Optional[float] = None,
        title: str = "",
    ):
        """
        Show a toast notification.
        
        Args:
            message: The message to display
            toast_type: Type of notification
            timeout: Auto-dismiss timeout (None for default)
            title: Optional title
        """
        toast = Toast(
            message=message,
            toast_type=toast_type,
            timeout=timeout or self._default_timeout,
            title=title,
        )
        self._queue.put_nowait(toast)
        
    def show_info(self, message: str, title: str = ""):
        """Show info toast."""
        self.show(message, ToastType.INFO, title=title)
        
    def show_success(self, message: str, title: str = ""):
        """Show success toast."""
        self.show(message, ToastType.SUCCESS, title=title)
        
    def show_warning(self, message: str, title: str = ""):
        """Show warning toast."""
        self.show(message, ToastType.WARNING, title=title)
        
    def show_error(self, message: str, title: str = ""):
        """Show error toast."""
        self.show(message, ToastType.ERROR, title=title)
        
    async def _process_queue(self):
        """Process the toast queue."""
        while self._running:
            try:
                toast = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=0.5
                )
                
                # Add to active toasts
                self._active_toasts.append(toast)
                if len(self._active_toasts) > self._max_visible:
                    self._active_toasts.pop(0)
                    
                self._schedule_refresh()
                
                # Wait for timeout then remove
                try:
                    await asyncio.wait_for(
                        self._remove_toast(toast),
                        timeout=toast.timeout
                    )
                except asyncio.TimeoutError:
                    pass
                    
            except asyncio.TimeoutError:
                continue
            except Exception:
                pass
                
    async def _remove_toast(self, toast: Toast):
        """Remove a toast after timeout."""
        await asyncio.sleep(toast.timeout)
        if toast in self._active_toasts:
            self._active_toasts.remove(toast)
            self._schedule_refresh()
            
    def get_active_toasts(self) -> list[Toast]:
        """Get currently active toasts."""
        return self._active_toasts.copy()
        
    def clear(self):
        """Clear all active toasts."""
        self._active_toasts.clear()
        self._schedule_refresh()
        
    def render(self) -> Optional[Panel]:
        """Render the toast stack."""
        if not self._active_toasts:
            return None
            
        lines = []
        for toast in self._active_toasts:
            icon, color = self.TYPE_STYLES[toast.toast_type]
            title = f"{toast.title}: " if toast.title else ""
            lines.append(f"[{color}]{icon}[/] {title}{toast.message}")
            
        content = "\n".join(lines)
        
        return Panel(
            content,
            title=f"[bold {self._theme.colors.primary}]Notifications[/]",
            border_style=self._theme.colors.primary,
        )


# Decorator for async functions to show toasts
def with_toast(manager: ToastManager):
    """Decorator to show toasts for async function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                manager.show_success(f"Completed: {func.__name__}")
                return result
            except Exception as e:
                manager.show_error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator
