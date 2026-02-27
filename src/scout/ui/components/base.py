#!/usr/bin/env python
"""
Base UI components - foundation for all Scout UI elements.

Provides abstract base classes for:
- BaseComponent: All UI components inherit from this
- Layout: Container for multiple components
- Panel: Styled container with borders and title
- ProgressIndicator: Base for progress displays
"""
from __future__ import annotations

import signal
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from rich.console import Console, ConsoleOptions, RenderResult
from rich.layout import Layout as RichLayout
from rich.panel import Panel as RichPanel
from rich.style import Style


class ComponentState(Enum):
    """Lifecycle states for UI components."""
    CREATED = auto()
    MOUNTED = auto()
    VISIBLE = auto()
    HIDDEN = auto()
    DESTROYED = auto()


class RefreshStrategy(Enum):
    """How often a component should refresh."""
    REALTIME = auto()      # Every frame (~60fps)
    FAST = auto()          # 10 times per second
    NORMAL = auto()        # 4 times per second (default)
    SLOW = auto()          # Once per second
    STATIC = auto()        # Only on state change


@dataclass
class ComponentConfig:
    """Configuration for a UI component."""
    refresh_rate: float = 4.0  # times per second
    refresh_strategy: RefreshStrategy = RefreshStrategy.NORMAL
    visible: bool = True
    width: Optional[int] = None
    height: Optional[int] = None
    style: Optional[Style] = None
    border_style: Optional[Style] = None
    padding: tuple[int, int] = (0, 1)


@dataclass
class LayoutConfig:
    """Configuration for layout containers."""
    orientation: str = "horizontal"  # horizontal | vertical
    split_sizes: list[int] = field(default_factory=list)
    minimal_sizes: dict[str, int] = field(default_factory=dict)


class BaseComponent(ABC):
    """
    Abstract base class for all UI components.
    
    All components must implement:
    - render() - Return a Rich renderable
    - update() - Handle state changes
    
    Optional lifecycle methods:
    - mount() - Called when component is added to tree
    - unmount() - Called when component is removed
    - resize(width, height) - Handle terminal resize
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        self._state = ComponentState.CREATED
        self._config = config or ComponentConfig()
        self._parent: Optional[BaseComponent] = None
        self._children: list[BaseComponent] = []
        self._console: Optional[Console] = None
        self._refresh_pending = False
        
    @property
    def state(self) -> ComponentState:
        return self._state
    
    @property
    def config(self) -> ComponentConfig:
        return self._config
    
    @property
    def is_visible(self) -> bool:
        return self._config.visible and self._state == ComponentState.VISIBLE
    
    def mount(self, console: Console, parent: Optional[BaseComponent] = None):
        """Called when component is added to the render tree."""
        self._console = console
        self._parent = parent
        self._state = ComponentState.MOUNTED
        
    def unmount(self):
        """Called when component is removed from the render tree."""
        self._state = ComponentState.DESTROYED
        self._parent = None
        
    @abstractmethod
    def render(self):
        """Render the component. Must be implemented by subclasses."""
        pass
    
    def update(self, *args, **kwargs):
        """Handle state updates. Override in subclasses."""
        self._schedule_refresh()
        
    def _schedule_refresh(self):
        """Mark component for refresh on next frame."""
        self._refresh_pending = True
        
    def should_refresh(self) -> bool:
        """Check if component needs to refresh."""
        if not self._config.visible:
            return False
        return self._refresh_pending
    
    def clear_refresh(self):
        """Clear the refresh flag after rendering."""
        self._refresh_pending = False
        
    def resize(self, width: int, height: int):
        """Handle terminal resize. Override in subclasses if needed."""
        pass
    
    def add_child(self, child: BaseComponent):
        """Add a child component."""
        child.mount(self._console, self)
        self._children.append(child)
        
    def remove_child(self, child: BaseComponent):
        """Remove a child component."""
        if child in self._children:
            child.unmount()
            self._children.remove(child)
            
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Rich console protocol implementation."""
        yield self.render()


class Layout(BaseComponent):
    """
    Container that arranges child components.
    
    Supports horizontal and vertical layouts with configurable splits.
    """
    
    def __init__(self, config: Optional[LayoutConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self._layout_config = config or LayoutConfig()
        self._panels: dict[str, BaseComponent] = {}
        
    def add_panel(self, name: str, component: BaseComponent):
        """Add a named panel to the layout."""
        self._panels[name] = component
        self.add_child(component)
        
    def remove_panel(self, name: str):
        """Remove a panel from the layout."""
        if name in self._panels:
            self.remove_child(self._panels[name])
            del self._panels[name]
            
    def get_panel(self, name: str) -> Optional[BaseComponent]:
        """Get a panel by name."""
        return self._panels.get(name)
    
    def render(self) -> Renderable:
        """Render the layout with all panels."""
        if not self._panels:
            return RichPanel("[dim]Empty layout[/dim]", border_style="cyan")
        
        # Create Rich Layout
        layout = RichLayout()
        layout.split_columns(
            *[
                RichLayout(name=name, ratio=1) 
                for name in self._panels.keys()
            ]
        )
        
        # Render each panel
        for name, component in self._panels.items():
            if component.is_visible:
                layout[name].update(component.render())
                
        return layout


class UIPanel(BaseComponent):
    """
    Styled container with border and optional title.
    
    Wraps Rich Panel but adds component lifecycle management.
    """
    
    def __init__(
        self,
        content = None,
        title: str = "",
        border_style: str = "cyan",
        **kwargs
    ):
        super().__init__(**kwargs)
        self._content = content
        self._title = title
        self._border_style = border_style
        
    @property
    def content(self):
        return self._content
    
    @content.setter
    def content(self, value):
        self._content = value
        self._schedule_refresh()
        
    @property
    def title(self) -> str:
        return self._title
    
    @title.setter
    def title(self, value: str):
        self._title = value
        self._schedule_refresh()
        
    def render(self):
        """Render the panel with border and title."""
        from rich.panel import Panel as RichPanel
        
        content = self._content
        if content is None:
            content = "[dim]No content[/dim]"
            
        return RichPanel(
            content,
            title=self._title,
            border_style=self._border_style,
            style=self._config.style,
            padding=self._config.padding,
        )


class ProgressIndicator(BaseComponent):
    """
    Base class for progress indicators.
    
    Provides common functionality for progress bars, spinners, etc.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._progress: float = 0.0
        self._message: str = ""
        self._completed: bool = False
        self._spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._spinner_index = 0
        
    @property
    def progress(self) -> float:
        return self._progress
    
    @progress.setter
    def progress(self, value: float):
        self._progress = max(0.0, min(1.0, value))
        if self._progress >= 1.0:
            self._completed = True
        self._schedule_refresh()
        
    @property
    def message(self) -> str:
        return self._message
    
    @message.setter
    def message(self, value: str):
        self._message = value
        self._schedule_refresh()
        
    @property
    def is_completed(self) -> bool:
        return self._completed
    
    def reset(self):
        """Reset the progress indicator."""
        self._progress = 0.0
        self._completed = False
        self._spinner_index = 0
        self._schedule_refresh()
        
    def spin(self) -> str:
        """Advance spinner and return current character."""
        char = self._spinner_chars[self._spinner_index]
        self._spinner_index = (self._spinner_index + 1) % len(self._spinner_chars)
        return char
        
    def render(self) -> Renderable:
        """Override in subclasses."""
        raise NotImplementedError


class ResizeHandler:
    """
    Mixin for handling terminal resize events.
    
    Use as a mixin to add SIGWINCH handling to components.
    """
    
    _resize_callbacks: list[Callable[[int, int], None]] = []
    _original_handler: Optional[signal.Handler] = None
    
    @classmethod
    def setup_resize_handler(cls):
        """Install SIGWINCH handler."""
        if cls._original_handler is not None:
            return  # Already installed
            
        def handle_resize(signum, frame):
            # Trigger callbacks
            for callback in cls._resize_callbacks:
                try:
                    callback()
                except Exception:
                    pass
                    
        cls._original_handler = signal.signal(signal.SIGWINCH, handle_resize)
        
    @classmethod
    def teardown_resize_handler(cls):
        """Restore original SIGWINCH handler."""
        if cls._original_handler is not None:
            signal.signal(signal.SIGWINCH, cls._original_handler)
            cls._original_handler = None
            
    @classmethod
    def on_resize(cls, callback: Callable[[int, int], None]):
        """Register a resize callback."""
        cls._resize_callbacks.append(callback)


@contextmanager
def component_context(console: Console):
    """Context manager for component lifecycle."""
    ResizeHandler.setup_resize_handler()
    try:
        yield
    finally:
        ResizeHandler.teardown_resize_handler()
