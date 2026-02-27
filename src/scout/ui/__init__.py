"""Scout UI layer — components, themes, state management, and configuration."""
from __future__ import annotations

# Components
from scout.ui.components.base import (
    BaseComponent,
    ComponentConfig,
    ComponentState,
    Layout,
    UIPanel,
    ProgressIndicator,
    RefreshStrategy,
)
from scout.ui.components.reasoning import ReasoningDisplay, ParallelProcessDisplay
from scout.ui.components.cost_display import CostBreakdown, CostTracker
from scout.ui.components.error_display import ErrorDisplay, ErrorBuilder, ErrorCategory, ErrorSeverity
from scout.ui.components.toast import ToastManager, Toast, ToastType
from scout.ui.components.command_palette import CommandPalette, Command, CommandPaletteCompleter

# Theme
from scout.ui.theme import ThemeManager, get_theme_manager
from scout.ui.theme.schemes import (
    Theme,
    ColorScheme,
    DEFAULT_THEME,
    MINIMAL_THEME,
    HIGH_CONTRAST_THEME,
    MONOCHROME_THEME,
)

# State management
from scout.ui.repl_state import (
    REPLState,
    REPLEvent,
    REPLStateMachine,
    EventBus,
    CommandQueue,
    get_state_machine,
    get_command_queue,
    reset_state,
)

# Configuration
from scout.ui.config import (
    UIConfig,
    get_ui_config,
    set_ui_config,
    load_ui_config,
    save_ui_config,
)

__all__ = [
    # Base components
    "BaseComponent",
    "ComponentConfig", 
    "ComponentState",
    "Layout",
    "UIPanel",
    "ProgressIndicator",
    "RefreshStrategy",
    # Display components
    "ReasoningDisplay",
    "ParallelProcessDisplay",
    "CostBreakdown",
    "CostTracker",
    "ErrorDisplay",
    "ErrorBuilder",
    "ErrorCategory",
    "ErrorSeverity",
    "ToastManager",
    "Toast",
    "ToastType",
    "CommandPalette",
    "Command",
    "CommandPaletteCompleter",
    # Theme
    "ThemeManager",
    "get_theme_manager",
    "Theme",
    "ColorScheme",
    "DEFAULT_THEME",
    "MINIMAL_THEME", 
    "HIGH_CONTRAST_THEME",
    "MONOCHROME_THEME",
    # State
    "REPLState",
    "REPLEvent",
    "REPLStateMachine",
    "EventBus",
    "CommandQueue",
    "get_state_machine",
    "get_command_queue",
    "reset_state",
    # Config
    "UIConfig",
    "get_ui_config",
    "set_ui_config",
    "load_ui_config",
    "save_ui_config",
]
