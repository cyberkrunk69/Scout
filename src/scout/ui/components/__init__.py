"""UI components package."""
from __future__ import annotations

from scout.ui.components.base import BaseComponent, Layout, UIPanel
from scout.ui.components.reasoning import ReasoningDisplay
from scout.ui.components.cost_display import CostBreakdown
from scout.ui.components.error_display import ErrorDisplay
from scout.ui.components.toast import Toast, ToastManager
from scout.ui.components.command_palette import CommandPalette

__all__ = [
    "BaseComponent",
    "Layout", 
    "UIPanel",
    "ReasoningDisplay",
    "CostBreakdown",
    "ErrorDisplay",
    "Toast",
    "ToastManager",
    "CommandPalette",
]