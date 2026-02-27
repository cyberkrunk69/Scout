"""UI theme system package."""
from __future__ import annotations

from scout.ui.theme.manager import ThemeManager, get_theme_manager
from scout.ui.theme.schemes import (
    Theme,
    ColorScheme,
    DEFAULT_THEME,
    MINIMAL_THEME,
    HIGH_CONTRAST_THEME,
    MONOCHROME_THEME,
)

__all__ = [
    "ThemeManager",
    "get_theme_manager",
    "Theme",
    "ColorScheme",
    "DEFAULT_THEME",
    "MINIMAL_THEME",
    "HIGH_CONTRAST_THEME",
    "MONOCHROME_THEME",
]
