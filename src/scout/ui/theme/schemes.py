#!/usr/bin/env python
"""
Theme color schemes - defines color palettes for Scout UI.

Each theme includes colors for:
- Primary UI elements (borders, titles)
- Status indicators (success, error, warning, info)
- Text styles (primary, secondary, dim)
- Special elements (reasoning, action, logs)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ColorScheme:
    """Complete color scheme for a theme."""
    
    # Primary UI colors
    primary: str = "cyan"
    secondary: str = "magenta"
    accent: str = "green"
    
    # Background colors
    background: str = ""
    panel_bg: str = ""
    highlight: str = ""
    
    # Text colors
    text_primary: str = "white"
    text_secondary: str = "dim"
    text_dim: str = "dim"
    
    # Status colors
    success: str = "green"
    error: str = "red"
    warning: str = "yellow"
    info: str = "blue"
    
    # Special element colors
    reasoning: str = "cyan"
    action: str = "green"
    logs: str = "magenta"
    brain_activity: str = "magenta"
    parallel: str = "yellow"
    
    # Border styles
    border_default: str = "cyan"
    border_error: str = "red"
    border_success: str = "green"
    border_warning: str = "yellow"


@dataclass 
class Typography:
    """Typography settings for a theme."""
    font_family: str = "default"
    font_size: int = 12
    line_height: int = 1
    title_style: str = "bold"
    dim_style: str = "dim"


@dataclass
class Spacing:
    """Spacing constants for layout."""
    panel_padding: tuple[int, int] = (0, 1)
    component_margin: int = 1
    grid_unit: int = 4


@dataclass
class Animation:
    """Animation settings."""
    refresh_rate: float = 4.0
    spinner_speed: float = 0.1
    transition_speed: float = 0.2
    enable_animations: bool = True


@dataclass
class Theme:
    """Complete theme definition."""
    name: str
    colors: ColorScheme
    typography: Typography
    spacing: Spacing
    animation: Animation
    description: str = ""
    

# Default theme - colorful, informative
DEFAULT_THEME = Theme(
    name="default",
    colors=ColorScheme(),
    typography=Typography(),
    spacing=Spacing(),
    animation=Animation(refresh_rate=4.0),
    description="Default Scout theme with rich colors",
)

# Minimal theme - reduced visual noise
MINIMAL_THEME = Theme(
    name="minimal",
    colors=ColorScheme(
        primary="white",
        secondary="white",
        accent="white",
        reasoning="white",
        action="white",
        logs="white",
        brain_activity="white",
        parallel="white",
    ),
    typography=Typography(dim_style=""),
    spacing=Spacing(panel_padding=(0, 0), component_margin=0),
    animation=Animation(refresh_rate=2.0, enable_animations=False),
    description="Minimal theme with reduced visual elements",
)

# High contrast theme - for accessibility
HIGH_CONTRAST_THEME = Theme(
    name="high-contrast",
    colors=ColorScheme(
        primary="bright_cyan",
        secondary="bright_magenta",
        accent="bright_green",
        text_primary="bright_white",
        text_secondary="white",
        success="bright_green",
        error="bright_red",
        warning="bright_yellow",
        info="bright_blue",
        reasoning="bright_cyan",
        action="bright_green",
    ),
    typography=Typography(font_size=14),
    spacing=Spacing(panel_padding=(1, 2), component_margin=2),
    animation=Animation(refresh_rate=4.0),
    description="High contrast theme for accessibility",
)

# Monochrome theme - black and white only
MONOCHROME_THEME = Theme(
    name="monochrome",
    colors=ColorScheme(
        primary="white",
        secondary="white",
        accent="white",
        text_primary="white",
        text_secondary="bright_black",
        success="white",
        error="white",
        warning="white",
        info="white",
        reasoning="white",
        action="white",
        logs="bright_black",
        brain_activity="bright_black",
        parallel="bright_black",
    ),
    typography=Typography(),
    spacing=Spacing(),
    animation=Animation(refresh_rate=2.0),
    description="Monochrome theme for terminals without color support",
)


# Registry of all available themes
THEME_REGISTRY: dict[str, Theme] = {
    "default": DEFAULT_THEME,
    "minimal": MINIMAL_THEME,
    "high-contrast": HIGH_CONTRAST_THEME,
    "monochrome": MONOCHROME_THEME,
}


def get_theme(name: str) -> Theme:
    """Get a theme by name, returning default if not found."""
    return THEME_REGISTRY.get(name, DEFAULT_THEME)


def list_themes() -> list[str]:
    """Get list of available theme names."""
    return list(THEME_REGISTRY.keys())
