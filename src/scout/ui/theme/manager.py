#!/usr/bin/env python
"""
Theme manager - handles theme loading, switching, and persistence.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from scout.ui.theme.schemes import (
    Theme,
    get_theme,
    list_themes,
    THEME_REGISTRY,
    DEFAULT_THEME,
)


class ThemeManager:
    """
    Manages theme selection, loading, and switching.
    
    Supports:
    - Built-in themes (default, minimal, high-contrast, monochrome)
    - Custom themes from ~/.scout/themes/
    - Live theme switching without restart
    - Theme persistence across sessions
    """
    
    def __init__(self, initial_theme: str = "default"):
        self._current_theme: Theme = get_theme(initial_theme)
        self._theme_name = initial_theme
        self._custom_themes: dict[str, Theme] = {}
        self._listeners: list[callable] = []
        self._load_custom_themes()
        
    @property
    def current_theme(self) -> Theme:
        return self._current_theme
    
    @property
    def theme_name(self) -> str:
        return self._theme_name
    
    def switch(self, theme_name: str) -> bool:
        """
        Switch to a different theme.
        
        Args:
            theme_name: Name of theme to switch to
            
        Returns:
            True if theme was found and switched, False otherwise
        """
        # Check built-in themes
        if theme_name in THEME_REGISTRY:
            self._current_theme = THEME_REGISTRY[theme_name]
            self._theme_name = theme_name
            self._notify_listeners()
            return True
            
        # Check custom themes
        if theme_name in self._custom_themes:
            self._current_theme = self._custom_themes[theme_name]
            self._theme_name = theme_name
            self._notify_listeners()
            return True
            
        return False
    
    def register_theme(self, theme: Theme, notify: bool = True):
        """
        Register a custom theme.
        
        Args:
            theme: Theme to register
            notify: Whether to notify listeners of the change
        """
        self._custom_themes[theme.name] = theme
        if notify:
            self._notify_listeners()
            
    def unregister_theme(self, theme_name: str) -> bool:
        """
        Unregister a custom theme.
        
        Args:
            theme_name: Name of theme to remove
            
        Returns:
            True if theme was removed, False if not found
        """
        if theme_name in self._custom_themes:
            del self._custom_themes[theme_name]
            return True
        return False
    
    def _load_custom_themes(self):
        """Load custom themes from ~/.scout/themes/"""
        themes_dir = self._get_themes_dir()
        if not themes_dir.exists():
            return
            
        for theme_file in themes_dir.glob("*.yaml"):
            try:
                self._load_theme_file(theme_file)
            except Exception as e:
                # Log but don't fail on bad theme file
                print(f"Warning: Failed to load theme {theme_file}: {e}")
                
        for theme_file in themes_dir.glob("*.json"):
            try:
                self._load_theme_file(theme_file)
            except Exception as e:
                print(f"Warning: Failed to load theme {theme_file}: {e}")
    
    def _load_theme_file(self, path: Path):
        """Load a theme from a file."""
        import json
        import yaml
        
        with open(path) as f:
            if path.suffix == ".json":
                data = json.load(f)
            else:
                data = yaml.safe_load(f)
                
        # Parse theme from dict (simplified - full implementation would be more robust)
        from scout.ui.theme.schemes import ColorScheme, Typography, Spacing, Animation, Theme
        
        colors = ColorScheme(**data.get("colors", {}))
        typography = Typography(**data.get("typography", {}))
        spacing = Spacing(**data.get("spacing", {}))
        animation = Animation(**data.get("animation", {}))
        
        theme = Theme(
            name=path.stem,
            colors=colors,
            typography=typography,
            spacing=spacing,
            animation=animation,
            description=data.get("description", ""),
        )
        
        self.register_theme(theme, notify=False)
        
    def _get_themes_dir(self) -> Path:
        """Get the custom themes directory."""
        home = Path.home()
        return home / ".scout" / "themes"
    
    def save(self):
        """Save current theme selection to config."""
        # This would save to ~/.scout/config.yaml or similar
        # Implementation depends on config system
        pass
    
    def load(self) -> str:
        """Load saved theme from config and return theme name."""
        # This would load from config
        # Returns the loaded theme name
        return self._theme_name
    
    def add_listener(self, callback: callable):
        """Add a listener that's called when theme changes."""
        self._listeners.append(callback)
        
    def remove_listener(self, callback: callable):
        """Remove a theme change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)
            
    def _notify_listeners(self):
        """Notify all listeners of theme change."""
        for listener in self._listeners:
            try:
                listener(self._current_theme)
            except Exception:
                pass
                
    def get_available_themes(self) -> list[str]:
        """Get list of all available theme names (built-in + custom)."""
        built_in = list(THEME_REGISTRY.keys())
        custom = list(self._custom_themes.keys())
        return sorted(set(built_in + custom))


# Global theme manager instance
_theme_manager: Optional[ThemeManager] = None


def get_theme_manager() -> ThemeManager:
    """Get the global theme manager instance."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager


def set_theme_manager(manager: ThemeManager):
    """Set the global theme manager (for testing)."""
    global _theme_manager
    _theme_manager = manager
