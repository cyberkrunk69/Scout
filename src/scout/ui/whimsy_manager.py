#!/usr/bin/env python
"""
WhimsyManager - State management for V12 Whimsy Engine UI.

Provides:
- Ghost Panel management (show/hide panels based on content)
- Dynamic UI string registry
- State persistence
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)

# Default state file location
DEFAULT_STATE_DIR = Path.home() / ".scout" / "state"
DEFAULT_STATE_FILE = DEFAULT_STATE_DIR / "whimsy_state.json"


class WhimsyManager:
    """
    Central manager for V12 Whimsy Engine UI state.
    
    Handles:
    - Ghost Panels (panels with 0 width when empty)
    - Dynamic UI strings
    - State persistence
    """
    
    def __init__(self, state_file: Optional[Path] = None):
        self._state_file = state_file or DEFAULT_STATE_FILE
        self._ghost_panels: dict[str, bool] = {}  # panel_id -> visible
        self._ui_strings: dict[str, str] = {}
        self._load_state()
    
    # =========================================================================
    # Ghost Panel Management
    # =========================================================================
    
    def register_ghost_panel(self, panel_id: str, default_visible: bool = True) -> None:
        """Register a ghost panel with default visibility."""
        if panel_id not in self._ghost_panels:
            self._ghost_panels[panel_id] = default_visible
            self._save_state()
    
    def show_ghost_panel(self, panel_id: str) -> None:
        """Show a ghost panel."""
        self._ghost_panels[panel_id] = True
        self._save_state()
    
    def hide_ghost_panel(self, panel_id: str) -> None:
        """Hide a ghost panel (set to 0 terminal width)."""
        self._ghost_panels[panel_id] = False
        self._save_state()
    
    def toggle_ghost_panel(self, panel_id: str) -> bool:
        """Toggle panel visibility. Returns new state."""
        current = self._ghost_panels.get(panel_id, True)
        self._ghost_panels[panel_id] = not current
        self._save_state()
        return not current
    
    def get_active_panels(self) -> list[str]:
        """Get list of visible panel IDs."""
        return [pid for pid, visible in self._ghost_panels.items() if visible]
    
    def is_panel_visible(self, panel_id: str) -> bool:
        """Check if a panel is visible."""
        return self._ghost_panels.get(panel_id, True)
    
    def get_panel_width(self, panel_id: str, default_width: int = 20) -> int:
        """Get terminal width for a panel (0 if hidden)."""
        if panel_id in self._ghost_panels and not self._ghost_panels[panel_id]:
            return 0  # Ghost panel - hidden
        return default_width
    
    # =========================================================================
    # Dynamic UI String Management
    # =========================================================================
    
    def register_string(self, key: str, value: str) -> None:
        """Register a UI string."""
        self._ui_strings[key] = value
        self._save_state()
    
    def get_string(self, key: str, default: str = "") -> str:
        """Get a UI string."""
        return self._ui_strings.get(key, default)
    
    def register_strings_from_dict(self, strings: dict[str, str]) -> None:
        """Register multiple UI strings at once."""
        self._ui_strings.update(strings)
        self._save_state()
    
    def get_all_strings(self) -> dict[str, str]:
        """Get all registered UI strings."""
        return self._ui_strings.copy()
    
    def clear_strings(self) -> None:
        """Clear all UI strings."""
        self._ui_strings.clear()
        self._save_state()
    
    # =========================================================================
    # Status String Convenience Methods
    # =========================================================================
    
    def get_status_string(self, status: str, default: str = "") -> str:
        """Get status string - convenience method for 'Generating plan...' etc."""
        return self.get_string(f"status_{status}", default)
    
    def set_status_string(self, status: str, value: str) -> None:
        """Set status string."""
        self.register_string(f"status_{status}", value)
    
    # =========================================================================
    # State Persistence
    # =========================================================================
    
    def _load_state(self) -> None:
        """Load state from file."""
        try:
            if self._state_file.exists():
                with open(self._state_file, 'r') as f:
                    data = json.load(f)
                    self._ghost_panels = data.get("ghost_panels", {})
                    self._ui_strings = data.get("ui_strings", {})
        except Exception as e:
            logger.warning(f"Failed to load whimsy state: {e}")
            self._ghost_panels = {}
            self._ui_strings = {}
    
    def _save_state(self) -> None:
        """Save state to file."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, 'w') as f:
                json.dump({
                    "ghost_panels": self._ghost_panels,
                    "ui_strings": self._ui_strings,
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save whimsy state: {e}")
    
    def reset(self) -> None:
        """Reset all state to defaults."""
        self._ghost_panels.clear()
        self._ui_strings.clear()
        self._save_state()


# Global instance
_whimsy_manager: Optional[WhimsyManager] = None


def get_whimsy_manager() -> WhimsyManager:
    """Get the global WhimsyManager instance."""
    global _whimsy_manager
    if _whimsy_manager is None:
        _whimsy_manager = WhimsyManager()
    return _whimsy_manager
