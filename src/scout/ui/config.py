#!/usr/bin/env python
"""
UI configuration - settings for the Scout UI layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class UIConfig:
    """Configuration for the Scout UI."""
    
    # Theme settings
    theme: str = "default"
    custom_themes_dir: Path = Path.home() / ".scout" / "themes"
    
    # Display settings
    refresh_rate: float = 4.0
    show_brain_activity: bool = True
    show_parallel_processes: bool = True
    max_reasoning_lines: int = 10
    max_log_lines: int = 15
    
    # Animation settings
    enable_animations: bool = True
    spinner_speed: float = 0.1
    
    # Cost display
    show_cost: bool = True
    budget_warning_threshold: float = 0.8
    default_budget: Optional[float] = None
    
    # Error display
    show_suggestions: bool = True
    show_traceback: bool = True
    
    # Command palette
    enable_command_palette: bool = True
    recent_commands_count: int = 20
    
    @classmethod
    def from_dict(cls, data: dict) -> "UIConfig":
        """Create config from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def to_dict(self) -> dict:
        """Convert config to dict."""
        return {
            "theme": self.theme,
            "custom_themes_dir": str(self.custom_themes_dir),
            "refresh_rate": self.refresh_rate,
            "show_brain_activity": self.show_brain_activity,
            "show_parallel_processes": self.show_parallel_processes,
            "max_reasoning_lines": self.max_reasoning_lines,
            "max_log_lines": self.max_log_lines,
            "enable_animations": self.enable_animations,
            "spinner_speed": self.spinner_speed,
            "show_cost": self.show_cost,
            "budget_warning_threshold": self.budget_warning_threshold,
            "default_budget": self.default_budget,
            "show_suggestions": self.show_suggestions,
            "show_traceback": self.show_traceback,
            "enable_command_palette": self.enable_command_palette,
            "recent_commands_count": self.recent_commands_count,
        }


# Global config instance
_config: Optional[UIConfig] = None


def get_ui_config() -> UIConfig:
    """Get the global UI config."""
    global _config
    if _config is None:
        _config = UIConfig()
    return _config


def set_ui_config(config: UIConfig):
    """Set the global UI config."""
    global _config
    _config = config


def load_ui_config() -> UIConfig:
    """Load UI config from file."""
    config = get_ui_config()
    
    # Try to load from ~/.scout/config.yaml
    config_path = Path.home() / ".scout" / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                data = yaml.safe_load(f)
            if data and "ui" in data:
                config = UIConfig.from_dict(data["ui"])
                set_ui_config(config)
        except Exception:
            pass
            
    return config


def save_ui_config(config: UIConfig):
    """Save UI config to file."""
    import yaml
    
    config_path = Path.home() / ".scout" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing config
    existing = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            existing = {}
            
    # Update UI section
    existing["ui"] = config.to_dict()
    
    # Save
    with open(config_path, "w") as f:
        yaml.safe_dump(existing, f, default_flow_style=False)
        
    set_ui_config(config)
