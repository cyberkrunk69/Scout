"""Timeout configuration for LLM calls.

Provides configurable timeouts for different LLM providers and operations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TimeoutConfig:
    """Configuration for LLM request timeouts."""
    
    # Timeouts in seconds
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    
    # Provider-specific overrides
    deepseek_connect: float = 10.0
    deepseek_read: float = 60.0
    
    minimax_connect: float = 15.0
    minimax_read: float = 90.0
    
    @classmethod
    def from_env(cls) -> "TimeoutConfig":
        """Create config from environment variables."""
        return cls(
            connect_timeout=float(os.environ.get("SCOUT_CONNECT_TIMEOUT", "10.0")),
            read_timeout=float(os.environ.get("SCOUT_READ_TIMEOUT", "60.0")),
            deepseek_connect=float(os.environ.get("SCOUT_DEEPSEEK_CONNECT_TIMEOUT", "10.0")),
            deepseek_read=float(os.environ.get("SCOUT_DEEPSEEK_READ_TIMEOUT", "60.0")),
            minimax_connect=float(os.environ.get("SCOUT_MINIMAX_CONNECT_TIMEOUT", "15.0")),
            minimax_read=float(os.environ.get("SCOUT_MINIMAX_READ_TIMEOUT", "90.0")),
        )
    
    def for_provider(self, provider: str) -> tuple[float, float]:
        """Get (connect_timeout, read_timeout) for a provider."""
        if provider == "deepseek":
            return self.deepseek_connect, self.deepseek_read
        elif provider == "minimax":
            return self.minimax_connect, self.minimax_read
        else:
            return self.connect_timeout, self.read_timeout


class TimeoutError(Exception):
    """Request timeout."""
    pass


# Global config instance
_config: Optional[TimeoutConfig] = None


def get_timeout_config() -> TimeoutConfig:
    """Get global timeout config."""
    global _config
    if _config is None:
        _config = TimeoutConfig.from_env()
    return _config


def set_timeout_config(config: TimeoutConfig) -> None:
    """Set global timeout config."""
    global _config
    _config = config
