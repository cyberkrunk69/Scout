"""Parser registry for DataClaw agent conversation log ingestion.

Registry pattern allowing new agent parsers to be added without modifying
existing code.

Usage:
    from scout.dataclaw.parsers import get_parser, register

    @register("cursor")
    class CursorParser:
        def discover(self, base_path: Path) -> list[dict]:
            ...

        def parse(self, session_id: str) -> dict:
            ...

    parser = get_parser("cursor")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Type

from .schema import ParsedSession


_PARSERS: Dict[str, Type["BaseParser"]] = {}


def register(agent: str):
    """Decorator to register a parser for a specific agent.

    Args:
        agent: The agent name (e.g., "cursor", "cline", "windsurf")

    Returns:
        Decorator function that registers the parser class
    """
    def decorator(cls: Type["BaseParser"]) -> Type["BaseParser"]:
        _PARSERS[agent] = cls
        return cls
    return decorator


def get_parser(agent: str) -> Optional[Type["BaseParser"]]:
    """Get the parser class for a specific agent.

    Args:
        agent: The agent name to get parser for

    Returns:
        Parser class or None if not registered
    """
    return _PARSERS.get(agent)


def list_parsers() -> list[str]:
    """Get list of all registered agent names."""
    return list(_PARSERS.keys())


class BaseParser:
    """Base class for all agent parsers.

    Subclasses must implement discover() and parse() methods.
    """

    def discover(self, base_path: Path) -> list[Dict[str, Any]]:
        """Discover available sessions for this agent.

        Args:
            base_path: Base path to agent data (e.g., ~/.cursor)

        Returns:
            List of session metadata dicts with keys:
            - id: Session identifier
            - path: Path to session data
            - timestamp: Session timestamp (ISO format)
        """
        raise NotImplementedError

    def parse(self, session_id: str, base_path: Optional[Path] = None) -> ParsedSession:
        """Parse a single session into the unified schema.

        Args:
            session_id: The session identifier
            base_path: Optional base path override

        Returns:
            ParsedSession conforming to unified schema

        Raises:
            ParseError: If session cannot be parsed
        """
        raise NotImplementedError


class ParseError(Exception):
    """Raised when a session cannot be parsed."""
    pass


__all__ = [
    "BaseParser",
    "ParseError",
    "get_parser",
    "list_parsers",
    "register",
]
