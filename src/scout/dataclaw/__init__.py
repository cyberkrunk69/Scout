"""DataClaw - Agent conversation log ingestion for Scout.

This module provides parsers for ingesting conversation logs from various
AI agent IDEs (Cursor, Cline, Windsurf, etc.) into a unified schema.

Usage:
    from scout.dataclaw import get_parser, ingest_agent

    # Get a specific parser
    parser = get_parser("cursor")

    # Discover sessions
    sessions = parser.discover()

    # Parse a session
    parsed = parser.parse(session_id)

    # Or use the ingestion interface
    summary = ingest_agent("cursor")
"""

# Import all parsers to trigger registration decorators
from scout.dataclaw.parsers import cursor  # noqa: F401

from scout.dataclaw.ingest import (
    discover_sessions,
    ingest_agent,
    parse_session,
)
from scout.dataclaw.parsers import (
    BaseParser,
    ParseError,
    get_parser,
    list_parsers,
    register,
)
from scout.dataclaw.parsers.schema import ParsedSession, SchemaMismatchError

# Note: Parsers must be imported before accessing get_parser/list_parsers
# The import of 'cursor' above triggers the @register decorator

__all__ = [
    # Registry
    "BaseParser",
    "get_parser",
    "list_parsers",
    "register",
    # Schema
    "ParsedSession",
    "SchemaMismatchError",
    "ParseError",
    # Ingestion
    "ingest_agent",
    "discover_sessions",
    "parse_session",
]
