"""Ingestion module for DataClaw parsers.

Provides functions to:
- Discover and parse sessions from registered agents
- Log parsing results to scout.audit
- Support CLI-style ingestion via dataclaw ingest --agent cursor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from scout.audit import AuditLog, get_audit

from scout.dataclaw.parsers import get_parser, list_parsers
from scout.dataclaw.parsers.cursor import CursorParser
from scout.dataclaw.parsers.schema import ParsedSession, SchemaMismatchError

logger = logging.getLogger(__name__)


def discover_sessions(
    agent: str,
    base_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Discover available sessions for an agent.

    Args:
        agent: Agent name (e.g., "cursor")
        base_path: Optional base path override

    Returns:
        List of session metadata

    Raises:
        ValueError: If agent is not registered
    """
    parser_cls = get_parser(agent)
    if not parser_cls:
        available = list_parsers()
        raise ValueError(
            f"No parser for agent '{agent}'. Available: {available}"
        )

    parser = parser_cls()
    return parser.discover(base_path)


def parse_session(
    agent: str,
    session_id: str,
    base_path: Optional[Path] = None,
    audit_log: Optional[AuditLog] = None
) -> ParsedSession:
    """Parse a single session and log to audit.

    Args:
        agent: Agent name
        session_id: Session ID to parse
        base_path: Optional base path override
        audit_log: Optional audit log instance (uses global if not provided)

    Returns:
        Parsed session in unified schema

    Raises:
        ValueError: If agent is not registered
        SchemaMismatchError: If parsed data doesn't match schema
    """
    parser_cls = get_parser(agent)
    if not parser_cls:
        available = list_parsers()
        raise ValueError(
            f"No parser for agent '{agent}'. Available: {available}"
        )

    parser = parser_cls()
    audit = audit_log or get_audit()

    try:
        parsed = parser.parse(session_id, base_path)

        # Log successful parse
        audit.log(
            "enrich",
            session_id=parsed.meta.session_id,
            source_agent=parsed.meta.source_agent,
            message_count=len(parsed.messages),
            project_path=parsed.meta.project_path,
            start_time=parsed.meta.start_time,
            end_time=parsed.meta.end_time,
            schema_version=parsed.schema_version,
        )

        logger.info(
            "Parsed session %s: %d messages, agent=%s",
            session_id,
            len(parsed.messages),
            agent
        )

        return parsed

    except SchemaMismatchError as e:
        # Log schema mismatch
        audit.log(
            "enrich",
            session_id=session_id,
            source_agent=agent,
            error="schema_mismatch",
            reason=str(e)[:200],
        )
        logger.error("Schema mismatch for session %s: %s", session_id, e)
        raise

    except Exception as e:
        # Log parse error
        audit.log(
            "enrich",
            session_id=session_id,
            source_agent=agent,
            error="parse_error",
            reason=str(e)[:200],
        )
        logger.error("Failed to parse session %s: %s", session_id, e)
        raise


def ingest_agent(
    agent: str,
    base_path: Optional[Path] = None,
    session_filter: Optional[str] = None,
    audit_log: Optional[AuditLog] = None
) -> Dict[str, Any]:
    """Ingest all sessions for an agent.

    Args:
        agent: Agent name
        base_path: Optional base path override
        session_filter: Optional session ID to filter (ingest single session)
        audit_log: Optional audit log instance

    Returns:
        Summary dict with counts
    """
    audit = audit_log or get_audit()

    # Discover sessions
    sessions = discover_sessions(agent, base_path)

    # Filter if requested
    if session_filter:
        sessions = [s for s in sessions if s["id"] == session_filter]

    total = len(sessions)
    success = 0
    errors = 0
    total_messages = 0

    for session in sessions:
        session_id = session["id"]
        try:
            parsed = parse_session(agent, session_id, base_path, audit)
            success += 1
            total_messages += len(parsed.messages)
        except Exception:
            errors += 1

    summary = {
        "agent": agent,
        "total_sessions": total,
        "successful": success,
        "errors": errors,
        "total_messages": total_messages,
    }

    logger.info(
        "Ingested %s: %d/%d sessions, %d messages",
        agent,
        success,
        total,
        total_messages
    )

    return summary


# CLI entry point (can be invoked via python -m scout.dataclaw.ingest)
def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for dataclaw ingestion.

    Usage:
        python -m scout.dataclaw.ingest --agent cursor
        python -m scout.dataclaw.ingest --agent cursor --path ~/.cursor
        python -m scout.dataclaw.ingest --agent cursor --session <session_id>
    """
    import argparse

    parser = argparse.ArgumentParser(description="DataClaw session ingestion")
    parser.add_argument(
        "--agent",
        required=True,
        help="Agent to ingest (e.g., cursor)"
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="Override base path"
    )
    parser.add_argument(
        "--session",
        help="Ingest specific session only"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available agents and exit"
    )

    args = parser.parse_args(argv)

    if args.list:
        agents = list_parsers()
        print("Available agents:", ", ".join(agents) if agents else "none")
        return 0

    try:
        summary = ingest_agent(
            agent=args.agent,
            base_path=args.path,
            session_filter=args.session,
        )
        print(f"Ingested {summary['agent']}:")
        print(f"  Sessions: {summary['successful']}/{summary['total_sessions']}")
        print(f"  Messages: {summary['total_messages']}")
        print(f"  Errors: {summary['errors']}")
        return 0
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
