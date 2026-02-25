"""Cursor parser for DataClaw.

Parses Cursor conversation logs from:
- globalStorage/cursorStorage/*/cursorDiskKV (session metadata)
- workspaceStorage/*/User/globalStorage/storage.json (workspace data)

The parser:
1. Copies DB to temp file (avoid locks)
2. Reads cursorDiskKV keys
3. Maps bubble data to unified schema
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from scout.dataclaw.parsers import BaseParser, ParseError, register
from scout.dataclaw.parsers.schema import (
    Message,
    ParsedSession,
    SessionMeta,
    compute_project_hash,
    validate_and_log,
)
from scout.dataclaw.parsers.utils import (
    extract_timestamp,
    get_all_kv_keys,
    normalize_path,
    parse_kv_store,
    temp_copy,
)

logger = logging.getLogger(__name__)

# Default Cursor paths
DEFAULT_CURSOR_PATHS = [
    Path("~/Library/Application Support/Cursor"),
    Path("~/.cursor"),
    Path("~/AppData/Roaming/Cursor"),
]


@register("cursor")
class CursorParser(BaseParser):
    """Parser for Cursor IDE conversation logs."""

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize parser with optional base path override.

        Args:
            base_path: Override for default Cursor data path
        """
        self._base_path = base_path or self._find_cursor_path()

    def _find_cursor_path(self) -> Optional[Path]:
        """Find Cursor data directory.

        Returns:
            Path to Cursor data directory or None
        """
        for path in DEFAULT_CURSOR_PATHS:
            expanded = path.expanduser()
            if expanded.exists():
                return expanded
        return None

    @property
    def base_path(self) -> Optional[Path]:
        """Get the configured base path."""
        return self._base_path

    def discover(self, base_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Discover available Cursor sessions.

        Args:
            base_path: Optional override for base path

        Returns:
            List of session metadata dicts
        """
        cursor_path = base_path or self._base_path
        if not cursor_path or not cursor_path.exists():
            logger.warning("Cursor path not found: %s", cursor_path)
            return []

        sessions = []
        storage_path = cursor_path / "User" / "globalStorage"

        if not storage_path.exists():
            logger.warning("Cursor storage path not found: %s", storage_path)
            return []

        # Each subdirectory under globalStorage is a workspace
        for workspace_dir in storage_path.iterdir():
            if not workspace_dir.is_dir():
                continue

            kv_path = workspace_dir / "cursorDiskKV"
            if not kv_path.exists():
                continue

            try:
                with temp_copy(kv_path) as temp_db:
                    conn = sqlite3.connect(temp_db)
                    try:
                        sessions.extend(self._discover_from_kv(conn, workspace_dir))
                    finally:
                        conn.close()
            except Exception as e:
                logger.warning("Failed to read cursorDiskKV in %s: %s", workspace_dir, e)

        return sessions

    def _discover_from_kv(
        self,
        conn: sqlite3.Connection,
        workspace_dir: Path
    ) -> List[Dict[str, Any]]:
        """Discover sessions from a cursorDiskKV database.

        Args:
            conn: SQLite connection to temp copy
            workspace_dir: Workspace directory path

        Returns:
            List of session metadata
        """
        sessions = []
        keys = get_all_kv_keys(conn)

        # Look for composerData keys which contain session info
        composer_keys = [k for k in keys if k.startswith("composerData:")]

        for key in composer_keys:
            try:
                cursor = conn.execute(
                    "SELECT value FROM cursorDiskKV WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                if not row:
                    continue

                value = json.loads(row[0])
                session_id = key.replace("composerData:", "")

                # Extract timestamp from the data
                timestamp = extract_timestamp(
                    value,
                    "timestamp",
                    "createdAt",
                    "created_at",
                    "ts"
                ) or datetime.now().isoformat()

                sessions.append({
                    "id": session_id,
                    "path": str(workspace_dir),
                    "timestamp": timestamp,
                    "workspace": workspace_dir.name,
                })
            except (json.JSONDecodeError, sqlite3.Error) as e:
                logger.warning("Failed to parse key %s: %s", key, e)

        # Also look for bubbleId keys
        bubble_keys = [k for k in keys if k.startswith("bubbleId:")]

        for key in bubble_keys:
            try:
                cursor = conn.execute(
                    "SELECT value FROM cursorDiskKV WHERE key = ?", (key,)
                )
                row = cursor.fetchone()
                if not row:
                    continue

                value = json.loads(row[0])
                session_id = key.replace("bubbleId:", "")

                timestamp = extract_timestamp(
                    value,
                    "timestamp",
                    "createdAt",
                    "created_at",
                    "ts"
                ) or datetime.now().isoformat()

                # Avoid duplicates
                if not any(s["id"] == session_id for s in sessions):
                    sessions.append({
                        "id": session_id,
                        "path": str(workspace_dir),
                        "timestamp": timestamp,
                        "workspace": workspace_dir.name,
                    })
            except (json.JSONDecodeError, sqlite3.Error) as e:
                logger.warning("Failed to parse key %s: %s", key, e)

        return sessions

    def parse(
        self,
        session_id: str,
        base_path: Optional[Path] = None
    ) -> ParsedSession:
        """Parse a single Cursor session into unified schema.

        Args:
            session_id: The session identifier (workspace path or session ID)
            base_path: Optional base path override

        Returns:
            ParsedSession conforming to unified schema

        Raises:
            ParseError: If session cannot be parsed
        """
        cursor_path = base_path or self._base_path
        if not cursor_path or not cursor_path.exists():
            raise ParseError(f"Cursor path not found: {cursor_path}")

        storage_path = cursor_path / "User" / "globalStorage"
        if not storage_path.exists():
            raise ParseError(f"Cursor storage not found: {storage_path}")

        # Find the workspace that contains this session
        session_data: Optional[Dict[str, Any]] = None
        workspace_path: Optional[Path] = None

        for workspace_dir in storage_path.iterdir():
            if not workspace_dir.is_dir():
                continue

            kv_path = workspace_dir / "cursorDiskKV"
            if not kv_path.exists():
                continue

            try:
                with temp_copy(kv_path) as temp_db:
                    conn = sqlite3.connect(temp_db)
                    try:
                        data = self._parse_session_from_kv(conn, session_id)
                        if data:
                            session_data = data
                            workspace_path = workspace_dir
                            break
                    finally:
                        conn.close()
            except Exception as e:
                logger.warning("Failed to read cursorDiskKV in %s: %s", workspace_dir, e)

        if not session_data:
            raise ParseError(f"Session not found: {session_id}")

        # Build the unified schema
        project_path = str(workspace_path) if workspace_path else None
        normalized_project = (
            normalize_path(Path(project_path)) if project_path else None
        )

        meta = SessionMeta(
            source_agent="cursor",
            session_id=session_id,
            project_hash=compute_project_hash(normalized_project),
            start_time=session_data.get("start_time"),
            end_time=session_data.get("end_time"),
            project_path=normalized_project,
        )

        messages = self._build_messages(session_data)

        raw = {
            "workspace": workspace_path.name if workspace_path else None,
            "raw_composer_data": session_data.get("raw_composer"),
            "raw_bubble_data": session_data.get("raw_bubble"),
        }

        parsed = ParsedSession(
            meta=meta,
            messages=messages,
            raw=raw,
        )

        # Validate and return
        return validate_and_log(
            parsed.model_dump(),
            session_id,
            log_raw=True
        )

    def _parse_session_from_kv(
        self,
        conn: sqlite3.Connection,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Parse a specific session from cursorDiskKV.

        Args:
            conn: SQLite connection
            session_id: Session ID to parse

        Returns:
            Session data dict or None
        """
        # Try composerData key first
        composer_key = f"composerData:{session_id}"
        cursor = conn.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?", (composer_key,)
        )
        row = cursor.fetchone()

        if row:
            try:
                composer_data = json.loads(row[0])
                return self._extract_session_from_composer(composer_data, session_id)
            except (json.JSONDecodeError, sqlite3.Error) as e:
                logger.warning("Failed to parse composerData: %s", e)

        # Try bubbleId key
        bubble_key = f"bubbleId:{session_id}"
        cursor = conn.execute(
            "SELECT value FROM cursorDiskKV WHERE key = ?", (bubble_key,)
        )
        row = cursor.fetchone()

        if row:
            try:
                bubble_data = json.loads(row[0])
                return self._extract_session_from_bubble(bubble_data, session_id)
            except (json.JSONDecodeError, sqlite3.Error) as e:
                logger.warning("Failed to parse bubbleId: %s", e)

        return None

    def _extract_session_from_composer(
        self,
        data: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Extract session data from composerData.

        Args:
            data: Raw composer data
            session_id: Session ID

        Returns:
            Extracted session data
        """
        messages = []

        # Extract conversation/history from composer data
        history = data.get("history", []) or data.get("messages", []) or []

        for item in history:
            role = item.get("role", "assistant")
            content = item.get("content", "") or item.get("text", "")

            if not content:
                continue

            msg = Message(
                role=role,
                content=str(content),
                timestamp=extract_timestamp(
                    item, "timestamp", "ts", "created_at"
                ),
                tool_calls=item.get("tool_calls"),
                tool_results=item.get("tool_results"),
                model=item.get("model"),
            )
            messages.append(msg)

        # Also look for conversation key
        conversation = data.get("conversation", [])
        for item in conversation:
            role = item.get("role", "assistant")
            content = item.get("content", "") or item.get("text", "")

            if not content:
                continue

            # Avoid duplicates
            if not any(m.content == content for m in messages):
                msg = Message(
                    role=role,
                    content=str(content),
                    timestamp=extract_timestamp(
                        item, "timestamp", "ts", "created_at"
                    ),
                )
                messages.append(msg)

        return {
            "start_time": extract_timestamp(data, "timestamp", "createdAt", "ts"),
            "end_time": extract_timestamp(data, "completedAt", "end_time"),
            "messages": messages,
            "raw_composer": data,
        }

    def _extract_session_from_bubble(
        self,
        data: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Extract session data from bubbleId data.

        Args:
            data: Raw bubble data
            session_id: Session ID

        Returns:
            Extracted session data
        """
        messages = []

        # Extract messages from bubble
        messages_data = data.get("messages", []) or data.get("history", []) or []

        for item in messages_data:
            role = item.get("role", "assistant")
            content = item.get("content", "") or item.get("text", "")

            if not content:
                continue

            msg = Message(
                role=role,
                content=str(content),
                timestamp=extract_timestamp(
                    item, "timestamp", "ts", "created_at"
                ),
                tool_calls=item.get("tool_calls"),
                tool_results=item.get("tool_results"),
            )
            messages.append(msg)

        return {
            "start_time": extract_timestamp(data, "timestamp", "createdAt", "ts"),
            "end_time": extract_timestamp(data, "completedAt", "end_time"),
            "messages": messages,
            "raw_bubble": data,
        }

    def _build_messages(self, session_data: Dict[str, Any]) -> List[Message]:
        """Build message list from session data.

        Args:
            session_data: Parsed session data

        Returns:
            List of Message objects
        """
        return session_data.get("messages", [])


def discover_cursor_sessions(
    base_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """Convenience function to discover all Cursor sessions.

    Args:
        base_path: Optional base path override

    Returns:
        List of session metadata
    """
    parser = CursorParser(base_path)
    return parser.discover()


def parse_cursor_session(
    session_id: str,
    base_path: Optional[Path] = None
) -> ParsedSession:
    """Convenience function to parse a Cursor session.

    Args:
        session_id: Session ID to parse
        base_path: Optional base path override

    Returns:
        Parsed session in unified schema
    """
    parser = CursorParser(base_path)
    return parser.parse(session_id)
