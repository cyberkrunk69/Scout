"""VS Code Extension Chat History Parsers.

Parses chat history from various VS Code AI assistant extensions:
- Copilot (GitHub)
- Cline (Claude Dev)
- Continue

Each parser conforms to the unified AgentParser interface.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator

from scout.config import (
    AGENT_COPILOT,
    AGENT_CLINE,
    AGENT_CONTINUE,
    COPILOT_CHAT_SESSION_INDEX,
    COPILOT_CHAT_SESSIONS_DIR,
    CLINE_EXTENSION_ID,
    CLINE_API_CONVERSATION_DIR,
    CLINE_API_CONVERSATION_FILE,
    CLINE_FALLBACK_FILE,
    CONTINUE_CONFIG_DIR,
    CONTINUE_SESSIONS_FILE,
    CONTINUE_HISTORY_DB,
)
from scout.vscode_storage import (
    locate_vscode_storage,
    locate_workspace_storage,
    parse_json_file,
    get_extension_storage,
)

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a normalized chat message."""
    role: str  # "user", "assistant", "system", "tool"
    content: str
    timestamp: Optional[datetime] = None
    model: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "role": self.role,
            "content": self.content,
        }
        if self.timestamp:
            result["timestamp"] = self.timestamp.isoformat()
        if self.model:
            result["model"] = self.model
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.tool_results:
            result["tool_results"] = self.tool_results
        return result


@dataclass
class ParsedSession:
    """Represents a parsed chat session."""
    session_id: str
    agent: str
    messages: List[ChatMessage]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "session_id": self.session_id,
            "agent": self.agent,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class ParseResult:
    """Result of parsing all sessions for an agent."""
    agent: str
    sessions: List[ParsedSession]
    total_sessions: int
    successful: int
    failed: int
    duration_ms: float

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_sessions == 0:
            return 0.0
        return (self.successful / self.total_sessions) * 100


class AgentParser(ABC):
    """Abstract base class for VS Code extension chat parsers."""

    def __init__(self, workspace_path: Optional[Path] = None):
        """Initialize parser with optional workspace path.

        Args:
            workspace_path: Optional path to workspace for workspace-specific storage
        """
        self.workspace_path = workspace_path
        self._storage_paths: Dict[str, Optional[Path]] = {}

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Return the agent identifier (copilot, cline, continue)."""
        pass

    @abstractmethod
    def get_storage_path(self) -> Optional[Path]:
        """Get the storage path for this agent's data.

        Returns:
            Path to agent storage or None if not found
        """
        pass

    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List available session identifiers.

        Returns:
            List of session IDs
        """
        pass

    @abstractmethod
    def parse_session(self, session_id: str) -> ParsedSession:
        """Parse a single session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ParsedSession object
        """
        pass

    def parse_all(self) -> ParseResult:
        """Parse all available sessions.

        Returns:
            ParseResult with all parsed sessions
        """
        import time
        start_time = time.time()

        sessions = self.list_sessions()
        results: List[ParsedSession] = []
        successful = 0
        failed = 0

        for session_id in sessions:
            try:
                session = self.parse_session(session_id)
                results.append(session)
                if session.success:
                    successful += 1
                    logger.info(f"Parsed {self.agent_id} session {session_id}: {len(session.messages)} messages")
                else:
                    failed += 1
                    logger.warning(f"Failed to parse {self.agent_id} session {session_id}: {session.error}")
            except Exception as e:
                failed += 1
                logger.error(f"Error parsing {self.agent_id} session {session_id}: {e}")

        duration_ms = (time.time() - start_time) * 1000

        return ParseResult(
            agent=self.agent_id,
            sessions=results,
            total_sessions=len(sessions),
            successful=successful,
            failed=failed,
            duration_ms=duration_ms,
        )

    def iter_sessions(self) -> Iterator[ParsedSession]:
        """Iterate over parsed sessions.

        Yields:
            ParsedSession objects
        """
        for session_id in self.list_sessions():
            yield self.parse_session(session_id)


class CopilotParser(AgentParser):
    """Parser for GitHub Copilot chat history."""

    @property
    def agent_id(self) -> str:
        return AGENT_COPILOT

    def get_storage_path(self) -> Optional[Path]:
        """Get Copilot storage path."""
        storage = locate_vscode_storage()
        workspace_storage = storage.get("workspaceStorage")

        if not workspace_storage or not workspace_storage.exists():
            return None

        # Look for Copilot extension storage
        # Copilot stores in workspaceStorage/<hash>/github.copilot
        for workspace in workspace_storage.iterdir():
            copilot_path = workspace / "github.copilot"
            if copilot_path.exists():
                return copilot_path

        return None

    def list_sessions(self) -> List[str]:
        """List Copilot chat sessions via state.vdb index."""
        storage_path = self.get_storage_path()
        if not storage_path:
            logger.warning("Copilot storage path not found")
            return []

        # Look for state.vdb which contains the session index
        # VS Code uses LevelDB for state.vdb, but we can look for IndexedDB
        # In practice, the sessions are in chatSessions folder
        sessions_dir = storage_path / COPILOT_CHAT_SESSIONS_DIR
        if not sessions_dir.exists():
            logger.warning(f"Copilot sessions directory not found: {sessions_dir}")
            return []

        session_files = sorted(sessions_dir.glob("session-*.json"))
        session_ids = [f.stem.replace("session-", "") for f in session_files]

        logger.debug(f"Found {len(session_ids)} Copilot sessions")
        return session_ids

    def parse_session(self, session_id: str) -> ParsedSession:
        """Parse a Copilot session file."""
        storage_path = self.get_storage_path()
        if not storage_path:
            return ParsedSession(
                session_id=session_id,
                agent=self.agent_id,
                messages=[],
                success=False,
                error="Storage path not found",
            )

        session_file = storage_path / COPILOT_CHAT_SESSIONS_DIR / f"session-{session_id}.json"
        if not session_file.exists():
            return ParsedSession(
                session_id=session_id,
                agent=self.agent_id,
                messages=[],
                success=False,
                error=f"Session file not found: {session_file}",
            )

        data = parse_json_file(session_file)
        if data is None:
            return ParsedSession(
                session_id=session_id,
                agent=self.agent_id,
                messages=[],
                success=False,
                error="Failed to parse session JSON",
            )

        messages: List[ChatMessage] = []
        created_at = None
        updated_at = None

        # Extract timestamps if available
        if "createdAt" in data:
            created_at = self._parse_timestamp(data["createdAt"])
        if "updatedAt" in data:
            updated_at = self._parse_timestamp(data["updatedAt"])

        # Map requests to messages
        requests = data.get("requests", [])
        for req in requests:
            # User message
            if "message" in req:
                user_content = req.get("message", {})
                if isinstance(user_content, dict):
                    text = user_content.get("text", "")
                else:
                    text = str(user_content)

                messages.append(ChatMessage(
                    role="user",
                    content=text,
                    raw=req,
                ))

            # Assistant response
            if "response" in req:
                resp = req.get("response", {})
                if isinstance(resp, dict):
                    text = resp.get("message", {}).get("content", "") if isinstance(resp.get("message"), dict) else str(resp)
                else:
                    text = str(resp)

                messages.append(ChatMessage(
                    role="assistant",
                    content=text,
                    model=resp.get("model") if isinstance(resp, dict) else None,
                    raw=req,
                ))

        return ParsedSession(
            session_id=session_id,
            agent=self.agent_id,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            success=True,
        )

    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts)
        if isinstance(ts, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(ts, fmt)
                except ValueError:
                    continue
        return None


class ClineParser(AgentParser):
    """Parser for Cline (Claude Dev) chat history."""

    @property
    def agent_id(self) -> str:
        return AGENT_CLINE

    def get_storage_path(self) -> Optional[Path]:
        """Get Cline storage path."""
        return get_extension_storage(CLINE_EXTENSION_ID)

    def list_sessions(self) -> List[str]:
        """List Cline task sessions."""
        storage_path = self.get_storage_path()
        if not storage_path:
            logger.warning("Cline storage path not found")
            return []

        tasks_dir = storage_path / CLINE_API_CONVERSATION_DIR
        if not tasks_dir.exists():
            logger.warning(f"Cline tasks directory not found: {tasks_dir}")
            return []

        session_dirs = [d.name for d in tasks_dir.iterdir() if d.is_dir()]
        logger.debug(f"Found {len(session_dirs)} Cline sessions")
        return sorted(session_dirs)

    def parse_session(self, session_id: str) -> ParsedSession:
        """Parse a Cline session."""
        storage_path = self.get_storage_path()
        if not storage_path:
            return ParsedSession(
                session_id=session_id,
                agent=self.agent_id,
                messages=[],
                success=False,
                error="Storage path not found",
            )

        # Try primary source: api_conversation_history.json
        primary_file = (
            storage_path / CLINE_API_CONVERSATION_DIR / session_id / CLINE_API_CONVERSATION_FILE
        )
        if primary_file.exists():
            return self._parse_from_api_history(primary_file, session_id)

        # Fallback: state.json
        fallback_file = storage_path / CLINE_API_CONVERSATION_DIR / session_id / CLINE_FALLBACK_FILE
        if fallback_file.exists():
            return self._parse_from_state(fallback_file, session_id)

        return ParsedSession(
            session_id=session_id,
            agent=self.agent_id,
            messages=[],
            success=False,
            error="No conversation history file found",
        )

    def _parse_from_api_history(self, file_path: Path, session_id: str) -> ParsedSession:
        """Parse from api_conversation_history.json (exact API payload)."""
        data = parse_json_file(file_path)
        if data is None:
            return ParsedSession(
                session_id=session_id,
                agent=self.agent_id,
                messages=[],
                success=False,
                error="Failed to parse API history",
            )

        messages: List[ChatMessage] = []

        # API payload format: array of messages with role/content
        # Also contains tools and tool_calls
        for msg in data:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Handle content as string or array of blocks
            if isinstance(content, list):
                content = self._parse_content_blocks(content)

            chat_msg = ChatMessage(
                role=role,
                content=content,
                tool_calls=msg.get("tool_calls"),
                raw=msg,
            )

            # Extract model if present
            if "model" in msg:
                chat_msg.model = msg["model"]

            messages.append(chat_msg)

        # Handle tool results (often in a separate field)
        if "tool_results" in data:
            for tool_result in data["tool_results"]:
                messages.append(ChatMessage(
                    role="tool",
                    content=tool_result.get("content", ""),
                    raw=tool_result,
                ))

        return ParsedSession(
            session_id=session_id,
            agent=self.agent_id,
            messages=messages,
            success=True,
        )

    def _parse_content_blocks(self, blocks: List[Dict[str, Any]]) -> str:
        """Parse content blocks to text."""
        text_parts = []
        for block in blocks:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    text_parts.append(f"[Tool: {block.get('name', 'unknown')}]")
                elif block_type == "tool_result":
                    text_parts.append(f"[Tool Result: {block.get('content', '')}]")
        return "\n".join(text_parts)

    def _parse_from_state(self, file_path: Path, session_id: str) -> ParsedSession:
        """Parse from state.json (fallback)."""
        data = parse_json_file(file_path)
        if data is None:
            return ParsedSession(
                session_id=session_id,
                agent=self.agent_id,
                messages=[],
                success=False,
                error="Failed to parse state file",
            )

        messages: List[ChatMessage] = []
        history = data.get("history", [])

        for item in history:
            role = item.get("role", "")
            content = item.get("content", "")

            if isinstance(content, list):
                content = self._parse_content_blocks(content)

            messages.append(ChatMessage(
                role=role,
                content=content,
                raw=item,
            ))

        return ParsedSession(
            session_id=session_id,
            agent=self.agent_id,
            messages=messages,
            success=True,
        )


class ContinueParser(AgentParser):
    """Parser for Continue chat history."""

    @property
    def agent_id(self) -> str:
        return AGENT_CONTINUE

    def get_storage_path(self) -> Optional[Path]:
        """Get Continue config directory."""
        # Continue stores in ~/.continue
        continue_dir = Path.home() / CONTINUE_CONFIG_DIR
        if continue_dir.exists():
            return continue_dir
        return None

    def list_sessions(self) -> List[str]:
        """List Continue sessions."""
        storage_path = self.get_storage_path()
        if not storage_path:
            logger.warning("Continue storage path not found")
            return []

        sessions: List[str] = []

        # Try sessions.json first
        sessions_file = storage_path / CONTINUE_SESSIONS_FILE
        if sessions_file.exists():
            data = parse_json_file(sessions_file)
            if data and isinstance(data, list):
                # sessions.json is an array of sessions
                sessions = [str(i) for i in range(len(data))]
                logger.debug(f"Found {len(sessions)} sessions in sessions.json")
                return sessions

        # Try history.db (SQLite)
        history_db = storage_path / CONTINUE_HISTORY_DB
        if history_db.exists():
            try:
                conn = sqlite3.connect(history_db)
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM sessions ORDER BY updated_at DESC")
                rows = cursor.fetchall()
                conn.close()
                sessions = [str(row[0]) for row in rows]
                logger.debug(f"Found {len(sessions)} sessions in history.db")
                return sessions
            except Exception as e:
                logger.warning(f"Error reading history.db: {e}")

        logger.warning("No Continue session files found")
        return sessions

    def parse_session(self, session_id: str) -> ParsedSession:
        """Parse a Continue session."""
        storage_path = self.get_storage_path()
        if not storage_path:
            return ParsedSession(
                session_id=session_id,
                agent=self.agent_id,
                messages=[],
                success=False,
                error="Storage path not found",
            )

        idx = int(session_id)

        # Try sessions.json
        sessions_file = storage_path / CONTINUE_SESSIONS_FILE
        if sessions_file.exists():
            data = parse_json_file(sessions_file)
            if data and isinstance(data, list) and idx < len(data):
                return self._parse_from_sessions_json(data[idx], session_id)

        # Try history.db
        history_db = storage_path / CONTINUE_HISTORY_DB
        if history_db.exists():
            return self._parse_from_db(history_db, session_id)

        return ParsedSession(
            session_id=session_id,
            agent=self.agent_id,
            messages=[],
            success=False,
            error="Session not found",
        )

    def _parse_from_sessions_json(self, session_data: Any, session_id: str) -> ParsedSession:
        """Parse session from sessions.json format."""
        messages: List[ChatMessage] = []

        if isinstance(session_data, dict):
            # Messages might be in 'messages' or 'history' field
            msg_list = session_data.get("messages", session_data.get("history", []))
        elif isinstance(session_data, list):
            msg_list = session_data
        else:
            msg_list = []

        for msg in msg_list:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, dict):
                # Might have different structure
                content = content.get("text", content.get("content", str(content)))

            messages.append(ChatMessage(
                role=role,
                content=str(content),
                raw=msg,
            ))

        created_at = None
        updated_at = None

        if isinstance(session_data, dict):
            if "created_at" in session_data:
                created_at = self._parse_timestamp(session_data["created_at"])
            if "updated_at" in session_data:
                updated_at = self._parse_timestamp(session_data["updated_at"])

        return ParsedSession(
            session_id=session_id,
            agent=self.agent_id,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            success=True,
        )

    def _parse_from_db(self, db_path: Path, session_id: str) -> ParsedSession:
        """Parse session from SQLite history.db."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Get session info
            cursor.execute(
                "SELECT created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,)
            )
            row = cursor.fetchone()

            created_at = None
            updated_at = None

            if row:
                if row[0]:
                    created_at = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                if row[1]:
                    updated_at = datetime.fromisoformat(row[1].replace("Z", "+00:00"))

            # Get messages for this session
            cursor.execute(
                "SELECT role, content, model FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,)
            )
            message_rows = cursor.fetchall()
            conn.close()

            messages = [
                ChatMessage(
                    role=row[0],
                    content=row[1],
                    model=row[2] if len(row) > 2 else None,
                )
                for row in message_rows
            ]

            return ParsedSession(
                session_id=session_id,
                agent=self.agent_id,
                messages=messages,
                created_at=created_at,
                updated_at=updated_at,
                success=True,
            )

        except Exception as e:
            return ParsedSession(
                session_id=session_id,
                agent=self.agent_id,
                messages=[],
                success=False,
                error=f"Database error: {e}",
            )

    def _parse_timestamp(self, ts: Any) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        if isinstance(ts, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
                try:
                    return datetime.strptime(ts, fmt)
                except ValueError:
                    continue
        return None


# Registry of available parsers
PARSER_REGISTRY: Dict[str, type] = {
    AGENT_COPILOT: CopilotParser,
    AGENT_CLINE: ClineParser,
    AGENT_CONTINUE: ContinueParser,
}


def get_parser(agent: str, workspace_path: Optional[Path] = None) -> Optional[AgentParser]:
    """Get parser instance for specified agent.

    Args:
        agent: Agent identifier (copilot, cline, continue)
        workspace_path: Optional workspace path for workspace-specific storage

    Returns:
        AgentParser instance or None if agent not supported
    """
    parser_class = PARSER_REGISTRY.get(agent.lower())
    if parser_class:
        return parser_class(workspace_path=workspace_path)
    return None


def parse_agent(
    agent: str,
    workspace_path: Optional[Path] = None,
    session_id: Optional[str] = None,
) -> ParseResult | ParsedSession:
    """Parse chat history for specified agent.

    Args:
        agent: Agent identifier (copilot, cline, continue)
        workspace_path: Optional workspace path
        session_id: Optional specific session to parse (if None, parse all)

    Returns:
        ParseResult for all sessions or ParsedSession for single session
    """
    parser = get_parser(agent, workspace_path=workspace_path)
    if not parser:
        raise ValueError(f"Unsupported agent: {agent}. Supported: {list(PARSER_REGISTRY.keys())}")

    if session_id:
        return parser.parse_session(session_id)

    return parser.parse_all()
