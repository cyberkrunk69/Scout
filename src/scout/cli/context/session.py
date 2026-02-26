#!/usr/bin/env python
"""
Session management - persist session state.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def _repo_root() -> Path:
    """Get repository root."""
    return Path.cwd().resolve()


def _scout_dir() -> Path:
    """Get .scout directory for session data."""
    scout_dir = _repo_root() / ".scout"
    scout_dir.mkdir(exist_ok=True)
    return scout_dir


@dataclass
class Session:
    """Scout CLI session."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    messages: list[dict] = field(default_factory=list)
    total_cost: float = 0.0
    total_tokens: int = 0
    last_plan_id: Optional[str] = None
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert session to dictionary."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "messages": self.messages,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "last_plan_id": self.last_plan_id,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create session from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            messages=data.get("messages", []),
            total_cost=data.get("total_cost", 0.0),
            total_tokens=data.get("total_tokens", 0),
            last_plan_id=data.get("last_plan_id"),
            config=data.get("config", {}),
        )

    def add_message(self, role: str, content: str):
        """Add a message to the session."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })

    def add_cost(self, cost: float, tokens: int = 0):
        """Add cost and token usage."""
        self.total_cost += cost
        self.total_tokens += tokens


def load_session(path: Optional[Path] = None) -> Session:
    """
    Load session from file.

    Args:
        path: Path to session file (default: .scout/session.json)

    Returns:
        Session object
    """
    if path is None:
        path = _scout_dir() / "session.json"

    if not path.exists():
        return Session()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Session.from_dict(data)
    except (json.JSONDecodeError, OSError):
        return Session()


def save_session(session: Session, path: Optional[Path] = None):
    """
    Save session to file.

    Args:
        session: Session to save
        path: Path to save to (default: .scout/session.json)
    """
    if path is None:
        path = _scout_dir() / "session.json"

    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(session.to_dict(), indent=2),
        encoding="utf-8"
    )


def clear_session(path: Optional[Path] = None):
    """
    Clear session file.

    Args:
        path: Path to session file (default: .scout/session.json)
    """
    if path is None:
        path = _scout_dir() / "session.json"

    if path.exists():
        path.unlink()


def export_session(session: Session, export_path: Path):
    """
    Export session to a file.

    Args:
        session: Session to export
        export_path: Path to export to
    """
    export_path.parent.mkdir(exist_ok=True)
    export_path.write_text(
        json.dumps(session.to_dict(), indent=2),
        encoding="utf-8"
    )


def get_session_history(path: Optional[Path] = None, limit: int = 10) -> list[dict]:
    """
    Get recent session history.

    Args:
        path: Path to session file (default: .scout/session.json)
        limit: Number of recent messages to return

    Returns:
        List of recent messages
    """
    session = load_session(path)
    return session.messages[-limit:] if session.messages else []


class SessionManager:
    """Manage multiple sessions."""

    def __init__(self, sessions_dir: Optional[Path] = None):
        self.sessions_dir = sessions_dir or (_scout_dir() / "sessions")
        self.sessions_dir.mkdir(exist_ok=True)

    def list_sessions(self) -> list[Session]:
        """List all sessions."""
        sessions = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                sessions.append(Session.from_dict(data))
            except (json.JSONDecodeError, OSError):
                continue
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        path = self.sessions_dir / f"{session_id}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Session.from_dict(data)
        except (json.JSONDecodeError, OSError):
            return None

    def save_session(self, session: Session):
        """Save a session to the sessions directory."""
        path = self.sessions_dir / f"{session.id}.json"
        path.write_text(
            json.dumps(session.to_dict(), indent=2),
            encoding="utf-8"
        )


if __name__ == "__main__":
    # Test session management
    session = Session()
    session.add_message("user", "Hello")
    session.add_message("assistant", "Hi there!")
    session.add_cost(0.05, 100)
    print(f"Session: {session.id}")
    print(f"Messages: {len(session.messages)}")
    save_session(session)
    loaded = load_session()
    print(f"Loaded messages: {len(loaded.messages)}")
