"""Unified schema for parsed agent conversation logs.

This module defines the Pydantic models that all parser outputs must conform to.
The schema includes a _version field to allow future evolution.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"


class Message(BaseModel):
    """A single message in a conversation."""

    role: str = Field(..., description="Role: user, assistant, system, tool")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="ISO timestamp")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool calls if any"
    )
    tool_results: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tool results if any"
    )
    model: Optional[str] = Field(None, description="Model used for this message")
    tokens: Optional[int] = Field(None, description="Token count if available")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"user", "assistant", "system", "tool", "tool_result"}
        if v not in allowed:
            logger.warning("Unknown role %s, allowing anyway", v)
        return v


class SessionMeta(BaseModel):
    """Metadata about the parsed session."""

    source_agent: str = Field(..., description="Source agent name (e.g., cursor)")
    session_id: str = Field(..., description="Unique session identifier")
    project_hash: Optional[str] = Field(
        None, description="Hash of project path for anonymity"
    )
    start_time: Optional[str] = Field(None, description="Session start ISO timestamp")
    end_time: Optional[str] = Field(None, description="Session end ISO timestamp")
    project_path: Optional[str] = Field(None, description="Normalized project path")


class ParsedSession(BaseModel):
    """Unified schema for parsed agent conversation sessions.

    This is the canonical output format for all parsers. Each parser
    must transform agent-specific data into this format.
    """

    meta: SessionMeta = Field(..., description="Session metadata")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    schema_version: str = Field(
        default=SCHEMA_VERSION,
        description="Schema version for future evolution"
    )

    # Optional extended data for agent-specific fields
    raw: Optional[Dict[str, Any]] = Field(
        None, description="Raw agent-specific data for debugging"
    )

    def model_dump(self, **kwargs: Any) -> Dict[str, Any]:
        """Serialize to dict, handling the schema_version field correctly."""
        data = super().model_dump(**kwargs)
        # Ensure version is at top level for consistency
        data["version"] = data.pop("schema_version", SCHEMA_VERSION)
        return data

    def to_audit_dict(self) -> Dict[str, Any]:
        """Convert to dict for audit logging."""
        return {
            "source_agent": self.meta.source_agent,
            "session_id": self.meta.session_id,
            "message_count": len(self.messages),
            "start_time": self.meta.start_time,
            "end_time": self.meta.end_time,
            "project_path": self.meta.project_path,
            "version": self.schema_version,
        }


def compute_project_hash(project_path: Optional[str]) -> Optional[str]:
    """Compute a hash of the project path for anonymity.

    Args:
        project_path: The normalized project path

    Returns:
        Truncated SHA256 hash or None
    """
    if not project_path:
        return None
    h = hashlib.sha256(project_path.encode()).hexdigest()
    return h[:12]


class SchemaMismatchError(Exception):
    """Raised when parsed data doesn't match expected schema."""

    def __init__(self, message: str, raw_data: Optional[Any] = None):
        super().__init__(message)
        self.raw_data = raw_data


def validate_and_log(
    data: Dict[str, Any],
    session_id: str,
    log_raw: bool = True
) -> ParsedSession:
    """Validate parsed data against schema and log mismatches.

    Args:
        data: Parsed data dictionary
        session_id: Session identifier for logging
        log_raw: Whether to include raw data in error log

    Returns:
        Validated ParsedSession

    Raises:
        SchemaMismatchError: If validation fails
    """
    try:
        return ParsedSession(**data)
    except Exception as e:
        if log_raw:
            logger.error(
                "Schema mismatch for session %s: %s. Raw data: %s",
                session_id,
                e,
                str(data)[:500] if log_raw else "<redacted>"
            )
        else:
            logger.error("Schema mismatch for session %s: %s", session_id, e)
        raise SchemaMismatchError(str(e), data if log_raw else None)
