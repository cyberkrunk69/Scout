"""Progress reporting stubs for scout-core.

This is a stub implementation to allow batch_pipeline to import.
The full progress system should be extracted from Vivarium in a future track.
"""

from enum import Enum
from typing import Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from datetime import datetime


class Status(Enum):
    """Status enum for progress reporting."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    CIRCUIT_OPEN = "circuit_open"
    COMPLETE = "complete"


@dataclass
class ProgressEvent:
    """Progress event for reporting."""
    status: Status
    message: str = ""
    details: Optional[dict] = None
    timestamp: Optional[datetime] = None
    task_id: str = ""  # Optional task identifier
    metadata: Optional[dict] = None  # Additional metadata
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.details is None:
            self.details = {}
        if self.metadata is None:
            self.metadata = {}


def format_deterministic(value: Any) -> str:
    """Format value for deterministic output."""
    return str(value)


class ProgressReporter:
    """Stub progress reporter for scout-core."""
    
    def __init__(self):
        self._events: list[ProgressEvent] = []
    
    async def emit_async(self, event: ProgressEvent) -> None:
        """Emit a progress event."""
        self._events.append(event)
    
    def emit(self, event: ProgressEvent) -> None:
        """Emit a progress event synchronously."""
        self._events.append(event)
    
    def get_events(self) -> list[ProgressEvent]:
        """Get all events."""
        return self._events
    
    def get_history(self) -> list[ProgressEvent]:
        """Get all events (alias for get_events)."""
        return self._events
    
    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()
