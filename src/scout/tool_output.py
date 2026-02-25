"""
ToolOutput - Standard data structure for Scout tool outputs.

This module provides:
- ToolOutput dataclass: standardized output representation
- ToolOutputRegistry: file-based storage for outputs

Phase 1 of Unified Tool Framework.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ToolOutput:
    """Standard output representation for Scout tools.

    Fields:
        tool_name: Name of the tool that produced this output
        content: The actual output content (any serializable type)
        cost_usd: Cost of generating this output in USD
        confidence: Optional confidence score (0.0-1.0)
        validation_errors: List of validation errors (populated later by validation pipeline)
        metadata: Additional metadata (input query, flags, etc.)
        dependencies: List of output_ids this output depends on
        output_id: Unique identifier (UUID)
        created_at: Timestamp of creation
    """

    tool_name: str
    content: Any
    cost_usd: float
    confidence: Optional[float] = None
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    output_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        result = asdict(self)
        # Convert datetime to ISO string
        result["created_at"] = self.created_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolOutput:
        """Deserialize from dictionary."""
        # Handle datetime parsing
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)

    @classmethod
    def from_content(
        cls,
        tool_name: str,
        content: Any,
        cost_usd: float = 0.0,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToolOutput":
        """Convenience constructor for tool migration.

        Args:
            tool_name: Name of the tool
            content: The output content
            cost_usd: Cost in USD (default 0.0)
            confidence: Optional confidence score
            metadata: Optional metadata dict

        Returns:
            ToolOutput instance
        """
        return cls(
            tool_name=tool_name,
            content=content,
            cost_usd=cost_usd,
            confidence=confidence,
            metadata=metadata or {},
        )


class ToolOutputRegistry:
    """File-based registry for ToolOutput storage.

    Storage location:
    - If inside a git repo: .scout/tool_outputs/ in repo root
    - Otherwise: ~/.scout/tool_outputs/

    Uses atomic writes (temp file + os.replace) for concurrent safety.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """Initialize registry with base path.

        Args:
            base_path: Custom base path. If None, auto-detect from git repo or home.
        """
        if base_path is not None:
            self._base_path = base_path
        else:
            self._base_path = self._detect_base_path()

        self._outputs_dir = self._base_path / "tool_outputs"

    @staticmethod
    def _detect_base_path() -> Path:
        """Detect appropriate base path for tool outputs.

        Priority:
        1. Current working directory if it's inside a git repo
        2. ~/.scout/ as fallback
        """
        cwd = Path.cwd()

        # Check if we're inside a git repo
        if cwd.is_dir():
            # Walk up looking for .git
            current = cwd
            for _ in range(20):  # Max 20 levels up
                if (current / ".git").exists():
                    return current / ".scout"
                parent = current.parent
                if parent == current:
                    break
                current = parent

        # Fallback to user home
        return Path.home() / ".scout"

    @property
    def outputs_dir(self) -> Path:
        """Get the outputs directory, creating if needed."""
        if not self._outputs_dir.exists():
            self._outputs_dir.mkdir(parents=True, exist_ok=True)
        return self._outputs_dir

    def save(self, output: ToolOutput) -> None:
        """Save a ToolOutput to the registry.

        Uses atomic write (temp file + os.replace) for concurrent safety.

        Args:
            output: The ToolOutput to save
        """
        output_file = self.outputs_dir / f"{output.output_id}.json"

        # Write to temp file first, then atomic rename
        temp_file = self.outputs_dir / f".{output.output_id}.json.tmp"
        temp_file.write_text(
            json.dumps(output.to_dict(), indent=2, default=str),
            encoding="utf-8"
        )

        # Atomic rename (works on POSIX, graceful fail on Windows)
        try:
            os.replace(temp_file, output_file)
        except OSError:
            # Windows fallback - just move the temp file
            if output_file.exists():
                output_file.unlink()
            temp_file.rename(output_file)

    def load(self, output_id: str) -> Optional[ToolOutput]:
        """Load a ToolOutput by ID.

        Args:
            output_id: The output ID to load

        Returns:
            ToolOutput if found, None otherwise
        """
        output_file = self.outputs_dir / f"{output_id}.json"

        if not output_file.exists():
            return None

        try:
            data = json.loads(output_file.read_text(encoding="utf-8"))
            return ToolOutput.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def list(self) -> List[str]:
        """List all stored output IDs.

        Returns:
            List of output IDs (filenames without .json)
        """
        if not self.outputs_dir.exists():
            return []

        output_ids = []
        for f in self.outputs_dir.iterdir():
            if f.is_file() and f.suffix == ".json" and not f.name.startswith("."):
                output_ids.append(f.stem)

        return sorted(output_ids)

    def delete(self, output_id: str) -> bool:
        """Delete an output by ID.

        Args:
            output_id: The output ID to delete

        Returns:
            True if deleted, False if not found
        """
        output_file = self.outputs_dir / f"{output_id}.json"

        if not output_file.exists():
            return False

        output_file.unlink()
        return True


# Module-level convenience functions

_default_registry: Optional[ToolOutputRegistry] = None


def get_registry() -> ToolOutputRegistry:
    """Get the default registry instance."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolOutputRegistry()
    return _default_registry


def save_output(output: ToolOutput) -> None:
    """Convenience function to save to default registry."""
    get_registry().save(output)


def load_output(output_id: str) -> Optional[ToolOutput]:
    """Convenience function to load from default registry."""
    return get_registry().load(output_id)


def list_outputs() -> List[str]:
    """Convenience function to list all outputs."""
    return get_registry().list()
