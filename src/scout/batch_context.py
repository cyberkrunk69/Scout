# TODO: This module is not yet integrated into the main application.
# It is planned for future use as part of the Batch Pipeline framework.
# See ADR-008 for design context.
from __future__ import annotations
"""BatchContext: Thread-safe state container for pipeline execution."""

import asyncio
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class TaskResult:
    """Result of a single task in a batch pipeline.

    Attributes:
        task_index: Index of the task in the batch sequence.
        command: The command or action that was executed.
        status: Execution status (e.g., 'success', 'error', 'skipped').
        output: The output produced by the task execution.
        error: Error message if the task failed, None otherwise.
    """
    task_index: int
    command: str
    status: str
    output: Any
    error: Optional[str] = None


class BatchContext:
    """Thread-safe state container for batch pipeline execution."""

    def __init__(self, initial_vars: Optional[dict] = None):
        """Initialize the batch context with optional initial variables.

        Args:
            initial_vars: Optional dictionary of initial variables to set.
        """
        self._variables = initial_vars or {}
        self._results = {}
        self._early_exit = None
        self._lock = asyncio.Lock()

    def set_var(self, name: str, value: Any) -> None:
        """Set a variable in the batch context.

        Args:
            name: Variable name (supports dot notation for nested access).
            value: Value to store.
        """
        self._variables[name] = value

    def get_var(self, name: str, default: Any = None) -> Any:
        """Get a variable from the batch context.

        Supports dot notation for nested access (e.g., 'task.output').

        Args:
            name: Variable name (dot notation supported).
            default: Default value if variable not found.

        Returns:
            The variable value or default if not found.
        """
        parts = name.split(".")
        current = self._variables
        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, TaskResult):
                current = getattr(current, part, None)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
            if current is None:
                return default
        return current

    def get_all_vars(self) -> dict:
        """Get a copy of all variables in the context.

        Returns:
            Dictionary of all variables.
        """
        return self._variables.copy()

    async def set_result(
        self,
        task_index: int,
        command: str,
        status: str,
        output: Any,
        error: Optional[str] = None,
    ) -> None:
        """Store the result of a completed task.

        Args:
            task_index: Index of the task.
            command: The command that was executed.
            status: Execution status.
            output: Task output.
            error: Error message if failed, None otherwise.
        """
        async with self._lock:
            self._results[task_index] = TaskResult(
                task_index, command, status, output, error
            )

    async def get_all_results(self) -> list:
        """Get all task results in order.

        Returns:
            List of TaskResult objects sorted by task index.
        """
        async with self._lock:
            return [self._results[i] for i in sorted(self._results.keys())]

    async def set_early_exit(self, index: int, reason: str) -> None:
        """Signal early exit from the batch pipeline.

        Args:
            index: Task index that triggered early exit.
            reason: Reason for early exit.
        """
        async with self._lock:
            self._early_exit = {"index": index, "reason": reason}

    async def check_early_exit(self) -> Optional[dict]:
        """Check if early exit was requested.

        Returns:
            Dict with 'index' and 'reason' if early exit requested, None otherwise.
        """
        async with self._lock:
            return self._early_exit.copy() if self._early_exit else None

    def to_dict(self) -> dict:
        """Convert context to dictionary for serialization.

        Returns:
            Dictionary representation of the context.
        """
        return {
            "variables": self._variables,
            "result_count": len(self._results),
            "early_exit": self._early_exit,
        }
