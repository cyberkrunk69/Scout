from __future__ import annotations
"""BatchContext: Thread-safe state container for pipeline execution."""

import asyncio
from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class TaskResult:
    task_index: int
    command: str
    status: str
    output: Any
    error: Optional[str] = None


class BatchContext:
    """Thread-safe state container for batch pipeline execution."""

    def __init__(self, initial_vars: Optional[dict] = None):
        self._variables = initial_vars or {}
        self._results = {}
        self._early_exit = None
        self._lock = asyncio.Lock()

    def set_var(self, name: str, value: Any) -> None:
        self._variables[name] = value

    def get_var(self, name: str, default: Any = None) -> Any:
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
        return self._variables.copy()

    async def set_result(
        self,
        task_index: int,
        command: str,
        status: str,
        output: Any,
        error: Optional[str] = None,
    ) -> None:
        async with self._lock:
            self._results[task_index] = TaskResult(
                task_index, command, status, output, error
            )

    async def get_all_results(self) -> list:
        async with self._lock:
            return [self._results[i] for i in sorted(self._results.keys())]

    async def set_early_exit(self, index: int, reason: str) -> None:
        async with self._lock:
            self._early_exit = {"index": index, "reason": reason}

    async def check_early_exit(self) -> Optional[dict]:
        async with self._lock:
            return self._early_exit.copy() if self._early_exit else None

    def to_dict(self) -> dict:
        return {
            "variables": self._variables,
            "result_count": len(self._results),
            "early_exit": self._early_exit,
        }
