"""Data classes for doc_generation package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from scout.adapters.base import SymbolTree


class BudgetExceededError(RuntimeError):
    """Raised when doc-sync exceeds the --budget limit."""

    def __init__(self, total_cost: float, budget: float) -> None:
        super().__init__(f"Budget exceeded: ${total_cost:.4f} >= ${budget}")
        self.total_cost = total_cost
        self.budget = budget


@dataclass
class FileProcessResult:
    """Result of processing a single file for doc generation."""

    success: bool
    cost_usd: float
    symbols_count: int
    calls_count: int
    types_count: int
    exports_count: int
    model: str
    skipped: bool = False  # True when skipped due to freshness (up to date)
    error: Optional[str] = None
    call_chain: Optional[str] = None  # e.g. "funcA → pkgB.funcB → pkgC.funcC"


@dataclass
class TraceResult:
    """Result of pure static analysis (no LLM)."""

    root_tree: SymbolTree
    symbols_to_doc: List[SymbolTree]
    all_calls: set
    all_types: set
    all_exports: set
    adapter: Any
    dependencies: List[str]
