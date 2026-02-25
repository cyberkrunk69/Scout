"""Query interface for code relationship graphs.

Provides functions to query call graphs, find usages, and perform
impact analysis using relationship data extracted from AST.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Any


logger = logging.getLogger(__name__)


@dataclass
class CallerInfo:
    """Information about a function that calls another."""
    symbol: str  # "path::function"
    file: Path
    line: int
    call_type: str = "direct"  # "direct", "dynamic", "method"
    ambiguous: bool = False  # True if symbol was ambiguous


@dataclass
class CalleeInfo:
    """Information about a function being called."""
    symbol: str
    file: Path
    line: int
    call_type: str = "direct"
    ambiguous: bool = False


@dataclass
class UsageInfo:
    """Information about symbol usage."""
    symbol: str
    file: Path
    line: int
    usage_type: str  # "call", "reference", "assignment", "import"


@dataclass
class PathStep:
    """Single step in a call path."""
    from_symbol: str
    to_symbol: str
    edge_type: str


# === Graph Cache ===

class GraphCache:
    """In-memory cache for aggregated graphs."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self._cache: Dict[str, Any] = {}
        self._cache_path = self.repo_root / ".scout" / "code_relationships.json"

    def load(self) -> Optional[Dict[str, Any]]:
        """Load cached graph if exists and fresh. Returns None if missing."""
        if not self._cache_path.exists():
            logger.warning("Graph cache not found at %s (run 'scout doc sync' to generate)", self._cache_path)
            return None
        try:
            data = json.loads(self._cache_path.read_text())
            return data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load graph cache: %s", e)
            return None

    def save(self, graph: Dict[str, Any]) -> None:
        """Cache graph and persist to disk."""
        # Ensure .scout directory exists at repo root
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(graph, indent=2))

    def invalidate(self, changed_files: List[Path]) -> None:
        """Invalidate cache for affected scopes."""
        # For now, just clear - could be smarter later
        if self._cache_path.exists():
            self._cache_path.unlink()


# === Symbol Resolution ===

def resolve_symbol(name: str, repo_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Resolve a simple symbol name to its index entry.

    Returns:
        {"options": [...], "ambiguous": bool} if found
        None if not found or graph doesn't exist
    """
    repo_root = repo_root or Path.cwd()
    cache = GraphCache(repo_root)
    graph = cache.load()

    if not graph:
        return None

    symbol_index = graph.get("symbolIndex", {})
    return symbol_index.get(name)


# === Query Functions ===

def get_callers(symbol: str, repo_root: Optional[Path] = None) -> List[CallerInfo]:
    """Find all functions that call the given symbol.

    If symbol is a simple name (no ::), uses symbolIndex to resolve ambiguity.
    Returns all matches marked with ambiguous=True if multiple options exist.
    """
    repo_root = repo_root or Path.cwd()
    cache = GraphCache(repo_root)
    graph = cache.load()

    if not graph:
        logger.warning("No graph available for get_callers. Run 'scout doc sync' first.")
        return []

    # Handle simple name resolution via symbolIndex
    resolved_symbol = symbol
    ambiguous = False
    if "::" not in symbol:
        symbol_index = graph.get("symbolIndex", {})
        index_entry = symbol_index.get(symbol)
        if index_entry:
            if index_entry.get("ambiguous"):
                # Return all possibilities with ambiguity flag
                callers = []
                for option in index_entry.get("options", []):
                    node_data = graph.get("nodes", {}).get(option, {})
                    callers.append(CallerInfo(
                        symbol=option,
                        file=repo_root / node_data.get("file", ""),
                        line=node_data.get("line", 0),
                        call_type="direct",
                        ambiguous=True,
                    ))
                return callers
            elif index_entry.get("options"):
                # Single match, use it
                resolved_symbol = index_entry["options"][0]

    callers = []
    edges = graph.get("edges", [])

    for edge in edges:
        if edge.get("type") == "calls" and edge.get("to") == resolved_symbol:
            from_key = edge.get("from", "")
            node_data = graph.get("nodes", {}).get(from_key, {})
            callers.append(CallerInfo(
                symbol=from_key,
                file=repo_root / node_data.get("file", ""),
                line=node_data.get("line", 0),
                call_type="direct",
                ambiguous=ambiguous,
            ))

    return callers


def get_callees(symbol: str, repo_root: Optional[Path] = None) -> List[CalleeInfo]:
    """Find all functions called by the given symbol.

    If symbol is a simple name (no ::), uses symbolIndex to resolve ambiguity.
    """
    repo_root = repo_root or Path.cwd()
    cache = GraphCache(repo_root)
    graph = cache.load()

    if not graph:
        logger.warning("No graph available for get_callees. Run 'scout doc sync' first.")
        return []

    # Handle simple name resolution via symbolIndex
    resolved_symbol = symbol
    if "::" not in symbol:
        symbol_index = graph.get("symbolIndex", {})
        index_entry = symbol_index.get(symbol)
        if index_entry and index_entry.get("options"):
            if index_entry.get("ambiguous"):
                # For ambiguous symbols, return callees for all options
                all_callees = []
                for option in index_entry.get("options", []):
                    callees = _get_callees_for_symbol(option, graph, repo_root)
                    for c in callees:
                        c.ambiguous = True
                    all_callees.extend(callees)
                return all_callees
            resolved_symbol = index_entry["options"][0]

    return _get_callees_for_symbol(resolved_symbol, graph, repo_root)


def _get_callees_for_symbol(symbol: str, graph: Dict[str, Any], repo_root: Path) -> List[CalleeInfo]:
    """Helper to get callees for a resolved symbol."""
    callees = []
    edges = graph.get("edges", [])

    for edge in edges:
        if edge.get("type") == "calls" and edge.get("from") == symbol:
            to_key = edge.get("to", "")
            node_data = graph.get("nodes", {}).get(to_key, {})
            callees.append(CalleeInfo(
                symbol=to_key,
                file=repo_root / node_data.get("file", ""),
                line=node_data.get("line", 0),
                call_type="direct",
            ))

    return callees


def find_usages(symbol: str, repo_root: Optional[Path] = None) -> List[UsageInfo]:
    """Find all usages of a symbol (calls, refs, imports).

    If symbol is a simple name (no ::), uses symbolIndex to resolve ambiguity.
    """
    repo_root = repo_root or Path.cwd()
    cache = GraphCache(repo_root)
    graph = cache.load()

    if not graph:
        logger.warning("No graph available for find_usages. Run 'scout doc sync' first.")
        return []

    # Handle simple name resolution via symbolIndex
    resolved_symbols = [symbol]
    if "::" not in symbol:
        symbol_index = graph.get("symbolIndex", {})
        index_entry = symbol_index.get(symbol)
        if index_entry:
            resolved_symbols = index_entry.get("options", [symbol])

    usages = []
    edges = graph.get("edges", [])

    for resolved in resolved_symbols:
        for edge in edges:
            if edge.get("from") == resolved or edge.get("to") == resolved:
                from_key = edge.get("from", "")
                to_key = edge.get("to", "")
                edge_type = edge.get("type", "reference")

                # Add caller usage
                if from_key == resolved:
                    node_data = graph.get("nodes", {}).get(to_key, {})
                    usages.append(UsageInfo(
                        symbol=to_key,
                        file=repo_root / node_data.get("file", ""),
                        line=edge.get("line", 0),
                        usage_type=f"calls_{edge_type}",
                    ))

                # Add callee usage
                if to_key == resolved:
                    node_data = graph.get("nodes", {}).get(from_key, {})
                    usages.append(UsageInfo(
                        symbol=from_key,
                        file=repo_root / node_data.get("file", ""),
                        line=edge.get("line", 0),
                        usage_type=f"called_by_{edge_type}",
                    ))

    return usages


def trace_path(
    from_symbol: str,
    to_symbol: str,
    repo_root: Optional[Path] = None,
    max_depth: int = 10,
) -> List[PathStep]:
    """Find shortest path between two symbols via BFS."""
    repo_root = repo_root or Path.cwd()
    cache = GraphCache(repo_root)
    graph = cache.load()

    if not graph:
        logger.warning("No graph available for trace_path. Run 'scout doc sync' first.")
        return []

    # Resolve simple names via symbolIndex
    from_resolved = _resolve_simple_name(from_symbol, graph)
    to_resolved = _resolve_simple_name(to_symbol, graph)

    if not from_resolved or not to_resolved:
        return []

    # BFS
    edges = graph.get("edges", [])
    edges_by_from: Dict[str, List[str]] = {}
    for edge in edges:
        if edge.get("type") == "calls":
            from_key = edge.get("from", "")
            to_key = edge.get("to", "")
            if from_key not in edges_by_from:
                edges_by_from[from_key] = []
            edges_by_from[from_key].append(to_key)

    # BFS to find path
    queue: List[tuple[str, List[PathStep]]] = [(from_resolved, [])]
    visited: Set[str] = {from_resolved}

    while queue and len(queue[0][1]) < max_depth:
        current, path = queue.pop(0)

        if current == to_resolved:
            return path

        for next_symbol in edges_by_from.get(current, []):
            if next_symbol not in visited:
                visited.add(next_symbol)
                new_path = path + [PathStep(
                    from_symbol=current,
                    to_symbol=next_symbol,
                    edge_type="calls",
                )]
                queue.append((next_symbol, new_path))

    return []


def _resolve_simple_name(symbol: str, graph: Dict[str, Any]) -> Optional[str]:
    """Resolve a simple symbol name to its qualified name using symbolIndex."""
    if "::" in symbol:
        return symbol  # Already qualified

    symbol_index = graph.get("symbolIndex", {})
    index_entry = symbol_index.get(symbol)
    if index_entry and index_entry.get("options"):
        # Return first option (non-ambiguous) or None if ambiguous (user must disambiguate)
        if not index_entry.get("ambiguous"):
            return index_entry["options"][0]
    return None


def impact_analysis(
    file_path: str,
    repo_root: Optional[Path] = None,
) -> Set[str]:
    """Find all symbols transitively affected by changes to file."""
    repo_root = repo_root or Path.cwd()
    cache = GraphCache(repo_root)
    graph = cache.load()

    if not graph:
        logger.warning("No graph available for impact_analysis. Run 'scout doc sync' first.")
        return set()

    # Find all symbols defined in this file
    nodes = graph.get("nodes", {})
    file_symbols = set()
    for node_key, node_data in nodes.items():
        if node_data.get("file") == file_path:
            file_symbols.add(node_key)

    if not file_symbols:
        return set()

    # Find all symbols that depend on these (transitive)
    edges = graph.get("edges", [])
    dependents: Set[str] = set()
    to_visit = list(file_symbols)
    visited: Set[str] = set(file_symbols)

    while to_visit:
        current = to_visit.pop()
        for edge in edges:
            if edge.get("type") == "calls" and edge.get("to") == current:
                caller = edge.get("from", "")
                if caller not in visited:
                    visited.add(caller)
                    dependents.add(caller)
                    to_visit.append(caller)

    return dependents


# === Graph Building ===

def build_relationship_graph(
    scope: Path,
    force: bool = False,
    config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build aggregated relationship graph for scope.

    Returns:
        {
            "nodes": {...},
            "edges": [...],
            "metadata": {"version": "1.0", "built_at": "..."}
        }
    """
    from datetime import datetime

    # TODO: Phase 2 - extract doc_sync module
    # from scout.doc_sync.ast_facts import ASTFactExtractor
    # ASTFactExtractor will be used here when available
    pass

    scope = Path(scope).resolve()
    # If scope is a file, use its parent directory for scanning
    scan_dir = scope.parent if scope.is_file() else scope
    # Use cwd as repo root for caching unless explicitly provided
    repo_root = Path.cwd()
    
    cache = GraphCache(repo_root)

    # Check cache
    if not force:
        cached = cache.load()
        if cached:
            return cached

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    all_symbols: Dict[str, List[str]] = {}  # simple_name -> [qualified_name, ...]

    # Find all Python files
    py_files = list(scan_dir.rglob("*.py")) if scan_dir.is_dir() else [scope]

    extractor = ASTFactExtractor()

    for py_path in py_files:
        if "__pycache__" in str(py_path):
            continue

        try:
            facts = extractor.extract(py_path)
        except Exception:
            continue

        rel_path = str(py_path.relative_to(scan_dir))

        # Add nodes for each symbol
        for sym_name, sym_fact in facts.symbols.items():
            node_key = f"{rel_path}::{sym_name}"
            nodes[node_key] = {
                "type": sym_fact.type,
                "file": rel_path,
                "line": sym_fact.defined_at,
            }

            # Track symbol for index
            if sym_name not in all_symbols:
                all_symbols[sym_name] = []
            all_symbols[sym_name].append(node_key)

        # Add edges from relations
        for sym_name, rel_data in facts.relations.items():
            from_key = f"{rel_path}::{sym_name}"

            for call in rel_data.calls:
                edges.append({
                    "from": from_key,
                    "to": call.callee,
                    "type": "calls",
                    "line": call.line,
                })

            for parent in rel_data.inherits_from:
                edges.append({
                    "from": from_key,
                    "to": parent,
                    "type": "inherits",
                    "line": 0,
                })

    # Build symbol index
    symbol_index: Dict[str, Dict[str, Any]] = {}
    for simple_name, options in all_symbols.items():
        symbol_index[simple_name] = {
            "options": options,
            "ambiguous": len(options) > 1,
        }

    graph = {
        "graph_version": "1.0",
        "built_at": datetime.now().isoformat(),
        "nodes": nodes,
        "edges": edges,
        "symbolIndex": symbol_index,
    }

    # Cache it
    cache.save(graph)

    return graph
