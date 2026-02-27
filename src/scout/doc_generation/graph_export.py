"""Graph export functions for documentation generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from scout.adapters.registry import get_adapter_for_path
from scout.doc_generation.utils import _module_to_file_path


def export_call_graph(
    target_path: Path,
    *,
    output_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Path:
    """
    Build and export call graph as JSON (nodes: path::symbol, edges: calls).
    Used by scout-pr for impact analysis.

    Format:
      nodes: { "path::symbol": { "type": "function", "file": "path" } }
      edges: [ { "from": "path::symbol", "to": "path::symbol", "type": "calls" } ]
    """
    target_path = Path(target_path).resolve()
    root = repo_root or Path.cwd().resolve()
    if output_path is None:
        docs_dir = target_path / ".docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        output_path = docs_dir / "call_graph.json"
    output_path = Path(output_path).resolve()

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, str]] = []

    for py_path in target_path.rglob("*.py"):
        if "__pycache__" in str(py_path):
            continue
        try:
            adapter = get_adapter_for_path(py_path, "python")
        except Exception:
            continue
        try:
            root_tree = adapter.parse(py_path)
        except (SyntaxError, UnicodeDecodeError):
            continue

        try:
            rel = str(py_path.relative_to(root))
        except ValueError:
            rel = str(py_path)

        for symbol in root_tree.iter_symbols():
            if symbol.name.startswith("_") and not symbol.name.startswith("__"):
                continue
            node_key = f"{rel}::{symbol.name}"
            if node_key not in nodes:
                nodes[node_key] = {"type": symbol.type, "file": rel}

            for call in getattr(symbol, "calls", None) or []:
                resolved = _module_to_file_path(root, call)
                if resolved:
                    callee_path, callee_sym = resolved
                    callee_key = f"{callee_path}::{callee_sym}"
                    if callee_key not in nodes:
                        nodes[callee_key] = {"type": "function", "file": callee_path}
                    edges.append({"from": node_key, "to": callee_key, "type": "calls"})

    payload = {"nodes": nodes, "edges": edges}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def get_downstream_impact(
    changed_files: List[Path],
    call_graph_path: Path,
    repo_root: Path,
) -> List[str]:
    """
    Given changed files and call_graph.json, return list of module paths
    affected (changed files + their downstream callees).
    """
    if not call_graph_path.exists():
        return []
    try:
        data = json.loads(call_graph_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    nodes = data.get("nodes", {})
    edges = data.get("edges", [])

    def _rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(repo_root))
        except ValueError:
            return str(p)

    changed_set: Set[str] = set()
    for f in changed_files:
        if f.suffix == ".py":
            changed_set.add(_rel(f))

    # Build reverse: callee -> callers (we need forward: caller -> callees)
    # Affected = changed files + all files they call (transitively)
    affected: Set[str] = set(changed_set)
    from_to: Dict[str, Set[str]] = {}
    for e in edges:
        fr = e.get("from", "")
        to = e.get("to", "")
        if "::" in fr and "::" in to:
            file_from = fr.split("::", 1)[0]
            file_to = to.split("::", 1)[0]
            if file_from not in from_to:
                from_to[file_from] = set()
            from_to[file_from].add(file_to)

    # Transitive closure: from each changed file, add all reachable callees
    work = list(changed_set)
    while work:
        f = work.pop()
        for callee in from_to.get(f, []):
            if callee not in affected:
                affected.add(callee)
                work.append(callee)

    return sorted(affected)


def export_knowledge_graph(
    target_path: Path,
    *,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Build and export knowledge graph as JSON (nodes: files/funcs/classes, edges: calls/uses/exports).

    Format compatible with Neo4j, Obsidian, or RAG ingestion.
    Returns path to written file.
    """
    target_path = Path(target_path).resolve()
    if output_path is None:
        output_path = target_path / "vivarium.kg.json"
    output_path = Path(output_path).resolve()

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_ids: Dict[str, str] = {}

    def _id(typ: str, path: str, name: str = "") -> str:
        key = f"{typ}:{path}:{name}"
        if key not in node_ids:
            nid = f"n{len(node_ids)}"
            node_ids[key] = nid
        return node_ids[key]

    for py_path in target_path.rglob("*.py"):
        if "__pycache__" in str(py_path):
            continue
        try:
            adapter = get_adapter_for_path(py_path, "python")
        except Exception:
            continue
        try:
            root = adapter.parse(py_path)
        except (SyntaxError, UnicodeDecodeError):
            continue

        rel = str(py_path.relative_to(target_path))
        file_id = _id("file", rel)
        nodes.append({
            "id": file_id,
            "type": "file",
            "path": rel,
            "name": py_path.stem,
            "tldr": None,
        })

        for symbol in root.iter_symbols():
            if symbol.name.startswith("_") and not symbol.name.startswith("__"):
                continue
            sym_type = symbol.type
            qual = f"{rel}::{symbol.name}"
            sym_id = _id("symbol", rel, symbol.name)
            nodes.append({
                "id": sym_id,
                "type": sym_type,
                "path": rel,
                "name": symbol.name,
                "qual": qual,
            })
            edges.append({"from": sym_id, "to": file_id, "type": "defined_in"})

            for call in getattr(symbol, "calls", None) or []:
                edges.append({"from": sym_id, "to": call, "type": "calls"})
            for typ in getattr(symbol, "uses_types", None) or []:
                edges.append({"from": sym_id, "to": typ, "type": "uses"})
            for exp in getattr(symbol, "exports", None) or []:
                edges.append({"from": sym_id, "to": exp, "type": "exports"})

    kg = {"nodes": nodes, "edges": edges}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(kg, indent=2), encoding="utf-8")
    return output_path
