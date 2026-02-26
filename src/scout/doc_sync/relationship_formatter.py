"""Format relationship data for injection into documentation.

This module queries the code relationship graph and formats the results
for inclusion in generated documentation (TL;DR, deep, ELIV).

Key design decisions:
- Graph is loaded once per repo and cached (not re-loaded per symbol)
- Simple symbol names are resolved to qualified names using symbolIndex
- Graceful fallback: empty data returned if graph unavailable or symbol not found
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from scout.config import ScoutConfig
from scout.graph import (
    GraphCache,
    CallerInfo,
    CalleeInfo,
    UsageInfo,
    get_callers as _get_callers,
    get_callees as _get_callees,
    find_usages as _find_usages,
)

logger = logging.getLogger(__name__)

# Module-level graph cache (loaded once per repo)
_graph_cache: Dict[str, Any] = {}
_loaded_repo_root: Optional[Path] = None


def _ensure_graph_loaded(repo_root: Path) -> Optional[Dict[str, Any]]:
    """Load graph once per repo. Returns None if unavailable."""
    global _graph_cache, _loaded_repo_root
    if _loaded_repo_root != repo_root:
        cache = GraphCache(repo_root)
        _graph_cache = cache.load() or {}
        _loaded_repo_root = repo_root
    return _graph_cache


def _resolve_qualified_name(
    symbol_name: str, file_path: Path, graph: Dict[str, Any]
) -> str:
    """Resolve simple name to qualified name using graph's symbolIndex.

    Graph stores keys as 'relpath::SymbolName'. First tries exact path match,
    then falls back to symbolIndex lookup.
    """
    # Normalize file path to relative
    try:
        rel_path = str(file_path.relative_to(Path.cwd()))
    except ValueError:
        rel_path = str(file_path)

    # Try exact match first
    qualified = f"{rel_path}::{symbol_name}"
    if qualified in graph.get("nodes", {}):
        return qualified

    # Fallback: try simple name lookup via symbolIndex
    index = graph.get("symbolIndex", {}).get(symbol_name, {})
    options = index.get("options", [])
    if options:
        return options[0]  # Return first match

    # Last resort: return as-is (will likely return empty results)
    return qualified


@dataclass
class RelationshipData:
    """Container for relationship data for a symbol."""

    callers: List[CallerInfo]
    callees: List[CalleeInfo]
    usages: List[UsageInfo]
    base_classes: List[str]
    subclasses: List[str]


def get_relationship_data(
    symbol_name: str,
    symbol_type: str,
    file_path: Path,
    repo_root: Optional[Path] = None,
    config: Optional[ScoutConfig] = None,
) -> RelationshipData:
    """Query the relationship graph for a symbol.

    Args:
        symbol_name: Simple name of the symbol (e.g., 'process_tokens')
        symbol_type: Type of symbol ('function', 'class', etc.)
        file_path: Path to the file containing the symbol
        repo_root: Repository root (defaults to cwd)
        config: ScoutConfig (used to check if feature is enabled)

    Returns:
        RelationshipData with all queried relationships. Returns empty lists
        if graph unavailable or symbol not found.
    """
    repo_root = repo_root or Path.cwd()

    # Check if feature is disabled via config
    if config is not None:
        doc_rels = config.get_doc_relationships_config()
        if not doc_rels.get("enabled", True):
            return RelationshipData([], [], [], [], [])

    graph = _ensure_graph_loaded(repo_root)
    if not graph:
        logger.debug("No graph available for relationship queries")
        return RelationshipData([], [], [], [], [])

    # Resolve to qualified name
    qualified_name = _resolve_qualified_name(symbol_name, file_path, graph)

    # Query graph using cached data
    # Note: _get_callers/_get_callees/_find_usages reload graph internally
    # but the cache above ensures we only load once per repo
    callers = _get_callers(qualified_name, repo_root)
    callees = _get_callees(qualified_name, repo_root)
    usages = _find_usages(qualified_name, repo_root)

    # Query inheritance from edges
    base_classes = []
    subclasses = []
    edges = graph.get("edges", [])
    for edge in edges:
        if edge.get("type") == "inherits":
            if edge.get("to") == qualified_name:
                base_classes.append(edge.get("from", ""))
            if edge.get("from") == qualified_name:
                subclasses.append(edge.get("to", ""))

    return RelationshipData(callers, callees, usages, base_classes, subclasses)


def format_callers(callers: List[CallerInfo], max_items: int = 5) -> str:
    """Format caller list as markdown.

    Args:
        callers: List of CallerInfo objects
        max_items: Maximum number to show (truncates with '... and N more')

    Returns:
        Markdown formatted string with caller list
    """
    if not callers:
        return ""

    lines = ["## Callers"]
    display_items = callers[:max_items]
    for caller in display_items:
        line_num = f" (line {caller.line})" if caller.line else ""
        ambiguous = " (AMBIGUOUS)" if caller.ambiguous else ""
        lines.append(f"- `{caller.symbol}`{line_num}{ambiguous}")

    if len(callers) > max_items:
        lines.append(f"- *... and {len(callers) - max_items} more*")

    return "\n".join(lines)


def format_callees(callees: List[CalleeInfo], max_items: int = 5) -> str:
    """Format callee list as markdown.

    Args:
        callees: List of CalleeInfo objects
        max_items: Maximum number to show (truncates with '... and N more')

    Returns:
        Markdown formatted string with callee list
    """
    if not callees:
        return ""

    lines = ["## Callees"]
    display_items = callees[:max_items]
    for callee in display_items:
        line_num = f" (line {callee.line})" if callee.line else ""
        ambiguous = " (AMBIGUOUS)" if callee.ambiguous else ""
        lines.append(f"- `{callee.symbol}`{line_num}{ambiguous}")

    if len(callees) > max_items:
        lines.append(f"- *... and {len(callees) - max_items} more*")

    return "\n".join(lines)


def format_examples(usages: List[UsageInfo], max_items: int = 3) -> str:
    """Format usage examples as code blocks.

    Extracts up to max_items usage snippets and formats as markdown code blocks.

    Args:
        usages: List of UsageInfo objects
        max_items: Maximum number of examples to show

    Returns:
        Markdown formatted string with code examples
    """
    if not usages:
        return ""

    lines = ["## Usage Examples"]
    display_items = usages[:max_items]

    for usage in display_items:
        # Format: source_file:line_number
        location = f"{usage.file.name}:{usage.line}" if usage.file else f"line {usage.line}"
        lines.append(f"\n```python")
        lines.append(f"# {location}")
        lines.append(f"# Usage type: {usage.usage_type}")
        lines.append(f"{usage.symbol}()")
        lines.append(f"```")

    if len(usages) > max_items:
        lines.append(f"\n*... and {len(usages) - max_items} more usages*")

    return "\n".join(lines)


def format_inheritance(base_classes: List[str], subclasses: List[str]) -> str:
    """Format inheritance tree for classes.

    Args:
        base_classes: List of base class qualified names
        subclasses: List of subclass qualified names

    Returns:
        Markdown formatted string with inheritance info
    """
    if not base_classes and not subclasses:
        return ""

    lines = ["## Inheritance"]

    if base_classes:
        lines.append("\n**Base classes:**")
        for bc in base_classes:
            # Extract just the class name from qualified name
            class_name = bc.split("::")[-1] if "::" in bc else bc
            lines.append(f"- `{class_name}`")

    if subclasses:
        lines.append("\n**Subclasses:**")
        for sc in subclasses:
            class_name = sc.split("::")[-1] if "::" in sc else sc
            lines.append(f"- `{class_name}`")

    return "\n".join(lines)


def generate_mermaid_inheritance(
    class_name: str,
    base_classes: List[str],
    subclasses: List[str],
) -> str:
    """Generate Mermaid classDiagram for class inheritance.

    Args:
        class_name: Name of the class
        base_classes: List of base class qualified names
        subclasses: List of subclass qualified names

    Returns:
        Mermaid classDiagram markup, or empty string if no relationships
    """
    if not base_classes and not subclasses:
        return ""

    lines = ["```mermaid", "classDiagram"]

    # Add the main class
    lines.append(f"    class {class_name} {{")

    # Add base classes
    for bc in base_classes:
        base_name = bc.split("::")[-1] if "::" in bc else bc
        lines.append(f"    {base_name} <|-- {class_name}")

    # Add subclasses
    for sc in subclasses:
        sub_name = sc.split("::")[-1] if "::" in sc else sc
        lines.append(f"    {class_name} <|-- {sub_name}")

    lines.append("```")

    return "\n".join(lines)


def inject_relationship_sections(
    content: str,
    symbol_name: str,
    symbol_type: str,
    file_path: Path,
    repo_root: Path,
    config: ScoutConfig,
) -> str:
    """Inject relationship sections into generated documentation.

    This is the main entry point for post-processing injection. It queries
    the relationship graph and formats the results as markdown sections.

    Args:
        content: Existing generated documentation content
        symbol_name: Name of the symbol being documented
        symbol_type: Type of symbol ('function', 'class', etc.)
        file_path: Path to the source file
        repo_root: Repository root path
        config: ScoutConfig for feature flags and limits

    Returns:
        Content with relationship sections appended
    """
    doc_rels = config.get_doc_relationships_config()

    # Check if feature is enabled
    if not doc_rels.get("enabled", True):
        return content

    # Get relationship data
    rel_data = get_relationship_data(
        symbol_name=symbol_name,
        symbol_type=symbol_type,
        file_path=file_path,
        repo_root=repo_root,
        config=config,
    )

    # Build sections based on config
    sections = []
    max_items = doc_rels.get("max_related_items", 5)

    if doc_rels.get("include_callers", True) and rel_data.callers:
        sections.append(format_callers(rel_data.callers, max_items))

    if doc_rels.get("include_callees", True) and rel_data.callees:
        sections.append(format_callees(rel_data.callees, max_items))

    if doc_rels.get("include_examples", True) and rel_data.usages:
        sections.append(format_examples(rel_data.usages, max_items=3))

    if symbol_type == "class":
        if doc_rels.get("include_inheritance", True):
            sections.append(
                format_inheritance(rel_data.base_classes, rel_data.subclasses)
            )
        if doc_rels.get("include_mermaid", True):
            mermaid = generate_mermaid_inheritance(
                symbol_name, rel_data.base_classes, rel_data.subclasses
            )
            if mermaid:
                sections.append(mermaid)

    if not sections:
        return content

    # Append sections to content
    return content + "\n\n" + "\n\n".join(sections)


