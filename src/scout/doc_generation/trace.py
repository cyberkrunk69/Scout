"""Call trace functions for documentation generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from scout.adapters.base import SymbolTree
from scout.adapters.registry import get_adapter_for_path
from scout.doc_generation.models import TraceResult
from scout.doc_generation.utils import _rel_path_for_display


# ANSI colors for tagged hops in rolling trace
_RESET = "\033[0m"
_TRACE_COLORS = ("\033[34m", "\033[35m", "\033[36m", "\033[32m", "\033[33m")
_MAX_CHAIN_LEN = 80
_ARROW = "\u27F6"  # ⟶


def _strip_ansi(s: str) -> str:
    """Return string with ANSI codes removed, for length calculation."""
    result = []
    i = 0
    while i < len(s):
        if s[i] == "\033" and i + 1 < len(s) and s[i + 1] == "[":
            j = i + 2
            while j < len(s) and s[j] != "m":
                j += 1
            i = j + 1
            continue
        result.append(s[i])
        i += 1
    return "".join(result)


def _build_rolling_call_trace(symbols_to_doc: List[SymbolTree]) -> Optional[str]:
    """
    Build a colorized, tagged rolling call chain for display.

    Collects all qualified calls from the file, tags each by module, colorizes,
    and truncates from the left if over 80 chars to keep latest context visible.
    """
    def _skip_call(qname: str) -> bool:
        parts = qname.split(".")
        last = parts[-1] if parts else ""
        return last == "__init__" or last.startswith("_")

    # Collect all calls in file order (entrypoint first, then calls per symbol)
    hops: List[str] = []
    seen: set = set()

    for symbol in symbols_to_doc:
        if symbol.name.startswith("_") and not symbol.name.startswith("__"):
            continue
        if symbol.name == "__init__":
            continue
        # Add entrypoint on first symbol with calls
        if not hops and (symbol.calls or []):
            hops.append(symbol.name)
        for qname in symbol.calls or []:
            if _skip_call(qname) or qname in seen:
                continue
            seen.add(qname)
            parts = qname.split(".")
            if len(parts) >= 2:
                module = parts[-2]
                func = parts[-1]
                tag = module[:3] if len(module) > 3 else module  # doc, llm, aud
            else:
                tag, func = "", qname

            color_idx = hash(tag) % len(_TRACE_COLORS) if tag else 0
            color = _TRACE_COLORS[color_idx]
            hop_str = f"{color}[{tag}]{_RESET} {func}" if tag else func
            hops.append(hop_str)

    if not hops:
        return None

    chain = f" {_ARROW} ".join(hops)

    # Truncate from left until it fits (keep latest context visible)
    while len(_strip_ansi(chain)) > _MAX_CHAIN_LEN and len(hops) > 1:
        hops = hops[1:]
        chain = f" {_ARROW} ".join(hops)

    return chain if chain else None


def _format_single_hop(qname: str) -> Optional[Tuple[str, str]]:
    """Format a qualified call to (tag, hop_str). Returns None if skip."""
    parts = qname.split(".")
    last = parts[-1] if parts else ""
    if last == "__init__" or last.startswith("_"):
        return None
    if len(parts) >= 2:
        module, func = parts[-2], parts[-1]
        tag = module[:3] if len(module) > 3 else module
        color_idx = hash(tag) % len(_TRACE_COLORS)
        hop_str = f"{_TRACE_COLORS[color_idx]}[{tag}]{_RESET} {func}"
        return (tag, hop_str)
    return ("", qname)


def _build_chain_from_hops(entrypoint: str, hop_strs: List[str]) -> str:
    """Build chain from entrypoint + hop strings, truncate from left if > 80 chars."""
    if not hop_strs:
        return entrypoint
    hops = [entrypoint] + hop_strs
    chain = f" {_ARROW} ".join(hops)
    while len(_strip_ansi(chain)) > _MAX_CHAIN_LEN and len(hops) > 1:
        hops = hops[1:]
        chain = f" {_ARROW} ".join(hops)
    return chain


async def _trace_file(
    target_path: Path,
    *,
    language_override: Optional[str] = None,
    dependencies_func: Optional[Callable[[Path], Union[List[str], Awaitable[List[str]]]]] = None,
    slot_id: Optional[int] = None,
    shared_display: Optional[Dict[str, Any]] = None,
) -> TraceResult:
    """
    Pure static analysis: AST + import map. No LLM calls.

    Returns SymbolTree with calls, types, exports. Updates dashboard with
    live call chain immediately (tracing is instant).
    """
    import asyncio

    adapter = get_adapter_for_path(target_path, language_override)
    root_tree = adapter.parse(target_path)

    dependencies: List[str] = []
    if dependencies_func:
        result = dependencies_func(target_path)
        if asyncio.iscoroutine(result):
            dependencies = (await result) or []
        else:
            dependencies = result or []

    symbols_to_doc: List[SymbolTree] = []
    for child in root_tree.children:
        symbols_to_doc.extend(list(child.iter_symbols()))
    if not symbols_to_doc:
        symbols_to_doc = [root_tree]

    all_calls: set = set()
    all_types: set = set()
    all_exports: set = set()
    for s in symbols_to_doc:
        all_calls.update(s.calls or [])
        all_types.add(s.type)
        all_exports.update(s.exports or [])
    root_exports = getattr(root_tree, "exports", None) or []
    all_exports.update(root_exports)

    chain = _build_rolling_call_trace(symbols_to_doc)
    if slot_id is not None and shared_display is not None:
        rel = _rel_path_for_display(target_path)
        shared_display[slot_id] = {
            "file": rel,
            "chain": chain or "…",
            "cost": 0.0,
            "status": "running",
            "pulse_hop": None,
        }

    return TraceResult(
        root_tree=root_tree,
        symbols_to_doc=symbols_to_doc,
        all_calls=all_calls,
        all_types=all_types,
        all_exports=all_exports,
        adapter=adapter,
        dependencies=dependencies,
    )
