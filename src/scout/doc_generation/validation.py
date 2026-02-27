"""Validation functions for generated documentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from scout.adapters.base import SymbolTree


def validate_no_placeholders(content: str, filepath: str) -> Tuple[bool, List[str]]:
    """
    Check for GAP/FALLBACK placeholders in generated content.

    Returns:
        (is_valid, list_of_found_markers).
    """
    forbidden = ["[FALLBACK]", "[GAP]", "[PLACEHOLDER]"]
    found: List[str] = []
    for marker in forbidden:
        if marker in content:
            found.append(marker)
    return (len(found) == 0, found)


def validate_content_for_placeholders(
    target_path: Path,
    *,
    recursive: bool = True,
) -> Tuple[bool, List[Tuple[str, List[str]]]]:
    """
    Scan all .tldr.md, .deep.md, .eliv.md files under target for placeholders.

    Returns:
        (all_clean, list of (filepath, list_of_markers) for violations).
    """
    target_path = Path(target_path).resolve()
    violations: List[Tuple[str, List[str]]] = []

    patterns = ["**/*.tldr.md", "**/*.deep.md", "**/*.eliv.md"] if recursive else ["*.tldr.md", "*.deep.md", "*.eliv.md"]
    for pattern in patterns:
        for f in target_path.glob(pattern):
            if not f.is_file() or "__pycache__" in str(f):
                continue
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            is_valid, found = validate_no_placeholders(content, str(f))
            if not is_valid:
                try:
                    rel = str(f.relative_to(Path.cwd().resolve()))
                except ValueError:
                    rel = str(f)
                violations.append((rel, found))

    return (len(violations) == 0, violations)


def validate_generated_docs(
    symbol: SymbolTree | Dict[str, Any],
    tldr_content: str,
    deep_content: str,
) -> Tuple[bool, List[str]]:
    """
    Validate generated documentation content for a symbol.

    Args:
        symbol: SymbolTree or dict (for context in error messages).
        tldr_content: Generated TL;DR content.
        deep_content: Generated deep content.

    Returns:
        (is_valid, list of error messages).
    """
    errors: List[str] = []
    name = symbol.name if isinstance(symbol, SymbolTree) else symbol.get("name", "?")

    if not tldr_content or not tldr_content.strip():
        errors.append(f"TL;DR content is empty for symbol '{name}'")
    elif tldr_content.strip().startswith("[TL;DR generation failed:"):
        errors.append(f"TL;DR generation failed for symbol '{name}'")
    elif len(tldr_content) > 100_000:
        errors.append(f"TL;DR content exceeds size limit for symbol '{name}'")

    if not deep_content or not deep_content.strip():
        errors.append(f"Deep content is empty for symbol '{name}'")
    elif deep_content.strip().startswith("[Deep content generation failed:"):
        errors.append(f"Deep content generation failed for symbol '{name}'")
    elif len(deep_content) > 500_000:
        errors.append(f"Deep content exceeds size limit for symbol '{name}'")

    return (len(errors) == 0, errors)
