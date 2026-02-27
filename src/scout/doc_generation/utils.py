"""Utility functions for doc_generation package."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scout.adapters.base import SymbolTree
from scout.config import ScoutConfig

logger = logging.getLogger(__name__)

# ANSI escape codes for terminal output
_RESET = "\033[0m"
_RED = "\033[91m"

# Fallback models when config does not specify
TLDR_MODEL = "llama-3.1-8b-instant"
DEEP_MODEL = "llama-3.1-8b-instant"
ELIV_MODEL = "llama-3.1-8b-instant"

# File extensions to process in directory mode
_DIRECTORY_PATTERNS = ["**/*.py", "**/*.js", "**/*.mjs", "**/*.cjs"]

# Path to groq model specs (relative to this package)
_GROQ_SPECS_PATH = Path(__file__).parent.parent / "config" / "groq_model_specs.json"

_groq_specs_cache: Optional[Dict[str, Any]] = None


def _get_tldr_meta_path(file_path: Path, output_dir: Optional[Path]) -> Path:
    """Path to .tldr.md.meta for freshness check. Mirrors write_documentation_files logic."""
    file_path = Path(file_path).resolve()
    if output_dir is not None:
        out = Path(output_dir).resolve()
        base_name = file_path.stem
        return out / f"{base_name}.tldr.md.meta"
    local_dir = file_path.parent / ".docs"
    base_name = file_path.name
    return local_dir / f"{base_name}.tldr.md.meta"


def _compute_source_hash(file_path: Path) -> str:
    """SHA256 of file content for freshness check."""
    data = file_path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def extract_source_snippet(file_path: Path, start_line: int, end_line: int) -> str:
    """
    Read a file and return the raw source code lines between
    start_line and end_line inclusive.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="strict") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except OSError as e:
        raise IOError(f"Could not read file {file_path}: {e}") from e
    except UnicodeDecodeError:
        raise

    if not lines:
        return ""

    start_idx = max(0, min(start_line - 1, len(lines) - 1))
    end_idx = max(0, min(end_line - 1, len(lines) - 1))
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    return "".join(lines[start_idx : end_idx + 1])


def _compute_symbol_hash(symbol: SymbolTree, file_path: Path) -> str:
    """SHA256 of symbol's source range for diff-aware patching."""
    snippet = extract_source_snippet(file_path, symbol.lineno, symbol.end_lineno)
    return hashlib.sha256(snippet.encode("utf-8")).hexdigest()


def _read_freshness_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    """Read .tldr.md.meta JSON. Returns None if missing or invalid."""
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _is_up_to_date(file_path: Path, output_dir: Optional[Path]) -> bool:
    """True if .tldr.md.meta exists and source_hash matches current file."""
    meta_path = _get_tldr_meta_path(file_path, output_dir)
    meta = _read_freshness_meta(meta_path)
    if not meta:
        return False
    current_hash = _compute_source_hash(file_path)
    return meta.get("source_hash") == current_hash


def _module_to_file_path(repo_root: Path, qual: str) -> Optional[Tuple[str, str]]:
    """
    Resolve qualified name (e.g. scout.llm.call_groq_async) to (file_path, symbol).
    Returns (repo_relative_path, symbol_name) or None if unresolvable.
    """
    parts = qual.split(".")
    if len(parts) < 2:
        return None
    symbol = parts[-1]
    # Try progressively shorter module paths: scout.llm -> scout -> vivarium
    for i in range(len(parts) - 1, 0, -1):
        mod = ".".join(parts[:i])
        path_str = mod.replace(".", "/")
        for candidate in [
            repo_root / f"{path_str}.py",
            repo_root / path_str / "__init__.py",
        ]:
            if candidate.exists():
                try:
                    rel = str(candidate.relative_to(repo_root))
                    return (rel, symbol)
                except ValueError:
                    pass
    return None


def _write_freshness_meta(
    meta_path: Path,
    source_hash: str,
    model: str,
    symbols: Optional[Dict[str, Dict[str, str]]] = None,
) -> None:
    """Write .tldr.md.meta. Not mirrored to docs/livingDoc/."""
    from datetime import datetime, timezone

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta: Dict[str, Any] = {
        "source_hash": source_hash,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": model,
    }
    if symbols:
        meta["symbols"] = symbols
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _resolve_doc_model(kind: str) -> str:
    """Resolve model from config (models.tldr, models.deep, models.eliv, models.pr_synthesis)."""
    config = ScoutConfig()
    models = config.get("models") or {}
    model = models.get(kind)
    if model:
        return model
    fallbacks = {
        "tldr": TLDR_MODEL,
        "deep": DEEP_MODEL,
        "eliv": ELIV_MODEL,
        "pr_synthesis": TLDR_MODEL,
    }
    return fallbacks.get(kind, TLDR_MODEL)


def get_model_specs() -> Dict[str, Any]:
    """Load groq_model_specs.json. Cached after first load."""
    global _groq_specs_cache
    if _groq_specs_cache is not None:
        return _groq_specs_cache
    if _GROQ_SPECS_PATH.exists():
        try:
            with open(_GROQ_SPECS_PATH, encoding="utf-8") as f:
                _groq_specs_cache = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load groq_model_specs.json: %s", e)
    _groq_specs_cache = _groq_specs_cache or {}
    return _groq_specs_cache


def _safe_workers_from_rpm(model_name: str, rpm: int) -> int:
    """Compute safe worker count: 80% of RPM, divided by 60 (per sec) and 3 (tldr/deep/eliv per file)."""
    safe = max(1, int((rpm * 0.8) / 60 / 3))
    return safe


def _max_concurrent_from_rpm(rpm: int) -> int:
    """Max concurrent LLM calls to stay just below rate limit.
    Assumes ~2 sec avg latency: rpm/60 req/sec * 2 sec ≈ rpm/30 in-flight. Use 85% for safety."""
    return min(100, max(1, int(rpm * 0.85 / 30)))


def _default_workers() -> int:
    """Default max concurrent LLM calls: min(8, cpu_count)."""
    n = os.cpu_count()
    return min(8, n if n is not None else 1)


def extract_source_snippet(file_path: Path, start_line: int, end_line: int) -> str:
    """
    Read a file and return the raw source code lines between
    start_line and end_line inclusive.
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="strict") as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except OSError as e:
        raise IOError(f"Could not read file {file_path}: {e}") from e
    except UnicodeDecodeError:
        raise

    if not lines:
        return ""

    start_idx = max(0, min(start_line - 1, len(lines) - 1))
    end_idx = max(0, min(end_line - 1, len(lines) - 1))
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    return "".join(lines[start_idx : end_idx + 1])


def _fallback_template_content(symbol: SymbolTree, kind: str) -> str:
    """
    Generate template doc content when LLM fails or budget exceeded.
    Returns [FALLBACK] header + template. kind: tldr, deep, eliv.
    """
    sig = getattr(symbol, "signature", None) or f"{symbol.name}(...)"
    args = "?"
    if "(" in sig:
        try:
            args = sig.split("(", 1)[1].rsplit(")", 1)[0] or "..."
        except IndexError:
            args = "..."
    call_list = ", ".join((symbol.calls or [])[:10]) or "none"
    type_list = ", ".join((symbol.uses_types or [])[:10]) or "none"
    header = "[FALLBACK]\n\n"
    if kind == "tldr":
        return f"{header}[AUTO] {symbol.name}({args}) → ?\nCalls: {call_list}\nTypes: {type_list}"
    if kind == "deep":
        return f"{header}[AUTO] {symbol.name}({args})\nCalls: {call_list}\nTypes: {type_list}\n\n(Generated from template; LLM unavailable.)"
    if kind == "eliv":
        return f"{header}{symbol.name} does something with {call_list or 'nothing'}."
    return f"{header}{symbol.name}"


def find_stale_files(
    target_path: Path,
    *,
    recursive: bool = True,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """
    Find .py files whose docs are stale (meta exists but source_hash mismatch).

    Returns list of file paths that need reprocessing.
    """
    if not target_path.exists():
        return []
    if target_path.is_file():
        if target_path.suffix == ".py":
            meta = _read_freshness_meta(_get_tldr_meta_path(target_path, output_dir))
            if meta and meta.get("source_hash") != _compute_source_hash(target_path):
                return [target_path]
        return []

    patterns = _DIRECTORY_PATTERNS if recursive else ["*.py", "*.js", "*.mjs", "*.cjs"]
    files: List[Path] = []
    for pattern in patterns:
        for f in target_path.glob(pattern):
            if f.is_file() and "__pycache__" not in str(f) and f.suffix == ".py":
                meta = _read_freshness_meta(_get_tldr_meta_path(f, output_dir))
                if meta and meta.get("source_hash") != _compute_source_hash(f):
                    files.append(f)
    return files


def _rel_path_for_display(path: Path) -> str:
    """Return path relative to cwd for compact display."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
