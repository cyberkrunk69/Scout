"""Core documentation generation functions."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from scout.adapters.base import SymbolTree
from scout.adapters.registry import get_adapter_for_path
from scout.audit import AuditLog
from scout.big_brain import call_big_brain_async
from scout.cli.plan import generate_with_quality_loop
from scout.config import ScoutConfig
from scout.doc_sync.relationship_formatter import inject_relationship_sections
from scout.ignore import IgnorePatterns

from .models import BudgetExceededError, FileProcessResult
from .trace import _build_rolling_call_trace, _trace_file
from .utils import (
    _compute_source_hash,
    _compute_symbol_hash,
    _default_workers,
    _get_tldr_meta_path,
    _is_up_to_date,
    _max_concurrent_from_rpm,
    _rel_path_for_display,
    _resolve_doc_model,
    _safe_workers_from_rpm,
    _write_freshness_meta,
    extract_source_snippet,
    find_stale_files,
    get_model_specs,
)
from .validation import validate_generated_docs, validate_no_placeholders

logger = logging.getLogger(__name__)

# ANSI escape codes for terminal output
_RESET = "\033[0m"
_RED = "\033[91m"
_CLEAR_SCREEN = "\033[H\033[J"
_INVERSE = "\033[7m"
_INVERSE_OFF = "\033[27m"

# Directory patterns for file discovery
_DIRECTORY_PATTERNS = ["**/*.py", "**/*.js", "**/*.mjs", "**/*.cjs"]


def write_documentation_files(
    file_path: Path,
    tldr_content: str,
    deep_content: str,
    eliv_content: str = "",
    output_dir: Optional[Path] = None,
    generate_eliv: bool = True,
    versioned_mirror_dir: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    """
    Write documentation files for a source file.

    If output_dir is provided:
        Writes <stem>.tldr.md, <stem>.deep.md, and <name>.eliv.md inside output_dir.
    If output_dir is None (default):
        Writes to local .docs/ next to source and mirrors to central docs/livingDoc/.

    If versioned_mirror_dir is set (e.g. docs/livingDoc/v0.1.0-dev/), also mirrors there.

    Returns:
        Tuple of (tldr_path, deep_path, eliv_path) for the primary (local) files.
        When generate_eliv is False, eliv_path is still returned but not written.
    """
    file_path = Path(file_path).resolve()
    repo_root = Path.cwd().resolve()

    if output_dir is not None:
        out = Path(output_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        base_name = file_path.stem
        tldr_path = out / f"{base_name}.tldr.md"
        deep_path = out / f"{base_name}.deep.md"
        eliv_path = (out / file_path.name).with_suffix(file_path.suffix + ".eliv.md")
        mirror_to_central = False
    else:
        local_dir = file_path.parent / ".docs"
        local_dir.mkdir(parents=True, exist_ok=True)
        base_name = file_path.name
        tldr_path = local_dir / f"{base_name}.tldr.md"
        deep_path = local_dir / f"{base_name}.deep.md"
        eliv_path = local_dir / f"{base_name}.eliv.md"
        mirror_to_central = True

    tldr_path.write_text(tldr_content, encoding="utf-8")
    deep_path.write_text(deep_content, encoding="utf-8")
    if generate_eliv:
        eliv_path.write_text(eliv_content, encoding="utf-8")

    if mirror_to_central:
        try:
            rel = file_path.relative_to(repo_root)
            central_dir = repo_root / "docs" / "livingDoc" / rel.parent
            central_dir.mkdir(parents=True, exist_ok=True)
            central_tldr = central_dir / f"{file_path.name}.tldr.md"
            central_deep = central_dir / f"{file_path.name}.deep.md"
            central_eliv = central_dir / f"{file_path.name}.eliv.md"
            central_tldr.write_text(tldr_content, encoding="utf-8")
            central_deep.write_text(deep_content, encoding="utf-8")
            if generate_eliv:
                central_eliv.write_text(eliv_content, encoding="utf-8")
            if versioned_mirror_dir is not None:
                try:
                    vdir = versioned_mirror_dir / rel.parent
                    vdir.mkdir(parents=True, exist_ok=True)
                    (vdir / f"{file_path.name}.tldr.md").write_text(tldr_content, encoding="utf-8")
                    (vdir / f"{file_path.name}.deep.md").write_text(deep_content, encoding="utf-8")
                    if generate_eliv:
                        (vdir / f"{file_path.name}.eliv.md").write_text(eliv_content, encoding="utf-8")
                except (ValueError, OSError) as e:
                    logger.warning("Could not mirror to versioned dir for %s: %s", file_path, e)
        except (ValueError, OSError) as e:
            logger.warning(
                "Could not mirror docs to docs/livingDoc/ for %s: %s",
                file_path,
                e,
            )

    return (tldr_path, deep_path, eliv_path)


def _fallback_template_content(symbol: SymbolTree, kind: str) -> str:
    """Generate template doc content when LLM fails or budget exceeded."""
    from .utils import _fallback_template_content as utils_fallback
    return utils_fallback(symbol, kind)


async def _generate_single_symbol_docs(
    adapter: Any,
    symbol: SymbolTree,
    dependencies: List[str],
    source_snippet: str,
    semaphore: asyncio.Semaphore,
    generate_eliv: bool = True,
    fallback_template: bool = False,
) -> Tuple[str, bool, str, str, str, float, str]:
    """
    Generate TL;DR, deep, and ELIV content for a single symbol using the adapter.

    Returns:
        Tuple of (symbol_name, is_valid, tldr_content, deep_content, eliv_content, cost_usd, model).
    """
    cost_usd = 0.0
    model_used = _resolve_doc_model("tldr")

    async with semaphore:
        # TL;DR — try auto-generation for simple symbols (skip LLM)
        tldr_content = None
        if hasattr(adapter, "try_auto_tldr"):
            tldr_content = adapter.try_auto_tldr(symbol, source_snippet)
        if tldr_content is not None:
            audit = AuditLog()
            audit.log(
                "tldr_auto_generated",
                cost=0.0,
                symbol=symbol.name,
            )
        else:
            try:
                tldr_prompt = adapter.get_tldr_prompt(symbol, dependencies)
                tldr_content, metadata, attempts = await generate_with_quality_loop(
                    tldr_prompt,
                    system="You are a documentation assistant. Be concise and accurate.",
                    task_type="synthesis",
                )
                cost_usd += metadata.get("cost", 0)
                model_used = metadata.get("model", "synthesis")
                audit = AuditLog()
                audit.log(
                    "tldr",
                    cost=metadata.get("cost", 0),
                    model=model_used,
                    input_t=metadata.get("tokens_in", 0),
                    output_t=metadata.get("tokens_out", 0),
                    symbol=symbol.name,
                )
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning("TL;DR generation failed for %s: %s", symbol.name, e)
                if fallback_template:
                    tldr_content = _fallback_template_content(symbol, "tldr")
                    AuditLog().log("tldr_fallback_template", cost=0.0, symbol=symbol.name)
                else:
                    tldr_content = f"[TL;DR generation failed: {e}]"

        # Deep
        try:
            deep_prompt = adapter.get_deep_prompt(symbol, dependencies, source_snippet)
            deep_content, metadata, attempts = await generate_with_quality_loop(
                deep_prompt,
                system="You are a documentation assistant. Provide structured, detailed analysis of code.",
                task_type="synthesis",
                max_tokens=1500,
            )
            cost_usd += metadata.get("cost", 0)
            model_used = metadata.get("model", "synthesis")
            audit = AuditLog()
            audit.log(
                "deep",
                cost=metadata.get("cost", 0),
                model=model_used,
                input_t=metadata.get("tokens_in", 0),
                output_t=metadata.get("tokens_out", 0),
                symbol=symbol.name,
            )
        except RuntimeError:
            raise
        except Exception as e:
            logger.warning("Deep content generation failed for %s: %s", symbol.name, e)
            if fallback_template:
                deep_content = _fallback_template_content(symbol, "deep")
                AuditLog().log("deep_fallback_template", cost=0.0, symbol=symbol.name)
            else:
                deep_content = f"[Deep content generation failed: {e}]"

        # ELIV (skip if generate_eliv disabled)
        eliv_content = ""
        if generate_eliv:
            try:
                eliv_prompt = adapter.get_eliv_prompt(symbol, dependencies, source_snippet)
                eliv_content, metadata, attempts = await generate_with_quality_loop(
                    eliv_prompt,
                    system="You are a friendly assistant that explains code in very simple terms for young children.",
                    task_type="synthesis",
                    max_tokens=450,
                )
                cost_usd += metadata.get("cost", 0)
                model_used = metadata.get("model", "synthesis")
                audit = AuditLog()
                audit.log(
                    "eliv",
                    cost=metadata.get("cost", 0),
                    model=model_used,
                    input_t=metadata.get("tokens_in", 0),
                    output_t=metadata.get("tokens_out", 0),
                    symbol=symbol.name,
                )
            except RuntimeError:
                raise
            except Exception as e:
                logger.warning("ELIV generation failed for %s: %s", symbol.name, e)
                if fallback_template:
                    eliv_content = _fallback_template_content(symbol, "eliv")
                    AuditLog().log("eliv_fallback_template", cost=0.0, symbol=symbol.name)
                else:
                    eliv_content = f"[ELIV generation failed: {e}]"

        is_valid, errors = validate_generated_docs(symbol, tldr_content, deep_content)
        if not is_valid:
            for err in errors:
                logger.warning("Validation failed for %s: %s", symbol.name, err)

        return (symbol.name, is_valid, tldr_content, deep_content, eliv_content, cost_usd, model_used)


def _merge_symbol_content(
    symbols: List[SymbolTree],
    cached: Dict[str, Dict[str, str]],
    generated: Dict[str, Tuple[str, str, str]],
) -> Tuple[str, str, str, Dict[str, Dict[str, str]]]:
    """Merge cached + generated per-symbol content in symbol order. Return aggregated docs + symbols_for_meta."""
    tldr_agg = ""
    deep_agg = ""
    eliv_agg = ""
    symbols_for_meta: Dict[str, Dict[str, str]] = {}

    for symbol in symbols:
        name = symbol.name
        if name in generated:
            tldr_c, deep_c, eliv_c = generated[name]
        elif name in cached:
            tldr_c = cached[name].get("tldr", "")
            deep_c = cached[name].get("deep", "")
            eliv_c = cached[name].get("eliv", "")
        else:
            continue

        header = f"# {name}\n\n"
        if tldr_agg:
            tldr_agg += "\n---\n\n"
        tldr_agg += header + tldr_c
        if deep_agg:
            deep_agg += "\n---\n\n"
        deep_agg += header + deep_c
        if eliv_agg:
            eliv_agg += "\n---\n\n"
        eliv_agg += f"# {name} ELIV\n\n{eliv_c}"

        symbols_for_meta[name] = {
            "tldr": tldr_c,
            "deep": deep_c,
            "eliv": eliv_c,
        }

    return (tldr_agg, deep_agg, eliv_agg, symbols_for_meta)


async def _generate_docs_for_symbols(
    target_path: Path,
    trace,
    *,
    output_dir: Optional[Path] = None,
    generate_eliv: bool = True,
    per_file_concurrency: int = 3,
    slot_id: Optional[int] = None,
    shared_display: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    fallback_template: bool = False,
    include_relations: bool = True,
) -> Tuple[str, str, str, float, str, Dict[str, Dict[str, str]]]:
    """
    Generate docs via LLM for traced symbols. Diff-aware: only re-generate for symbols whose hash changed.

    Returns (tldr_agg, deep_agg, eliv_agg, total_cost, model_used, symbols_for_meta).
    symbols_for_meta: {name: {hash, tldr, deep, eliv}} for persistence.
    """
    from .trace import _read_freshness_meta

    symbols_to_doc = trace.symbols_to_doc
    adapter = trace.adapter
    dependencies = trace.dependencies

    meta_path = _get_tldr_meta_path(target_path, output_dir)
    meta = _read_freshness_meta(meta_path)
    meta_symbols = (meta or {}).get("symbols") or {}

    # Partition: unchanged (reuse from meta) vs changed (generate)
    to_reuse: Dict[str, Dict[str, str]] = {}
    to_generate: List[SymbolTree] = []

    for symbol in symbols_to_doc:
        current_hash = _compute_symbol_hash(symbol, target_path)
        prev = meta_symbols.get(symbol.name)
        if prev and prev.get("hash") == current_hash:
            to_reuse[symbol.name] = {
                "hash": current_hash,
                "tldr": prev.get("tldr", ""),
                "deep": prev.get("deep", ""),
                "eliv": prev.get("eliv", ""),
            }
        else:
            to_generate.append(symbol)

    if to_reuse or to_generate:
        logger.debug(
            "Diff-aware: %d reused, %d to generate",
            len(to_reuse),
            len(to_generate),
        )

    per_file_semaphore = asyncio.Semaphore(per_file_concurrency)
    running_cost = [0.0]

    # Show symbol-level progress for large files so users know it's not frozen
    if slot_id is not None and shared_display is not None and slot_id in shared_display:
        shared_display[slot_id]["symbols_total"] = len(to_generate)
        shared_display[slot_id]["symbols_done"] = 0

    def _on_symbol_done(sym_cost: float) -> None:
        running_cost[0] += sym_cost
        if progress_callback:
            progress_callback(running_cost[0])
        if slot_id is not None and shared_display is not None and slot_id in shared_display:
            shared_display[slot_id]["cost"] = running_cost[0]
            shared_display[slot_id]["symbols_done"] = (
                shared_display[slot_id].get("symbols_done", 0) + 1
            )

    async def _wrapped(symbol: SymbolTree) -> Tuple[str, str, str, str, float, str]:
        res = await _generate_single_symbol_docs(
            adapter,
            symbol,
            symbol.dependencies if symbol.dependencies else dependencies,
            extract_source_snippet(target_path, symbol.lineno, symbol.end_lineno),
            per_file_semaphore,
            generate_eliv,
            fallback_template=fallback_template,
        )
        _on_symbol_done(res[5])
        return (res[0], res[2], res[3], res[4], res[5], res[6])

    generated: Dict[str, Tuple[str, str, str]] = {}
    total_cost = 0.0
    model_used = _resolve_doc_model("tldr")

    if to_generate:
        tasks = [asyncio.create_task(_wrapped(s)) for s in to_generate]
        results = await asyncio.gather(*tasks)

        for (symbol, res) in zip(to_generate, results):
            sym_name, tldr_content, deep_content, eliv_content, sym_cost, sym_model = res
            total_cost += sym_cost
            if sym_model:
                model_used = sym_model
            generated[sym_name] = (tldr_content, deep_content, eliv_content)

    # Merge cached (to_reuse) + generated, preserving symbol order
    cached_for_merge: Dict[str, Dict[str, str]] = {}
    for name, data in to_reuse.items():
        cached_for_merge[name] = {
            "tldr": data["tldr"],
            "deep": data["deep"],
            "eliv": data["eliv"],
        }

    tldr_agg, deep_agg, eliv_agg, symbols_for_meta = _merge_symbol_content(
        symbols_to_doc, cached_for_merge, generated
    )

    # Inject relationship sections if enabled
    if include_relations:
        config = ScoutConfig()
        repo_root = Path.cwd().resolve()

        # Rebuild aggregated content with relationship sections
        tldr_parts = []
        deep_parts = []

        for symbol in symbols_to_doc:
            name = symbol.name
            if name in symbols_for_meta:
                sym_tldr = symbols_for_meta[name].get("tldr", "")
                sym_deep = symbols_for_meta[name].get("deep", "")

                # Inject relationships for this symbol
                sym_tldr = inject_relationship_sections(
                    sym_tldr,
                    symbol.name,
                    symbol.type,
                    target_path,
                    repo_root,
                    config,
                )
                sym_deep = inject_relationship_sections(
                    sym_deep,
                    symbol.name,
                    symbol.type,
                    target_path,
                    repo_root,
                    config,
                )

                # Rebuild aggregated content preserving symbol order
                header = f"# {name}\n\n"
                tldr_parts.append(header + sym_tldr)
                deep_parts.append(header + sym_deep)

        tldr_agg = "\n\n---\n\n".join(tldr_parts)
        deep_agg = "\n\n---\n\n".join(deep_parts)

    # Add hashes to symbols_for_meta for persistence
    for symbol in symbols_to_doc:
        name = symbol.name
        if name in symbols_for_meta:
            symbols_for_meta[name]["hash"] = _compute_symbol_hash(symbol, target_path)

    return (tldr_agg, deep_agg, eliv_agg, total_cost, model_used, symbols_for_meta)


async def process_single_file_async(
    target_path: Path,
    *,
    output_dir: Optional[Path] = None,
    dependencies_func: Optional[Callable[[Path], List[str]]] = None,
    per_file_concurrency: int = 3,
    language_override: Optional[str] = None,
    generate_eliv: Optional[bool] = None,
    quiet: bool = False,
    force: bool = False,
    slot_id: Optional[int] = None,
    shared_display: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    fallback_template: bool = False,
    no_fallback: bool = False,
    versioned_mirror_dir: Optional[Path] = None,
    include_relations: Optional[bool] = None,
) -> FileProcessResult:
    """
    Process a single file for documentation generation (async).

    Phase 0: freshness check — skip if up to date (unless --force).
    Phase 1: _trace_file() — pure static analysis, instant chain in dashboard.
    Phase 2: _generate_docs_for_symbols() — LLM only, updates cost in dashboard.
    """
    from .trace import _read_freshness_meta

    if not target_path.exists() or not target_path.is_file():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    # Phase 0: skip if up to date (hash-based freshness)
    if not force and _is_up_to_date(target_path, output_dir):
        if slot_id is not None and shared_display is not None:
            rel = _rel_path_for_display(target_path)
            shared_display[slot_id] = {
                "file": rel,
                "chain": None,
                "cost": 0.0,
                "status": "done",
                "success": True,
                "skipped": True,
            }
        if not quiet:
            rel = _rel_path_for_display(target_path)
            print(f"✓ {rel} (up to date)", file=sys.stdout)
        return FileProcessResult(
            success=True,
            cost_usd=0.0,
            symbols_count=0,
            calls_count=0,
            types_count=0,
            exports_count=0,
            model="",
            skipped=True,
        )

    # Resolve generate_eliv and include_relations from config if not explicitly set
    if generate_eliv is None or include_relations is None:
        config = ScoutConfig()
        if generate_eliv is None:
            doc_gen = config.get("doc_generation") or {}
            generate_eliv = doc_gen.get("generate_eliv", True)
        if include_relations is None:
            doc_rels = config.get_doc_relationships_config()
            include_relations = doc_rels.get("enabled", True)

    # Phase 1: pure static analysis — instant, no LLM
    try:
        trace = await _trace_file(
            target_path,
            language_override=language_override,
            dependencies_func=dependencies_func,
            slot_id=slot_id,
            shared_display=shared_display,
        )
    except (ValueError, SyntaxError, UnicodeDecodeError, IOError) as e:
        logger.warning("Parse error for %s: %s", target_path, e)
        if slot_id is not None and shared_display is not None:
            shared_display[slot_id] = {
                "file": _rel_path_for_display(target_path),
                "chain": None,
                "cost": 0.0,
                "status": "done",
                "success": False,
                "error": str(e),
            }
        if not quiet:
            rel = _rel_path_for_display(target_path)
            print(f"{_RED}✗ {rel}: {e}{_RESET}", file=sys.stderr)
        return FileProcessResult(
            success=False,
            cost_usd=0.0,
            symbols_count=0,
            calls_count=0,
            types_count=0,
            exports_count=0,
            model="",
            error=str(e),
        )

    symbols_to_doc = trace.symbols_to_doc
    all_calls = trace.all_calls
    all_types = trace.all_types
    all_exports = trace.all_exports

    # Phase 2: LLM doc generation — diff-aware, only changed symbols
    tldr_agg_content, deep_agg_content, eliv_agg_content, total_cost, model_used, symbols_for_meta = (
        await _generate_docs_for_symbols(
            target_path,
            trace,
            output_dir=output_dir,
            generate_eliv=generate_eliv,
            per_file_concurrency=per_file_concurrency,
            fallback_template=fallback_template,
            slot_id=slot_id,
            shared_display=shared_display,
            progress_callback=progress_callback,
            include_relations=include_relations,
        )
    )

    # TICKET-89: Validate no GAP/FALLBACK placeholders before writing
    for label, content in [
        ("tldr", tldr_agg_content),
        ("deep", deep_agg_content),
        ("eliv", eliv_agg_content),
    ]:
        if content:
            is_valid, found = validate_no_placeholders(content, str(target_path))
            if not is_valid:
                if no_fallback:
                    raise ValueError(
                        f"Placeholder(s) {found} found in {target_path} ({label}); "
                        "refusing to write (--no-fallback). Remove flag to allow placeholders."
                    )
                for marker in found:
                    logger.warning(
                        "Placeholder %s found in %s (%s); writing anyway",
                        marker,
                        target_path,
                        label,
                    )

    if not tldr_agg_content.strip() and not deep_agg_content.strip():
        msg = (
            f"No valid content to write for {target_path}: "
            "all symbols failed validation (empty LLM response or generation error)."
        )
        logger.warning(msg)
        if slot_id is not None and shared_display is not None and slot_id in shared_display:
            shared_display[slot_id]["cost"] = total_cost
            shared_display[slot_id]["status"] = "done"
            shared_display[slot_id]["success"] = False
            shared_display[slot_id]["error"] = msg
        if not quiet:
            rel = _rel_path_for_display(target_path)
            print(f"{_RED}✗ {rel}: {msg}{_RESET}", file=sys.stderr)
        return FileProcessResult(
            success=False,
            cost_usd=total_cost,
            symbols_count=len(symbols_to_doc),
            calls_count=len(all_calls),
            types_count=len(all_types),
            exports_count=len(all_exports),
            model=model_used,
            error=msg,
        )

    tldr_path, deep_path, eliv_path = write_documentation_files(
        target_path,
        tldr_agg_content,
        deep_agg_content,
        eliv_agg_content,
        output_dir,
        generate_eliv,
        versioned_mirror_dir=versioned_mirror_dir,
    )
    logger.info("Wrote %s, %s, and %s", tldr_path, deep_path, eliv_path)

    # Write freshness meta with per-symbol hashes (not mirrored to docs/livingDoc/)
    meta_path = _get_tldr_meta_path(target_path, output_dir)
    _write_freshness_meta(
        meta_path,
        _compute_source_hash(target_path),
        model_used,
        symbols=symbols_for_meta,
    )

    call_chain = _build_rolling_call_trace(symbols_to_doc)
    result = FileProcessResult(
        success=True,
        cost_usd=total_cost,
        symbols_count=len(symbols_to_doc),
        calls_count=len(all_calls),
        types_count=len(all_types),
        exports_count=len(all_exports),
        model=model_used,
        call_chain=call_chain,
    )

    if slot_id is not None and shared_display is not None and slot_id in shared_display:
        shared_display[slot_id]["cost"] = total_cost
        shared_display[slot_id]["status"] = "done"
        shared_display[slot_id]["success"] = True
        shared_display[slot_id]["chain"] = call_chain  # final chain (canonical order)
        shared_display[slot_id]["pulse_hop"] = None  # clear pulse for done

    if not quiet:
        rel = _rel_path_for_display(target_path)
        if call_chain:
            line = f"✔ {rel} ━╸ {call_chain} | {model_used} | ${total_cost:.4f}"
        else:
            line = (
                f"✔ {rel} — traced {len(all_calls)} calls, {len(all_types)} types, "
                f"{len(all_exports)} exports | {model_used} | ${total_cost:.4f}"
            )
        print(line, file=sys.stdout)

    return result


def process_single_file(
    target_path: Path,
    *,
    output_dir: Optional[Path] = None,
    dependencies_func: Optional[Callable[[Path], List[str]]] = None,
    language_override: Optional[str] = None,
    quiet: bool = False,
) -> bool:
    """
    Process a single file for documentation generation (sync wrapper).
    """
    result = asyncio.run(
        process_single_file_async(
            target_path,
            output_dir=output_dir,
            dependencies_func=dependencies_func,
            language_override=language_override,
            quiet=quiet,
        )
    )
    return result.success


def _gather_package_component_roles(package_dir: Path, repo_root: Path) -> List[str]:
    """Parse each .py file in package to extract exports and top-level calls for truth-based cascading."""
    lines: List[str] = []
    for py_path in sorted(package_dir.glob("*.py")):
        if py_path.name.startswith("_"):
            continue
        try:
            adapter = get_adapter_for_path(py_path, "python")
            root = adapter.parse(py_path)
            exports = getattr(root, "exports", None) or []
            exports_str = ", ".join(exports) if exports else "(top-level defs/classes)"
            all_calls: List[str] = []
            for child in root.children:
                calls = getattr(child, "calls", None) or []
                all_calls.extend(calls)
            seen: set = set()
            unique_calls = [c for c in all_calls if c not in seen and not seen.add(c)]
            calls_str = ", ".join(unique_calls[:12]) if unique_calls else "(none traced)"
            lines.append(f"- {py_path.name}: Exports {exports_str}. Calls: {calls_str}")
        except Exception as e:
            logger.debug("Skip tracing %s: %s", py_path, e)
    return lines


async def _update_module_brief_async(package_dir: Path, repo_root: Path) -> bool:
    """
    Generate module-level brief (__init__.py.module.md) from real component roles + child .tldr.md.

    Truth-based cascading: uses traced exports/calls from each file, then synthesizes orchestration.
    Async: uses await call_groq_async (no asyncio.run nesting).
    """
    config = ScoutConfig()
    drafts = config.get("drafts") or {}
    if not drafts.get("enable_module_briefs", True):
        return False

    ignore = IgnorePatterns(repo_root=repo_root)
    init_py = package_dir / "__init__.py"
    if not init_py.exists():
        return False

    try:
        rel = package_dir.relative_to(repo_root)
    except ValueError:
        return False

    if ignore.matches(package_dir, repo_root):
        return False

    docs_dir = package_dir / ".docs"
    if not docs_dir.exists():
        return False

    component_roles = _gather_package_component_roles(package_dir, repo_root)
    roles_block = "\n".join(component_roles) if component_roles else "(no traced components)"

    tldr_parts: List[str] = []
    for md in sorted(docs_dir.glob("*.tldr.md")):
        try:
            tldr_parts.append(f"### {md.stem}\n\n{md.read_text(encoding='utf-8', errors='replace')}")
        except OSError:
            pass

    if not tldr_parts:
        return False

    combined = "\n\n".join(tldr_parts)
    if len(combined) > 8000:
        combined = combined[:8000] + "\n\n[... truncated ...]"

    module_name = ".".join(rel.parts)
    prompt = f"""Synthesize a package overview from REAL component roles. Do not guess.

Package: {module_name}
Path: {rel}

Component roles (traced from call graph):
{roles_block}

Child summaries (.tldr.md) for context:
{combined}

Output a concise Markdown overview with ## headings:
1. ## Purpose — What this package does (one sentence).
2. ## Components — Ingress/Processing/Egress: which file handles what (from traced roles).
3. ## Key Invariants — Constraints from the code (e.g. plain-text Git workflow, no external deps).
Use only facts from the traced roles and summaries. No speculation."""

    system = "You are a documentation assistant. Be concise and accurate."
    use_big_brain = bool(os.environ.get("GEMINI_API_KEY"))
    try:
        if use_big_brain:
            response = await call_big_brain_async(
                prompt,
                system=system,
                max_tokens=800,
                task_type="module_brief",
            )
        else:
            content, metadata, attempts = await generate_with_quality_loop(
                prompt,
                system=system,
                task_type="synthesis",
                max_tokens=800,
            )
    except Exception as e:
        logger.warning("Module brief LLM failed for %s: %s", package_dir, e)
        return False

    if not use_big_brain:
        audit = AuditLog()
        audit.log(
            "module_brief",
            cost=metadata.get("cost", 0),
            model=metadata.get("model", "synthesis"),
            package=str(rel),
        )

    content = content.strip()
    module_md_name = "__init__.py.module.md"
    local_path = docs_dir / module_md_name
    central_dir = repo_root / "docs" / "livingDoc" / rel
    central_dir.mkdir(parents=True, exist_ok=True)
    central_path = central_dir / module_md_name
    try:
        local_path.write_text(content, encoding="utf-8")
        central_path.write_text(content, encoding="utf-8")
        logger.info("Wrote module brief: %s (mirrored to %s)", local_path, central_path)
    except OSError as e:
        logger.warning("Could not write module brief for %s: %s", package_dir, e)
        return False
    return True


def _update_module_brief(package_dir: Path, repo_root: Path, *, is_async: bool = False):
    """
    Sync wrapper for _update_module_brief_async.
    is_async=False (default): uses asyncio.run for CLI/sync callers.
    is_async=True: returns coroutine for caller to await.
    """
    if is_async:
        return _update_module_brief_async(package_dir, repo_root)
    return asyncio.run(_update_module_brief_async(package_dir, repo_root))


async def _process_file_with_semaphore(
    semaphore: asyncio.Semaphore,
    file_path: Path,
    *,
    output_dir: Optional[Path] = None,
    dependencies_func: Optional[Callable[[Path], List[str]]] = None,
    language_override: Optional[str] = None,
    generate_eliv: Optional[bool] = None,
    quiet: bool = False,
    force: bool = False,
    slot_queue: Optional[asyncio.Queue] = None,
    shared_display: Optional[Dict[str, Any]] = None,
    fallback_template: bool = False,
    no_fallback: bool = False,
    versioned_mirror_dir: Optional[Path] = None,
    include_relations: Optional[bool] = None,
) -> FileProcessResult:
    """Process a single file with semaphore for concurrency control."""
    async with semaphore:
        slot_id = None
        if slot_queue is not None:
            slot_id = await slot_queue.get()
        try:
            def _progress_cb(cost: float) -> None:
                if slot_id is not None and shared_display is not None and slot_id in shared_display:
                    shared_display[slot_id]["cost"] = cost

            include_relations_val = include_relations
            return await process_single_file_async(
                file_path,
                output_dir=output_dir,
                dependencies_func=dependencies_func,
                language_override=language_override,
                generate_eliv=generate_eliv,
                quiet=quiet,
                force=force,
                slot_id=slot_id,
                shared_display=shared_display,
                progress_callback=_progress_cb,
                fallback_template=fallback_template,
                no_fallback=no_fallback,
                versioned_mirror_dir=versioned_mirror_dir,
                include_relations=include_relations_val,
            )
        finally:
            if slot_queue is not None and slot_id is not None:
                slot_queue.put_nowait(slot_id)


def _format_status_bar(
    completed: int,
    total: int,
    last_file: Optional[str],
    last_calls: int,
    last_cost: float,
    total_cost: float,
    processed: int = 0,
) -> str:
    """Build status bar: [✓ 24/114] file → traced N calls, $X.XXXX | Est. total: $X.XX"""
    denom = processed if processed > 0 else completed
    est_total = total_cost * total / denom if denom > 0 else 0.0
    file_part = f" {last_file} →" if last_file else ""
    stats = f"traced {last_calls} calls, ${last_cost:.4f}" if last_file else "…"
    return f"[✓ {completed}/{total}]{file_part} {stats} | Est. total: ${est_total:.2f}"


async def process_directory_async(
    target_path: Path,
    *,
    recursive: bool = False,
    output_dir: Optional[Path] = None,
    dependencies_func: Optional[Callable[[Path], List[str]]] = None,
    language_override: Optional[str] = None,
    workers: Optional[int] = None,
    show_progress: bool = True,
    generate_eliv: Optional[bool] = None,
    quiet: bool = False,
    budget: Optional[float] = None,
    force: bool = False,
    changed_files: Optional[List[Path]] = None,
    fallback_template: bool = False,
    no_fallback: bool = False,
    versioned_mirror_dir: Optional[Path] = None,
    include_relations: Optional[bool] = None,
) -> None:
    """
    Process a directory of files for documentation generation (async).

    Uses asyncio.Semaphore to limit concurrent LLM calls. Preserves idempotency.
    """
    import os

    if not target_path.exists() or not target_path.is_dir():
        raise NotADirectoryError(f"Target directory not found: {target_path}")

    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-set workers and LLM concurrency from model RPM if not explicitly provided
    specs = get_model_specs()
    model_name = _resolve_doc_model("deep")
    spec = specs.get(model_name, {})
    rpm = spec.get("rpm_limit", 1000)

    if workers is None:
        workers = _safe_workers_from_rpm(model_name, rpm)
        logger.info(
            "Auto-set workers to %d based on %s RPM limit (%d)",
            workers,
            model_name,
            rpm,
        )
    workers = max(1, workers)
    semaphore = asyncio.Semaphore(workers)

    files: List[Path] = []
    for pattern in _DIRECTORY_PATTERNS:
        for f in target_path.glob(pattern):
            if f.is_file() and "__pycache__" not in str(f):
                files.append(f)

    if changed_files is not None:
        changed_set = {p.resolve() for p in changed_files}
        files = [f for f in files if f.resolve() in changed_set]
        if not files:
            logger.info("No changed files to process (--changed-only)")
            return

    total = len(files)

    # Doc-sync: set LLM concurrency just below rate limit for bulk runs.
    if (
        total > 1
        and "SCOUT_MAX_CONCURRENT_CALLS" not in os.environ
    ):
        desired = _max_concurrent_from_rpm(rpm)
        os.environ["SCOUT_MAX_CONCURRENT_CALLS"] = str(desired)
        # Bump workers so file-level parallelism doesn't bottleneck LLM throughput
        workers = min(16, max(workers, desired // 2))
        semaphore = asyncio.Semaphore(workers)
        logger.info(
            "Doc-sync: set SCOUT_MAX_CONCURRENT_CALLS=%d, workers=%d (just below %d RPM limit)",
            desired,
            workers,
            rpm,
        )

    if generate_eliv is None:
        config = ScoutConfig()
        doc_gen = config.get("doc_generation") or {}
        generate_eliv = doc_gen.get("generate_eliv", True)

    # Resolve include_relations from config if not explicitly set
    if include_relations is None:
        config = ScoutConfig()
        doc_rels = config.get_doc_relationships_config()
        include_relations = doc_rels.get("enabled", True)

    repo_root = Path.cwd().resolve()
    use_progress = (
        show_progress and not quiet and total > 1 and sys.stdout.isatty()
    )
    use_dashboard = use_progress  # full parallel trace dashboard
    processed_package_dirs: set = set()
    completed = [0]
    processed = [0]
    total_cost = [0.0]
    last_result: List[Optional[FileProcessResult]] = [None]
    last_file: List[Optional[str]] = [None]
    lock = asyncio.Lock()

    slot_queue: Optional[asyncio.Queue] = None
    shared_display: Dict[int, Dict[str, Any]] = {}
    dashboard_done = asyncio.Event()
    if use_dashboard:
        slot_queue = asyncio.Queue()
        for i in range(workers):
            slot_queue.put_nowait(i)

    def _apply_pulse(chain: str, pulse_hop: Optional[str]) -> str:
        """Wrap pulse_hop in inverse video for 1 frame, then clear."""
        if not pulse_hop or pulse_hop not in chain:
            return chain
        plain = _strip_ansi(pulse_hop)
        return chain.replace(
            pulse_hop,
            f"{_INVERSE}{plain}{_INVERSE_OFF}",
            1,
        )

    def _strip_ansi(s: str) -> str:
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

    async def _display_refresh_task() -> None:
        """Refresh dashboard every 100ms (10 FPS)."""
        while not dashboard_done.is_set():
            await asyncio.sleep(0.1)
            if dashboard_done.is_set():
                break
            async with lock:
                display_completed = completed[0]
                display_processed = processed[0]
                display_total_cost = total_cost[0]
            sys.stdout.write(_CLEAR_SCREEN)
            est_total = (
                display_total_cost * total / display_processed
                if display_processed > 0
                else 0.0
            )
            for slot_id in range(workers):
                entry = shared_display.get(slot_id)
                if entry:
                    status = entry.get("status", "running")
                    file_str = entry.get("file", "…")
                    chain = entry.get("chain")
                    cost = entry.get("cost", 0.0)
                    pulse_hop = entry.pop("pulse_hop", None)
                    if chain and pulse_hop:
                        chain = _apply_pulse(chain, pulse_hop)
                    slot_prefix = f"[{slot_id}] "
                    if status == "done":
                        if entry.get("skipped"):
                            line = f"{slot_prefix}✓ {file_str} (up to date)"
                        else:
                            success = entry.get("success", False)
                            if success and chain:
                                line = f"{slot_prefix}✔ {file_str} ━╸ {chain} | ${cost:.4f}"
                            elif success:
                                line = f"{slot_prefix}✔ {file_str} | ${cost:.4f}"
                            else:
                                err = entry.get("error", "failed")
                                line = f"{slot_prefix}{_RED}✗ {file_str}: {err}{_RESET}"
                        sys.stdout.write(line + "\n")
                    else:
                        sym_done = entry.get("symbols_done", 0)
                        sym_total = entry.get("symbols_total", 0)
                        sym_progress = (
                            f" [{sym_done}/{sym_total}]" if sym_total > 1 else ""
                        )
                        if chain:
                            line = f"{slot_prefix}{file_str}{sym_progress} ━╸ {chain} | ${cost:.4f}"
                        else:
                            line = f"{slot_prefix}{file_str}{sym_progress} | ${cost:.4f}"
                        sys.stdout.write(line + "\n")
            bar = f"[✓ {display_completed}/{total}] | ${display_total_cost:.4f} spent | Est. total: ${est_total:.2f}"
            sys.stdout.write("\n" + bar)
            sys.stdout.flush()

    def _render_status() -> None:
        if not use_progress or use_dashboard:
            return
        r = last_result[0]
        bar = _format_status_bar(
            completed[0],
            total,
            last_file[0],
            r.calls_count if r else 0,
            r.cost_usd if r else 0.0,
            total_cost[0],
            processed=processed[0],
        )
        sys.stdout.write("\r" + bar)
        sys.stdout.flush()

    async def _process_and_track(f: Path) -> None:
        result: Optional[FileProcessResult] = None
        try:
            result = await _process_file_with_semaphore(
                semaphore,
                f,
                output_dir=output_dir,
                dependencies_func=dependencies_func,
                language_override=language_override,
                generate_eliv=generate_eliv,
                quiet=True,
                force=force,
                slot_queue=slot_queue,
                shared_display=shared_display,
                fallback_template=fallback_template,
                no_fallback=no_fallback,
                versioned_mirror_dir=versioned_mirror_dir,
                include_relations=include_relations,
            )
        except (ValueError, OSError) as e:
            logger.warning("Skip %s: %s", f, e)
            result = FileProcessResult(
                success=False,
                cost_usd=0.0,
                symbols_count=0,
                calls_count=0,
                types_count=0,
                exports_count=0,
                model="",
                error=str(e),
            )
        finally:
            async with lock:
                completed[0] += 1
                if result:
                    if not result.skipped:
                        total_cost[0] += result.cost_usd
                        processed[0] += 1
                        if budget is not None and total_cost[0] > budget:
                            raise BudgetExceededError(total_cost[0], budget)
                    last_result[0] = result
                    last_file[0] = _rel_path_for_display(f)
                    if result.success and output_dir is None and (f.parent / "__init__.py").exists():
                        processed_package_dirs.add(f.parent)

            if quiet:
                return
            if use_dashboard:
                return

            if result:
                rel = _rel_path_for_display(f)
                if result.skipped:
                    if use_progress:
                        sys.stdout.write("\r" + " " * 100 + "\r")
                    print(f"✓ {rel} (up to date)", file=sys.stdout)
                elif result.success:
                    if result.call_chain:
                        line = f"✔ {rel} ━╸ {result.call_chain} | {result.model} | ${result.cost_usd:.4f}"
                    else:
                        line = (
                            f"✔ {rel} — traced {result.calls_count} calls, "
                            f"{result.types_count} types, {result.exports_count} exports | "
                            f"{result.model} | ${result.cost_usd:.4f}"
                        )
                    if use_progress:
                        sys.stdout.write("\r" + " " * 100 + "\r")
                    print(line, file=sys.stdout)
                else:
                    err_msg = f"{_RED}✗ {rel}: {result.error}{_RESET}"
                    if use_progress:
                        sys.stdout.write("\r" + " " * 100 + "\r")
                    print(err_msg, file=sys.stderr)

            if use_progress and not use_dashboard:
                _render_status()

    if use_dashboard:
        refresh_task = asyncio.create_task(_display_refresh_task())

    task_objs = [asyncio.create_task(_process_and_track(f)) for f in files]
    try:
        await asyncio.gather(*task_objs)
    except BudgetExceededError:
        for t in task_objs:
            t.cancel()
        await asyncio.gather(*task_objs, return_exceptions=True)
        raise

    if use_dashboard:
        dashboard_done.set()
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass
        sys.stdout.write("\n")
        sys.stdout.flush()
    elif use_progress:
        sys.stdout.write("\r" + " " * 100 + "\r")
        sys.stdout.write("\n")
        sys.stdout.flush()

    if output_dir is None:
        for pkg_dir in sorted(processed_package_dirs):
            try:
                await _update_module_brief_async(pkg_dir, repo_root)
            except Exception as e:
                logger.warning("Skip module brief for %s: %s", pkg_dir, e)


def process_directory(
    target_path: Path,
    *,
    recursive: bool = False,
    output_dir: Optional[Path] = None,
    dependencies_func: Optional[Callable[[Path], List[str]]] = None,
    language_override: Optional[str] = None,
    workers: Optional[int] = None,
    show_progress: bool = True,
    generate_eliv: Optional[bool] = None,
    quiet: bool = False,
    budget: Optional[float] = None,
    force: bool = False,
    changed_files: Optional[List[Path]] = None,
    fallback_template: bool = False,
    no_fallback: bool = False,
    versioned_mirror_dir: Optional[Path] = None,
    include_relations: Optional[bool] = None,
) -> None:
    """Process a directory of files for documentation generation."""
    asyncio.run(
        process_directory_async(
            target_path,
            recursive=recursive,
            output_dir=output_dir,
            dependencies_func=dependencies_func,
            language_override=language_override,
            workers=workers,
            show_progress=show_progress,
            generate_eliv=generate_eliv,
            quiet=quiet,
            budget=budget,
            force=force,
            changed_files=changed_files,
            fallback_template=fallback_template,
            no_fallback=no_fallback,
            versioned_mirror_dir=versioned_mirror_dir,
            include_relations=include_relations,
        )
    )
