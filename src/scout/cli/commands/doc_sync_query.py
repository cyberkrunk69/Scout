#!/usr/bin/env python
"""
doc_sync_query command - Generate docs for files relevant to a query.

Uses multi-layer retrieval to find the right files, then generates docs.
Integrates with trust subsystem to immediately mark new docs as trusted.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Set

from scout.audit import AuditLog
from scout.cli.formatting.progress_stub import PlanProgress
from scout.llm.dispatch import call_llm_async
from scout.trust.models import TrustRecord
from scout.trust.orchestrator import TrustOrchestrator

logger = logging.getLogger(__name__)

# Cost estimates (TODO: move to ScoutConfig when pricing model stabilizes)
COST_PER_FILE_ESTIMATE = 0.02  # Approximate cost per doc generation
COST_DEEP_MODE_ESTIMATE = 0.001  # Approximate cost for LLM refinement

# Limits
MAX_LLM_SUGGESTIONS = 5  # Cap LLM suggestions to prevent explosion

# Output directory
DEFAULT_DOC_OUTPUT_DIR = "docs/livingDoc"


async def run(query: str, top_k: int = 10, confidence: int = 50, deep: bool = False) -> int:
    """
    Entry point for CLI dispatch.

    Returns exit code (0 for success, non-zero for failure).
    """
    repo_root = Path.cwd().resolve()
    return await _run(repo_root, query, top_k, confidence, deep)


async def _run(
    repo_root: Path,
    query: str,
    top_k: int,
    confidence: int,
    deep: bool,
) -> int:
    """Main implementation."""
    progress = PlanProgress()
    progress.start(f"Finding files relevant to: {query}")

    # Step 1: BM25 retrieval
    nav_results = await _query_bm25(repo_root, query, top_k)
    candidates: Set[Path] = set()

    for r in nav_results:
        if r.get("confidence", 0) >= confidence:
            file_path = repo_root / r["target_file"]
            if file_path.exists():
                candidates.add(file_path)

    if not candidates:
        progress.warning("No files found with sufficient confidence.")
        return 0

    progress.info(f"Found {len(candidates)} files via BM25.")

    # Step 2: Import graph expansion using AST
    expanded = await _expand_with_imports(candidates, repo_root, progress)
    progress.info(f"Expanded to {len(expanded)} files including imports.")

    # Step 3: Deep mode - LLM refinement
    if deep:
        llm_suggestions = await _llm_suggest_files(query, expanded, repo_root, progress)
        for f in llm_suggestions:
            if f.exists():
                expanded.add(f)
        progress.info(f"Deep mode added {len(llm_suggestions)} more files.")

    # Step 4: Generate docs in parallel
    progress.start(f"Generating docs for {len(expanded)} files")

    orchestrator = TrustOrchestrator(repo_root)
    await orchestrator.initialize()

    tasks = [_generate_and_verify(file, orchestrator, repo_root, progress) for file in expanded]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = []
    failures = []

    # Handle results and log individual failures
    expanded_list = list(expanded)
    for i, result in enumerate(results):
        if i < len(expanded_list):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate doc for {expanded_list[i]}: {result}")
                failures.append(result)
            else:
                successes.append(result)

    # Step 5: Audit logging
    total_cost = (
        len(successes) * COST_PER_FILE_ESTIMATE
        + (COST_DEEP_MODE_ESTIMATE if deep else 0)
    )
    audit = AuditLog()
    audit.log(
        "doc_sync",
        cost=total_cost,
        files=[str(f) for f in successes],
        reason=(
            f"query='{query}', files_generated={len(successes)}, "
            f"files_failed={len(failures)}, deep={deep}"
        ),
    )

    progress.complete(
        f"Generated {len(successes)} docs, {len(failures)} failed. "
        f"Cost: ${total_cost:.4f}"
    )

    return 0 if not failures else 1


async def _query_bm25(
    repo_root: Path,
    query: str,
    limit: int,
) -> List[dict]:
    """Simple BM25-like search using file index.
    
    This is a simplified version that doesn't require the legacy cli/index.py import.
    For full BM25 functionality, the index must be built first.
    """
    # Simple file-based search as fallback
    # This will be improved in future iterations
    import os
    results = []
    
    # Search in docs directory
    docs_dir = repo_root / "docs"
    if docs_dir.exists():
        for root, dirs, files in os.walk(docs_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for f in files:
                if f.endswith(('.md', '.py')):
                    file_path = Path(root) / f
                    # Simple relevance check
                    try:
                        content = file_path.read_text(errors='ignore').lower()
                        if query.lower() in content:
                            results.append({
                                'target_file': str(file_path.relative_to(repo_root)),
                                'confidence': 0.5,
                            })
                    except Exception:
                        pass
    
    # Return top results limited by limit
    return results[:limit]


async def _expand_with_imports(
    files: Set[Path],
    repo_root: Path,
    progress: PlanProgress,
) -> Set[Path]:
    """Find files imported by the candidates using AST extraction."""
    expanded = set(files)

    for file in files:
        try:
            # AST parsing is CPU-bound - run in thread pool
            loop = asyncio.get_event_loop()
            imported_paths = await loop.run_in_executor(
                None, extract_imports_from_file, file
            )

            for mod_path in imported_paths:
                full_path = repo_root / mod_path
                if full_path.exists() and full_path.is_file():
                    expanded.add(full_path)
        except Exception as e:
            logger.debug(f"Failed to extract imports from {file}: {e}")

    return expanded


def extract_imports_from_file(file_path: Path) -> Set[str]:
    """
    Extract all imported modules from a Python file using AST.

    Handles:
    - Absolute imports: import foo, import foo.bar
    - From imports: from foo import bar, from foo.bar import baz
    - Aliases: import foo as bar

    Note: Relative imports (from . import x) not yet resolved.
          More sophisticated resolution could check both .py and __init__.py.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(file_path))
    except (SyntaxError, FileNotFoundError):
        return set()

    imports: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_path = _module_to_file_path(alias.name)
                if module_path:
                    imports.add(module_path)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_path = _module_to_file_path(node.module)
                if module_path:
                    imports.add(module_path)
            # Relative imports: TODO - resolve relative to package root

    return imports


def _module_to_file_path(module_name: str) -> str:
    """Convert Python module name to filesystem path."""
    return module_name.replace(".", "/") + ".py"


async def _llm_suggest_files(
    query: str,
    current_files: Set[Path],
    repo_root: Path,
    progress: PlanProgress,
) -> List[Path]:
    """Use a cheap LLM to suggest additional relevant files."""
    file_list = "\n".join(str(f.relative_to(repo_root)) for f in current_files)
    prompt = (
        f"Given the query: \"{query}\"\n"
        f"And these files already identified as relevant:\n"
        f"{file_list}\n\n"
        "What other files in the codebase might be important for documentation? "
        "List only the file paths, one per line, with no extra text. Be specific."
    )

    system = "You are a helpful assistant that knows Python codebases."

    try:
        response, cost = await call_llm_async(
            model="llama-3.1-8b-instant",
            prompt=prompt,
            system=system,
            max_tokens=300,
        )

        # Parse response for file paths
        lines = response.strip().split("\n")
        paths = []
        for line in lines:
            line = line.strip()
            if line.endswith(".py"):
                full = repo_root / line
                if full.exists():
                    paths.append(full)

        return paths[:MAX_LLM_SUGGESTIONS]
    except Exception as e:
        logger.warning(f"LLM suggestion failed: {e}")
        return []


async def _generate_and_verify(
    file: Path,
    orchestrator: TrustOrchestrator,
    repo_root: Path,
    progress: PlanProgress,
) -> Path:
    """Generate doc for a file and update trust store."""
    try:
        progress.spin(f"Generating {file.name}")

        # Lazy import to avoid circular import issues with doc_generation package
        from scout.doc_generation import process_single_file_async

        await process_single_file_async(
            target_path=file,
            output_dir=repo_root / DEFAULT_DOC_OUTPUT_DIR,
            force=True,
            quiet=True,
        )

        # Verify to get trust info
        trust_result = await orchestrator.verify(file)

        # Update store with new trust info
        record = TrustRecord(
            source_path=str(file),
            doc_path=str(trust_result.verification.doc_path),
            trust_level=trust_result.verification.trust_level.value,
            embedded_checksum=trust_result.verification.embedded_checksum,
            current_checksum=trust_result.verification.current_checksum,
            stale_symbols=trust_result.verification.stale_symbols,
            fresh_symbols=trust_result.verification.fresh_symbols,
            penalty=trust_result.penalty,
            last_validated=datetime.utcnow().isoformat(),
        )
        await orchestrator.store.upsert(record)

        return file
    except Exception as e:
        logger.exception(f"Failed to generate doc for {file}")
        raise e


if __name__ == "__main__":
    asyncio.run(run("test query"))
