"""Stub module for query_for_nav - ported from scout."""
from pathlib import Path
from typing import Any


def query_for_nav(
    repo_root: Path,
    query: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Stub implementation - returns empty results.
    
    Original from scout.cli.index.query_for_nav
    """
    return []
