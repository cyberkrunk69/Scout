#!/usr/bin/env python
"""
File context injection - read files for context.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional


class FileContext:
    """Read files for context injection."""

    # Approximate token to character ratio
    TOKENS_TO_CHARS = 4

    def __init__(self, max_tokens: int = 40000):
        self.max_tokens = max_tokens
        self.max_chars = max_tokens * self.TOKENS_TO_CHARS

    def read_files(self, patterns: list[str]) -> str:
        """
        Read files matching patterns, return combined context.

        Args:
            patterns: List of file patterns (glob patterns or paths)

        Returns:
            Combined file contents with headers
        """
        files = []
        for pattern in patterns:
            # Handle glob patterns
            if "*" in pattern or "?" in pattern:
                files.extend(glob.glob(pattern, recursive=True))
            else:
                # Single file or directory
                path = Path(pattern)
                if path.is_file():
                    files.append(pattern)
                elif path.is_dir():
                    # Read all Python files in directory
                    for ext in ["*.py", "*.md", "*.txt"]:
                        files.extend(glob.glob(f"{pattern}/{ext}", recursive=True))

        # Deduplicate and read contents
        context_parts = []
        total_chars = 0

        for filepath in sorted(set(files)):
            try:
                path = Path(filepath)
                if not path.exists():
                    continue

                content = path.read_text(encoding="utf-8", errors="replace")
                header = f"\n--- {filepath} ---\n"
                chunk = header + content

                if total_chars + len(chunk) > self.max_chars:
                    # Truncate this file if it fits
                    remaining = self.max_chars - total_chars
                    if remaining > 100:  # At least some content
                        chunk = header + content[:remaining]
                        context_parts.append(chunk)
                    break

                context_parts.append(chunk)
                total_chars += len(chunk)

            except Exception as e:
                # Skip files that can't be read
                continue

        return "".join(context_parts)

    def read_file(self, filepath: str, max_lines: Optional[int] = None) -> str:
        """
        Read a single file with optional line limit.

        Args:
            filepath: Path to file
            max_lines: Maximum number of lines to read

        Returns:
            File contents
        """
        path = Path(filepath)
        if not path.exists():
            return f"Error: File not found: {filepath}"

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if max_lines:
                lines = content.split("\n")
                content = "\n".join(lines[:max_lines])
                if len(lines) > max_lines:
                    content += f"\n... ({len(lines) - max_lines} more lines)"
            return content
        except Exception as e:
            return f"Error reading {filepath}: {e}"

    def get_file_info(self, filepath: str) -> dict:
        """
        Get file information.

        Args:
            filepath: Path to file

        Returns:
            Dictionary with file info
        """
        path = Path(filepath)
        if not path.exists():
            return {"error": "File not found"}

        stat = path.stat()
        return {
            "path": str(path),
            "name": path.name,
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "is_file": path.is_file(),
            "is_dir": path.is_dir(),
            "extension": path.suffix,
        }

    def expand_glob(self, pattern: str) -> list[str]:
        """
        Expand a glob pattern to list of files.

        Args:
            pattern: Glob pattern

        Returns:
            List of matching file paths
        """
        return glob.glob(pattern, recursive=True)


def read_from_stdin() -> Optional[str]:
    """Read content from stdin if available (for pipe support)."""
    import sys

    if not sys.stdin.isatty():
        return sys.stdin.read()
    return None


def combine_context(args: list[str], max_tokens: int = 40000) -> str:
    """
    Combine file context from arguments and stdin.

    Args:
        args: Command line arguments (file paths/patterns)
        max_tokens: Maximum tokens for context

    Returns:
        Combined context string
    """
    fc = FileContext(max_tokens=max_tokens)
    parts = []

    # Read from files/patterns
    if args:
        file_context = fc.read_files(args)
        if file_context:
            parts.append(file_context)

    # Read from stdin (pipe)
    stdin_content = read_from_stdin()
    if stdin_content:
        parts.append(f"\n--- stdin ---\n{stdin_content}")

    return "\n".join(parts)


if __name__ == "__main__":
    import sys
    fc = FileContext()
    if len(sys.argv) > 1:
        print(fc.read_files(sys.argv[1:]))
    else:
        print("Usage: python -m context.files <file1> <file2> ...")
