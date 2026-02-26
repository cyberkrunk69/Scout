"""
Summarization utilities for Scout CLI tools.

Provides intelligent summarization of large outputs, with support for:
- Simple truncation (head + tail) for quick summaries
- LLM-based summarization (future extension via MiniMax)

Usage:
    from vivarium.scout.utils.summarize import summarize

    # Simple truncation (default)
    summary = summarize(large_text, max_length=500)

    # LLM-based summarization (when available)
    summary = summarize(large_text, use_llm=True)
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypedDict

# Default configuration
DEFAULT_MAX_LENGTH = 500
DEFAULT_HEAD_LINES = 20
DEFAULT_TAIL_LINES = 10

# Thresholds for smart summarization
SMALL_OUTPUT_THRESHOLD = 500  # chars - return full
MEDIUM_OUTPUT_THRESHOLD = 2000  # chars - return summary
LARGE_OUTPUT_THRESHOLD = 10000  # chars - return compressed or temp file


class OutputMode(Enum):
    """Output mode for summarization decisions."""

    FULL = "full"
    SUMMARY = "summary"
    COMPRESSED = "compressed"  # e.g., diff only shows changed function names


class Context(TypedDict, total=False):
    """Context information for making summarization decisions."""

    command: str  # e.g., "run", "git diff", "audit"
    output_size: int  # character count
    line_count: int  # line count
    user_intent: str  # "interactive", "agent", "api", "log"
    flags: dict  # parsed CLI flags
    output_type: str  # "text", "json", "structured"


@dataclass
class SummarizationResult:
    """Result from maybe_summarize() with output and metadata."""

    output: str
    mode: OutputMode
    was_summarized: bool
    metadata: dict = field(default_factory=dict)


# Command-specific handling
MASSIVE_OUTPUT_COMMANDS: dict[str, OutputMode] = {
    "git diff": OutputMode.COMPRESSED,
    "git log": OutputMode.COMPRESSED,
    "git show": OutputMode.COMPRESSED,
    "run": OutputMode.SUMMARY,
    "audit": OutputMode.SUMMARY,
}

# User intent presets
INTENT_DEFAULTS: dict[str, OutputMode] = {
    "interactive": OutputMode.SUMMARY,  # Terminal user
    "agent": OutputMode.FULL,  # AI agent needs full
    "api": OutputMode.FULL,  # API consumer expects full
    "log": OutputMode.COMPRESSED,  # Logging needs minimal
}


@dataclass
class SummarizeConfig:
    """Configuration for summarization behavior."""

    max_length: int = DEFAULT_MAX_LENGTH
    head_lines: int = DEFAULT_HEAD_LINES
    tail_lines: int = DEFAULT_TAIL_LINES
    use_llm: bool = False  # Future: enable LLM summarization


def _truncate_text(text: str, max_length: int = DEFAULT_MAX_LENGTH) -> str:
    """
    Simple truncation that preserves beginning and end of text.

    Returns a balanced sample: first N lines + last M lines,
    with a note about omitted content if needed.
    """
    if not text:
        return ""

    lines = text.split("\n")

    # If text fits within max_length, return as-is
    if len(text) <= max_length:
        return text

    # Calculate how many chars to take from start vs end
    # We want head_lines from start and tail_lines from end
    head_lines = DEFAULT_HEAD_LINES
    tail_lines = DEFAULT_TAIL_LINES

    head = "\n".join(lines[:head_lines])
    tail = "\n".join(lines[-tail_lines:])

    # Build summary with placeholder
    omitted_lines = len(lines) - head_lines - tail_lines
    if omitted_lines > 0:
        placeholder = f"\n... {omitted_lines} more lines omitted ...\n"
    else:
        placeholder = ""

    return head + placeholder + tail


def _count_summary(text: str) -> str:
    """
    Generate a simple count-based summary.

    Returns counts of lines, characters, and provides a brief description.
    """
    if not text:
        return "Empty output"

    lines = text.split("\n")
    line_count = len(lines)
    char_count = len(text)
    word_count = len(text.split())

    # Generate a brief description
    description = f"Output: {line_count} lines, {word_count} words, {char_count} chars"

    # Add first line preview if available
    first_line = lines[0].strip() if lines else ""
    if first_line:
        description += f'\nFirst line: "{first_line[:80]}..."'

    return description


def summarize(
    text: str,
    max_length: int = DEFAULT_MAX_LENGTH,
    use_llm: bool = False,
    config: Optional[SummarizeConfig] = None,
) -> str:
    """
    Summarize large text output.

    Args:
        text: The text to summarize
        max_length: Maximum length of summary (characters)
        use_llm: Whether to use LLM for summarization (future feature)
        config: Optional SummarizeConfig for detailed settings

    Returns:
        A summarized version of the text
    """
    if not text:
        return ""

    # Use config if provided
    if config:
        max_length = config.max_length
        use_llm = config.use_llm

    # For empty or short text, return as-is
    if len(text) <= max_length:
        return text

    # If LLM is requested (future feature)
    if use_llm:
        # Placeholder for future MiniMax integration
        # For now, fall back to truncation
        pass

    # Use truncation with head+tail pattern
    return _truncate_text(text, max_length)


def compress_git_diff(diff_text: str) -> str:
    """
    Return only changed file/function names, not full diff.
    Useful for high-level overviews.

    Args:
        diff_text: Full git diff output

    Returns:
        Compressed representation with file names and stats
    """
    if not diff_text:
        return "(empty diff)"

    lines = diff_text.split("\n")
    files: list[str] = []
    stats: dict = {"files": 0, "additions": 0, "deletions": 0}

    current_file = ""
    for line in lines:
        # Track file changes
        if line.startswith("diff --git"):
            stats["files"] += 1
            # Extract file name from "diff --git a/path b/path"
            parts = line.split()
            if len(parts) >= 3:
                # Get the "b/" version
                current_file = parts[-1].replace("b/", "")
                files.append(current_file)

        # Count additions/deletions
        elif line.startswith("+") and not line.startswith("+++"):
            stats["additions"] += 1
        elif line.startswith("-") and not line.startswith("---"):
            stats["deletions"] += 1

    # Build compressed output
    result_lines = [
        f"{stats['files']} file(s) changed, +{stats['additions']} -{stats['deletions']}",
        "",
    ]

    # Add file list (limited to avoid overwhelming output)
    if files:
        result_lines.append("Changed files:")
        for f in files[:20]:  # Limit to 20 files
            result_lines.append(f"  - {f}")
        if len(files) > 20:
            result_lines.append(f"  ... and {len(files) - 20} more")

    return "\n".join(result_lines)


def compress_large_output(text: str, max_lines: int = 50) -> str:
    """
    Compress large output by extracting key information.

    Args:
        text: The large text output
        max_lines: Maximum number of lines to include

    Returns:
        Compressed version of the output
    """
    if not text:
        return ""

    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text

    # Take first and last N lines
    head_count = max_lines // 2
    tail_count = max_lines - head_count

    head = "\n".join(lines[:head_count])
    tail = "\n".join(lines[-tail_count:])
    omitted = len(lines) - head_count - tail_count

    return f"{head}\n\n... {omitted} more lines ...\n\n{tail}"


def maybe_summarize(
    output: str,
    context: Context,
) -> SummarizationResult:
    """
    Decide whether to return full, summary, or compressed output.

    Args:
        output: The text output to potentially summarize
        context: Context information for making the decision

    Returns:
        SummarizationResult with the final output and metadata
    """
    # Handle empty output
    if not output:
        return SummarizationResult(
            output="",
            mode=OutputMode.FULL,
            was_summarized=False,
            metadata={"original_size": 0, "truncated_size": 0},
        )

    original_size = len(output)
    line_count = output.count("\n") + 1

    # Extract context values with defaults
    command = context.get("command", "generic")
    user_intent = context.get("user_intent", "interactive")
    flags = context.get("flags", {})

    # Update context with computed values
    context["output_size"] = original_size
    context["line_count"] = line_count

    # Decision tree (from design doc):
    # 1. If context.user_intent == "api" → Always return FULL
    if user_intent == "api":
        return SummarizationResult(
            output=output,
            mode=OutputMode.FULL,
            was_summarized=False,
            metadata={
                "original_size": original_size,
                "truncated_size": original_size,
                "reason": "api context",
            },
        )

    # 2. If context.flags.get("full") == True → Return FULL (explicit request)
    if flags.get("full", False):
        return SummarizationResult(
            output=output,
            mode=OutputMode.FULL,
            was_summarized=False,
            metadata={
                "original_size": original_size,
                "truncated_size": original_size,
                "reason": "explicit --full flag",
            },
        )

    # 3. If output_size < 500 chars → Return FULL (small output)
    if original_size < SMALL_OUTPUT_THRESHOLD:
        return SummarizationResult(
            output=output,
            mode=OutputMode.FULL,
            was_summarized=False,
            metadata={
                "original_size": original_size,
                "truncated_size": original_size,
                "reason": "below small threshold",
            },
        )

    # 4. If command in MASSIVE_OUTPUT_COMMANDS → Return COMPRESSED
    if command in MASSIVE_OUTPUT_COMMANDS:
        target_mode = MASSIVE_OUTPUT_COMMANDS[command]
        if target_mode == OutputMode.COMPRESSED:
            # Special handling for git diff
            if command == "git diff":
                compressed = compress_git_diff(output)
            else:
                compressed = compress_large_output(output)
            return SummarizationResult(
                output=compressed,
                mode=OutputMode.COMPRESSED,
                was_summarized=True,
                metadata={
                    "original_size": original_size,
                    "truncated_size": len(compressed),
                    "reason": f"command {command} uses compressed mode",
                },
            )
        elif target_mode == OutputMode.SUMMARY:
            # Use summary mode for these commands
            summarized = summarize(output, max_length=MEDIUM_OUTPUT_THRESHOLD)
            return SummarizationResult(
                output=summarized,
                mode=OutputMode.SUMMARY,
                was_summarized=True,
                metadata={
                    "original_size": original_size,
                    "truncated_size": len(summarized),
                    "reason": f"command {command} uses summary mode",
                },
            )

    # 5. If output_size < 2000 chars → Return SUMMARY (head+tail truncation)
    if original_size < MEDIUM_OUTPUT_THRESHOLD:
        summarized = summarize(output, max_length=MEDIUM_OUTPUT_THRESHOLD)
        return SummarizationResult(
            output=summarized,
            mode=OutputMode.SUMMARY,
            was_summarized=True,
            metadata={
                "original_size": original_size,
                "truncated_size": len(summarized),
                "reason": "medium output - summary mode",
            },
        )

    # 6. If output_size >= 2000 chars → Return COMPRESSED or write to temp file
    # For very large outputs, compress or write to temp file
    if original_size >= LARGE_OUTPUT_THRESHOLD:
        # Write to temp file and return compressed
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as f:
                f.write(output)
                temp_path = f.name

            compressed = compress_large_output(output, max_lines=30)
            return SummarizationResult(
                output=f"{compressed}\n\n[Full output written to: {temp_path}]",
                mode=OutputMode.COMPRESSED,
                was_summarized=True,
                metadata={
                    "original_size": original_size,
                    "truncated_size": len(compressed),
                    "temp_file": temp_path,
                    "reason": "large output - temp file written",
                },
            )
        except Exception:
            # Fallback to compression if temp file fails
            compressed = compress_large_output(output, max_lines=30)
            return SummarizationResult(
                output=compressed,
                mode=OutputMode.COMPRESSED,
                was_summarized=True,
                metadata={
                    "original_size": original_size,
                    "truncated_size": len(compressed),
                    "reason": "large output - compressed",
                },
            )

    # Default fallback: summary
    summarized = summarize(output, max_length=DEFAULT_MAX_LENGTH)
    return SummarizationResult(
        output=summarized,
        mode=OutputMode.SUMMARY,
        was_summarized=True,
        metadata={
            "original_size": original_size,
            "truncated_size": len(summarized),
            "reason": "default fallback",
        },
    )


def summarize_json(
    data: dict,
    max_length: int = DEFAULT_MAX_LENGTH,
    use_llm: bool = False,
) -> dict:
    """
    Summarize JSON-structured data.

    For dicts with 'output' or 'result' fields, summarizes those.
    Adds a 'summary' field with a count-based description.

    Args:
        data: The JSON data to summarize
        max_length: Maximum length for text fields
        use_llm: Whether to use LLM (future)

    Returns:
        Dict with original data plus summary info
    """
    import json

    result = dict(data)

    # Add summary field based on content
    # Try to find the main output field
    output_field = None
    for f in ["output", "result", "stdout", "stderr", "content", "events"]:
        if f in data and isinstance(data[f], str):
            output_field = f
            break

    if output_field:
        text = data[output_field]
        if len(text) > max_length:
            result[f"{output_field}_summary"] = _truncate_text(text, max_length)

    # Add count-based summary
    if "events" in data and isinstance(data["events"], list):
        result["summary"] = f"Found {len(data['events'])} events"
    elif "errors" in data and isinstance(data["errors"], list):
        result["summary"] = f"{len(data['errors'])} errors found"
    elif "files_processed" in data:
        files = data.get("files_processed", [])
        if isinstance(files, list):
            result["summary"] = f"Processed {len(files)} files"

    return result


def create_summary_response(
    raw_output: str,
    summary: str,
    include_raw: bool = True,
) -> dict:
    """
    Create a standardized response with both summary and raw output.

    Used when --full flag is given to include both modes.

    Args:
        raw_output: The full raw output
        summary: The summarized version
        include_raw: Whether to include raw in the response

    Returns:
        Dict with 'summary' and optionally 'raw' fields
    """
    result = {"summary": summary}

    if include_raw:
        result["raw"] = raw_output

    return result


# Export commonly used items


def slugify(text: str) -> str:
    """
    Convert text to a URL-safe slug.

    Args:
        text: The text to convert

    Returns:
        A lowercase, hyphenated version of the text
    """
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    return text.strip('-')


__all__ = [
    "summarize",
    "summarize_json",
    "create_summary_response",
    "SummarizeConfig",
    "DEFAULT_MAX_LENGTH",
    "DEFAULT_HEAD_LINES",
    "DEFAULT_TAIL_LINES",
    "summarize_git_output",
    # New exports for smart summarization
    "maybe_summarize",
    "OutputMode",
    "SummarizationResult",
    "Context",
    "compress_git_diff",
    "SMALL_OUTPUT_THRESHOLD",
    "MEDIUM_OUTPUT_THRESHOLD",
    "LARGE_OUTPUT_THRESHOLD",
]


# Git-specific summarization (for scout-git CLI)
DEFAULT_CHUNK_SIZE = 2000


async def _call_llm_summarize_git(text: str, model: str = "MiniMax-M2") -> tuple[str, bool]:
    """
    Call MiniMax to summarize git text.
    Returns (summary, success).

    Args:
        text: Raw git output to summarize
        model: Model to use (default: MiniMax-M2)
    """
    try:
        from vivarium.scout.llm.minimax import call_minimax_async
    except ImportError:
        return "", False

    system_prompt = (
        "You are a concise technical summarizer. Summarize the following git output "
        "in 1-2 sentences. Focus on key changes, numbers, and actionable info."
    )
    prompt = f"Summarize this git output:\n\n{text[:3000]}"

    try:
        summary, _ = await call_minimax_async(
            prompt=prompt,
            system=system_prompt,
            max_tokens=256,
            model=model,
        )
        return summary.strip(), True
    except Exception:
        return "", False


def _fallback_git_summarize(command: str, raw_output: str) -> str:
    """Simple count-based fallback when LLM fails."""
    lines = raw_output.strip().split("\n")

    if command == "status":
        staged = modified = untracked = 0
        for line in lines:
            if line.startswith("??"):
                untracked += 1
            elif line and line[0] in "MADRC":
                staged += 1
            elif line.startswith(" ") and "M" in line[:3]:
                modified += 1
        parts = []
        if staged:
            parts.append(f"{staged} staged")
        if modified:
            parts.append(f"{modified} modified")
        if untracked:
            parts.append(f"{untracked} untracked")
        return f"Git status: {', '.join(parts) if parts else 'clean'}"

    elif command == "diff":
        additions = sum(
            1 for line in lines if line.startswith("+") and not line.startswith("+++")
        )
        deletions = sum(
            1 for line in lines if line.startswith("-") and not line.startswith("---")
        )
        files = len([line for line in lines if line.startswith("diff --git")])
        return f"Diff: {files} file(s) changed, +{additions} -{deletions}"

    elif command == "log":
        return f"Log: {len(lines)} commit(s)"

    elif command == "branch":
        current = [line for line in lines if line.strip().startswith("*")]
        current_name = current[0].replace("*", "").strip() if current else "unknown"
        branch_count = len([line for line in lines if line.strip()])
        return f"Branches: {branch_count} total, current: {current_name}"

    elif command == "show":
        return f"Commit details: {len(lines)} lines"

    return f"{command}: {len(lines)} lines of output"


async def summarize_git_async(
    text: str, command: str = "generic", max_length: int = DEFAULT_MAX_LENGTH,
    model: str = "MiniMax-M2"
) -> str:
    """
    Summarize git output using LLM with chunking for large outputs.

    Args:
        text: Raw git output
        command: Git command name (status, diff, log, branch, show)
        max_length: Target max length for summary
        model: Model to use for summarization (default: MiniMax-M2)
               Available: MiniMax-M2.5, MiniMax-M2.5-highspeed,
               MiniMax-M2.1, MiniMax-M2.1-highspeed, MiniMax-M2

    Returns:
        Concise summary string
    """
    import asyncio

    if not text.strip():
        return f"{command}: (empty output)"

    # For small outputs, use simple fallback
    if len(text) < DEFAULT_CHUNK_SIZE:
        # Try LLM first for small outputs
        summary, success = await _call_llm_summarize_git(text, model=model)
        if success and summary:
            return summary
        return _fallback_git_summarize(command, text)

    # Large output: chunk by logical boundaries
    chunks: list[str] = []

    if command == "diff":
        # Split by file (diff --git ...)
        current_chunk: list[str] = []
        all_lines = text.split("\n")
        for line in all_lines:
            if line.startswith("diff --git") and current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            current_chunk.append(line)
        if current_chunk:
            chunks.append("\n".join(current_chunk))
    elif command == "log":
        # Split by commit (--- commit ...)
        log_chunk: list[str] = []
        all_lines = text.split("\n")
        for line in all_lines:
            if line.startswith("--- ") and log_chunk:
                chunks.append("\n".join(log_chunk))
                log_chunk = []
            log_chunk.append(line)
        if log_chunk:
            chunks.append("\n".join(log_chunk))
    else:
        # Default: split by lines
        lines_list = text.split("\n")
        for i in range(0, len(lines_list), DEFAULT_CHUNK_SIZE // 50):
            chunks.append("\n".join(lines_list[i: i + DEFAULT_CHUNK_SIZE // 50]))

    if not chunks:
        chunks = [text]

    # Summarize each chunk in parallel
    async def summarize_chunk(chunk: str) -> str:
        summary, success = await _call_llm_summarize_git(chunk, model=model)
        if success and summary:
            return summary
        return _fallback_git_summarize(command, chunk)

    results = await asyncio.gather(*[summarize_chunk(c) for c in chunks])

    # Combine chunk summaries
    combined = " | ".join(r for r in results if r)
    if len(combined) <= max_length:
        return combined

    # If combined is too long, summarize again
    final_summary, success = await _call_llm_summarize_git(combined, model=model)
    if success and final_summary:
        return final_summary
    return combined[:max_length] + "..."


def summarize_git_output(
    text: str, command: str = "generic", max_length: int = DEFAULT_MAX_LENGTH
) -> str:
    """
    Synchronous wrapper for summarize_git_async.

    Summarizes git command output using LLM when available,
    with fallback to count-based summary.
    """
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we need to create a new loop in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, summarize_git_async(text, command, max_length)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                summarize_git_async(text, command, max_length)
            )
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(summarize_git_async(text, command, max_length))
