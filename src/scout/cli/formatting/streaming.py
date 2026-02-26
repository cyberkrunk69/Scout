#!/usr/bin/env python
"""
Streaming output for long responses.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from rich.console import Console
from rich.live import Live
from rich.text import Text


class StreamingOutput:
    """Streaming output with character-by-character display."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.text = Text()
        self.live: Optional[Live] = None

    async def stream(self, chunks: AsyncIterator[str], show_live: bool = True):
        """
        Stream chunks to console.

        Args:
            chunks: Async iterator of string chunks
            show_live: Whether to show live updating
        """
        if show_live:
            self.live = Live(console=self.console, refresh_per_second=10)
            self.live.start()

        async for chunk in chunks:
            self.text.append(chunk)
            if self.live:
                self.live.update(self.text)
            else:
                self.console.print(chunk, end="")

        if self.live:
            self.live.stop()
            self.console.print(self.text)

    def stream_sync(self, chunks: AsyncIterator[str], show_live: bool = True):
        """Synchronous wrapper for stream."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Can't use async in sync context with live display
                import sys
                for chunk in chunks:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                print()
                return
            loop.run_until_complete(self.stream(chunks, show_live))
        except RuntimeError:
            # No event loop, create one
            asyncio.run(self.stream(chunks, show_live))


async def stream_text(text: str, delay: float = 0.02) -> AsyncIterator[str]:
    """
    Generator that yields characters with a delay.

    Args:
        text: Text to stream
        delay: Delay between characters in seconds
    """
    for char in text:
        yield char
        await asyncio.sleep(delay)


async def stream_lines(lines: list[str], delay: float = 0.1) -> AsyncIterator[str]:
    """
    Generator that yields lines with a delay.

    Args:
        lines: Lines to stream
        delay: Delay between lines in seconds
    """
    for i, line in enumerate(lines):
        yield line + ("\n" if i < len(lines) - 1 else "")
        await asyncio.sleep(delay)


class StreamingFormatter:
    """Format text for streaming display."""

    @staticmethod
    def truncate(text: str, max_length: int = 1000) -> str:
        """Truncate text for display."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."

    @staticmethod
    def strip_ansi(text: str) -> str:
        """Remove ANSI codes from text."""
        import re
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)


if __name__ == "__main__":
    async def test():
        output = StreamingOutput()
        chunks = stream_text("Hello, this is a streaming test!")
        await output.stream(chunks)

    asyncio.run(test())
