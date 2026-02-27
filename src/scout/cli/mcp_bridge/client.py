#!/usr/bin/env python
"""
Scout Client - Uses native Python imports via tool_loader.

No subprocess, no MCP - calls tools directly with dependency injection.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

from scout.cli.tool_loader import (
    call_scout_tool,
    list_available_tools,
    load_and_call_tool,
    TOOL_MAP,
)

# Suppress verbose logging
logging.getLogger("scout.cache_deps").setLevel(logging.ERROR)

from scout.cli.context.session import Session


class ScoutClient:
    """Direct client to Scout tools via native Python imports."""

    def __init__(self):
        self.tools = list_available_tools()

    async def call_tool(self, tool_name: str, params: dict = None) -> dict:
        """Call a tool by name using native imports."""
        params = params or {}
        
        try:
            result = await call_scout_tool(tool_name, params)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    async def call_tool_streaming(self, tool_name: str, params: dict = None) -> AsyncIterator[str]:
        """Streaming - yields result when complete."""
        result = await self.call_tool(tool_name, params)
        yield str(result)

    async def list_tools(self) -> list[str]:
        """List available tools."""
        return self.tools

    async def get_tool_info(self, tool_name: str) -> dict:
        """Get tool info."""
        if tool_name in TOOL_MAP:
            module, func = TOOL_MAP[tool_name]
            return {"description": f"Tool from {module}.{func}"}
        return {"description": "Unknown tool"}

    async def close(self):
        pass


# Aliases
MCPClient = ScoutClient


_client: Optional[ScoutClient] = None


def get_scout_client() -> ScoutClient:
    global _client
    if _client is None:
        _client = ScoutClient()
    return _client


get_mcp_client = get_scout_client


if __name__ == "__main__":
    async def main():
        client = ScoutClient()
        print("Available tools:")
        for tool in await client.list_tools():
            print(f"  {tool}")
    
    asyncio.run(main())
