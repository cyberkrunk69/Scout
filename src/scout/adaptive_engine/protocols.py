"""
Protocols and type definitions for the Adaptive Engine.

This module defines the core interfaces that tools and components must implement
to participate in the adaptive execution framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Protocol


@dataclass
class ToolContext:
    """Context passed to a tool for execution.
    
    Attributes:
        request: The user's request or command
        state: Current planning state
        config: Configuration dict for the tool
    """
    request: str
    state: dict[str, Any]
    config: dict[str, Any]


@dataclass
class ToolResult:
    """Result returned by a tool after execution.
    
    Attributes:
        success: Whether the tool executed successfully
        output: The tool's output content
        error: Error message if success is False
        metadata: Additional metadata about execution
    """
    success: bool
    output: str = ""
    error: str = ""
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Tool(Protocol):
    """Protocol defining the interface for adaptive engine tools.
    
    Any tool that wants to participate in the adaptive execution framework
    must implement this protocol.
    """

    async def execute(self, context: ToolContext) -> ToolResult:
        """Execute the tool with the given context.
        
        Args:
            context: Execution context with request and state
            
        Returns:
            ToolResult containing the execution outcome
        """
        ...


class AsyncTool(Protocol):
    """Protocol for async tools that may perform long-running operations.
    
    This is an alias for tools that explicitly declare async behavior.
    """

    async def execute(self, context: ToolContext) -> Awaitable[ToolResult]:
        """Execute the tool asynchronously.
        
        Args:
            context: Execution context with request and state
            
        Returns:
            Awaitable ToolResult
        """
        ...


# Re-export for convenience
__all__ = [
    "ToolContext",
    "ToolResult",
    "Tool",
    "AsyncTool",
]
