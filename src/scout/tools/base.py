"""Legacy compatibility layer – re‑exports Tool classes from adaptive_engine.protocols.

This shim provides the import path expected by legacy modules (e.g., pr_creator.py)
while maintaining DRY: Tool/ToolResult are defined once in adaptive_engine.protocols.
"""
from scout.adaptive_engine.protocols import Tool, ToolResult

__all__ = ["Tool", "ToolResult"]
