#!/usr/bin/env python
"""
Parameter Mapping Registry

Maps CLI parameter names to tool function parameter names.
Centralized registry for parameter name translation.
"""

from __future__ import annotations

from typing import Callable, Optional


class ParameterRegistry:
    """Registry for mapping CLI parameters to tool parameters."""
    
    _instance: Optional["ParameterRegistry"] = None
    
    def __new__(cls):
        """Create or return the singleton ParameterRegistry instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._map = {}
            cls._instance._reverse_map = {}
        return cls._instance
    
    def register(self, tool_name: str, cli_param: str, tool_param: str):
        """Register a parameter mapping for a tool."""
        if tool_name not in self._map:
            self._map[tool_name] = {}
        
        self._map[tool_name][cli_param] = tool_param
        # Also build reverse
        if tool_name not in self._reverse_map:
            self._reverse_map[tool_name] = {}
        self._reverse_map[tool_name][tool_param] = cli_param
    
    def get_tool_param(self, tool_name: str, cli_param: str) -> Optional[str]:
        """Get tool parameter name from CLI parameter."""
        return self._map.get(tool_name, {}).get(cli_param)
    
    def get_cli_param(self, tool_name: str, tool_param: str) -> Optional[str]:
        """Get CLI parameter name from tool parameter."""
        return self._reverse_map.get(tool_name, {}).get(tool_param)
    
    def transform_params(self, tool_name: str, cli_params: dict) -> dict:
        """Transform CLI params to tool params."""
        if tool_name not in self._map:
            return cli_params  # No mapping, return as-is
        
        tool_params = {}
        for key, value in cli_params.items():
            mapped_key = self._map[tool_name].get(key, key)
            tool_params[mapped_key] = value
        
        return tool_params


# ============================================================
# PARAMETER MAPPING INVENTORY
# Based on audit of scout_mcp_server.py tool signatures
# ============================================================

# Tool: scout_plan
# Signature: async def scout_plan(request: str, output_dir: str = "docs/plans", json_output: bool = True, structured: bool = False)
CLI_TOOL_PARAM_MAP = {
    "scout_plan": {
        "query": "request",  # CLI sends 'query', tool expects 'request'
        "request": "request",
    },
    # Tool: scout_nav
    # Signature: async def scout_nav(query: str = None, task: str = None, entry: str = None)
    "scout_nav": {
        "query": "task",  # CLI sends 'query', tool expects 'task'
        "task": "task", 
        "entry": "entry",
    },
    # Tool: scout_query
    # Signature: async def scout_query(query: str, scope: str = None, json_output: bool = True)
    "scout_query": {
        "query": "query",
        "question": "query",  # Alternative
        "scope": "scope",
    },
    # Tool: scout_grep
    # Signature: async def scout_grep(pattern: str, path: str = None)
    "scout_grep": {
        "pattern": "pattern",
        "query": "pattern",  # Alternative
    },
    # Tool: scout_batch
    # Signature: async def scout_batch(tasks_json: str, auto_execute: bool = True)
    "scout_batch": {
        "tasks": "tasks_json",
        "tasks_json": "tasks_json",
    },
    # Tool: scout_generate_docs
    # Signature: async def scout_generate_docs(request: str, audience: str = "developer", output_format: str = "markdown", target_files: str | None = None, technical_depth: str = "medium", include_code_examples: bool = True, include_api_reference: bool = True, ctx: Any = None)
    "scout_generate_docs": {
        "request": "request",
        "target_files": "target_files",
        "audience": "audience",
        "output_format": "output_format",
    },
    # Tool: scout_roast
    # Signature: async def scout_roast(scope: str = None)
    "scout_roast": {
        "scope": "scope",
    },
    # Tool: scout_brief
    # Signature: async def scout_brief(task: str, scope: str = None)
    "scout_brief": {
        "task": "task",
        "query": "task",
    },
    # Tool: scout_edit
    # Signature: async def scout_edit(file_path: str, instruction: str, dry_run: bool = False)
    "scout_edit": {
        "file": "file_path",
        "file_path": "file_path",
        "prompt": "instruction",
        "instruction": "instruction",
        "dry_run": "dry_run",
    },
    # Tool: scout_propose_tool
    # Signature: async def scout_propose_tool(request: str)
    "scout_propose_tool": {
        "query": "request",
        "request": "request",
    },
    # Git tools - map 'query' to the actual tool parameters
    # scout_git_status: (full=False, json_output=False, summarize=False, ctx=None)
    "scout_git_status": {
        "query": "summarize",  # 'query' maps to summarize param for summary generation
    },
    # scout_git_diff: (cached=False, full=False, json_output=False, ctx=None)
    "scout_git_diff": {
        "query": "full",  # Just ignore extra params via filtering
    },
    # scout_git_log: (max_count=10, full=False, json_output=False, ctx=None)
    # NOTE: Removed "query" -> "max_count" mapping. Signature-based filtering now
    # handles param validation, so we don't need (or want) this incorrect mapping.
    "scout_git_log": {
    },
    # scout_git_branch: (full=False, json_output=False, ctx=None)
    "scout_git_branch": {},
    # scout_git_show: (ref="HEAD", full=False, json_output=False, ctx=None)
    "scout_git_show": {
        "query": "ref",
    },
    # scout_git_add: (paths=None, all_files=False, patch=False, json_output=False, ctx=None)
    "scout_git_add": {
        "query": "paths",
    },
    # scout_git_commit: (message: str, allow_empty=False, json_output=False, ctx=None)
    "scout_git_commit": {
        "query": "message",
        "message": "message",
    },
    # scout_pr: from github.py
    "scout_pr": {},
    # scout_pr_info: from github.py
    "scout_pr_info": {},
    # Validation tools
    "scout_env": {},
    "scout_function_info": {},
    "scout_lint": {
        "query": "path",
    },
    "scout_validate_module": {
        "query": "module_name",
    },
}


def register_all_mappings():
    """Register all parameter mappings."""
    registry = ParameterRegistry()
    
    for tool_name, mappings in CLI_TOOL_PARAM_MAP.items():
        for cli_param, tool_param in mappings.items():
            if cli_param != tool_param:  # Only register if different
                registry.register(tool_name, cli_param, tool_param)


def transform_params(tool_name: str, cli_params: dict) -> dict:
    """Transform CLI params to tool params using registry."""
    registry = ParameterRegistry()
    return registry.transform_params(tool_name, cli_params)


# Auto-register on import
register_all_mappings()


if __name__ == "__main__":
    # Test the registry
    registry = ParameterRegistry()
    
    # Test scout_plan mapping
    cli_params = {"query": "add caching", "json_output": True}
    tool_params = transform_params("scout_plan", cli_params)
    print(f"scout_plan: {cli_params} -> {tool_params}")
    
    # Test scout_query mapping
    cli_params = {"question": "how does auth work?"}
    tool_params = transform_params("scout_query", cli_params)
    print(f"scout_query: {cli_params} -> {tool_params}")
