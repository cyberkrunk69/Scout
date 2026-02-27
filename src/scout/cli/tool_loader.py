#!/usr/bin/env python
"""
Tool Loader - Dependency Injection Mock for Scout CLI.

Trick: Inject a fake FastMCP module to bypass import failures.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# Direct Tool Loader
# ============================================================

async def load_and_call_tool(tool_module: str, func_name: str, params: dict) -> Any:
    """
    Import and call a tool directly, bypassing MCP.
    
    Args:
        tool_module: e.g., "scout.tools.llm"
        func_name: e.g., "scout_plan"
        params: kwargs to pass to the function
    
    Returns:
        The result from the tool function
    """
    # Load the module directly (no mock needed - FastMCP removed)
    try:
        # Now import the "poisoned" module safely
        module = importlib.import_module(tool_module)

        # Get the actual function
        tool_func = getattr(module, func_name)

        # Execute directly
        if asyncio.iscoroutinefunction(tool_func):
            return await tool_func(**params)
        else:
            # Run sync functions in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: tool_func(**params))

    except Exception as e:
        return {"error": f"Native load failed: {str(e)}"}


def check_tool_availability(tool_module: str, func_name: str) -> tuple[bool, str]:
    """
    Check if a tool can be loaded without calling it.
    
    Returns:
        (can_load, error_message)
    """
    # FastMCP removed - no mock needed
    try:
        spec = importlib.util.find_spec(tool_module)
        if spec is None:
            return False, f"Module not found: {tool_module}"

        module = importlib.import_module(tool_module)
        if not hasattr(module, func_name):
            return False, f"Function {func_name} not in {tool_module}"

        return True, ""

    except Exception as e:
        return False, str(e)


# ============================================================
# STEP 3: TOOL REGISTRY (which tools map to which modules)
# ============================================================

TOOL_MAP = {
    # Tools from scout.tools.llm
    "scout_plan": ("scout.tools.llm", "scout_plan"),
    "scout_propose_tool": ("scout.tools.llm", "scout_propose_tool"),
    "scout_nav": ("scout.tools.llm", "scout_nav"),
    "scout_query": ("scout.tools.llm", "scout_query"),
    "scout_roast": ("scout.tools.llm", "scout_roast"),
    "scout_brief": ("scout.tools.llm", "scout_brief"),

    # Tools from scout.tools.batch
    "scout_batch": ("scout.tools.batch", "scout_batch"),
    "scout_run": ("scout.tools.batch", "scout_run"),

    # Tools from scout.tools.doc_gen
    "scout_generate_docs": ("scout.tools.doc_gen", "scout_generate_docs"),
    "scout_update_docs": ("scout.tools.doc_gen", "scout_update_docs"),
    "scout_doc_sync": ("scout.tools.doc_gen", "scout_doc_sync"),

    # Tools from scout.tools.git
    "scout_git_add": ("scout.tools.git", "scout_git_add"),
    "scout_git_branch": ("scout.tools.git", "scout_git_branch"),
    "scout_git_commit": ("scout.tools.git", "scout_git_commit"),
    "scout_git_diff": ("scout.tools.git", "scout_git_diff"),
    "scout_git_log": ("scout.tools.git", "scout_git_log"),
    "scout_git_show": ("scout.tools.git", "scout_git_show"),
    "scout_git_status": ("scout.tools.git", "scout_git_status"),
    "scout_pr": ("scout.tools.git", "scout_pr"),
    "scout_pr_info": ("scout.tools.git", "scout_pr_info"),

    # Tools from scout.tools.validation
    "scout_env": ("scout.tools.validation", "scout_env"),
    "scout_function_info": ("scout.tools.validation", "scout_function_info"),
    "scout_lint": ("scout.tools.validation", "scout_lint"),
    "scout_validate_module": ("scout.tools.validation", "scout_validate_module"),

    # Tools from scout.tools.file_ops
    "scout_read_file": ("scout.tools.file_ops", "scout_read_file"),
    "scout_write_with_review": ("scout.tools.file_ops", "scout_write_with_review"),
    "scout_edit": ("scout.tools.file_ops", "scout_edit"),
    "scout_delete_with_review": ("scout.tools.file_ops", "scout_delete_with_review"),

    # Tools from scout.tools.admin
    "scout_grep": ("scout.tools.admin", "scout_grep"),
    "scout_shell": ("scout.tools.admin", "scout_shell"),
    "scout_help": ("scout.tools.admin", "scout_help"),
    "scout_status": ("scout.tools.admin", "scout_status"),
    "scout_index": ("scout.tools.admin", "scout_index"),
    "scout_search_docs": ("scout.tools.admin", "scout_search_docs"),
    "scout_reindex_docs": ("scout.tools.admin", "scout_reindex_docs"),
}


def get_tool_schemas() -> list[dict]:
    """
    Extract tool schemas (name + docstring) for LLM-based routing.
    
    Returns a list of dicts with 'name' and 'description' keys.
    Uses up to 10 lines of docstring to provide enough context for routing decisions.
    """
    import inspect
    import importlib
    
    schemas = []
    
    for tool_name, (module_path, func_name) in TOOL_MAP.items():
        try:
            module = importlib.import_module(module_path)
            func = getattr(module, func_name, None)
            if func and func.__doc__:
                # Get up to 10 lines of docstring for better routing context
                doc_lines = func.__doc__.strip().split('\n')[:10]
                doc = ' '.join(line.strip() for line in doc_lines if line.strip())
                schemas.append({
                    "name": tool_name,
                    "description": doc
                })
            else:
                schemas.append({
                    "name": tool_name,
                    "description": f"Tool: {tool_name}"
                })
        except Exception:
            schemas.append({
                "name": tool_name,
                "description": f"Tool: {tool_name}"
            })
    
    return schemas


def get_tool_signature(tool_name: str) -> dict:
    """
    Introspect a tool's function signature to get accepted parameters.
    
    Returns a dict of param_name -> default_value (or ... for required)
    Uses ToolSignatureRegistry if available, otherwise falls back to direct introspection.
    """
    import inspect
    
    if tool_name not in TOOL_MAP:
        return {}
    
    # Try registry first for cached metadata
    try:
        from scout.registry import ToolSignatureRegistry
        registry = ToolSignatureRegistry()
        cached = registry.get_signature(tool_name)
        if cached:
            return cached
    except ImportError:
        pass
    
    # Fall back to direct introspection
    module_path, func_name = TOOL_MAP[tool_name]
    try:
        # FastMCP removed - no mock needed
        module = importlib.import_module(module_path)
        func = getattr(module, func_name, None)
        if func:
            sig = inspect.signature(func)
            params = {}
            for name, param in sig.parameters.items():
                if param.default is inspect.Parameter.empty:
                    params[name] = ...  # Required param
                else:
                    params[name] = param.default
            return params
    except Exception:
        pass
    return {}


def filter_params_to_signature(tool_name: str, params: dict) -> tuple[dict, dict]:
    """
    Filter params to only include those accepted by the tool function.
    
    Returns (filtered_params, ignored_params).
    Uses introspection to discover what parameters the function accepts.
    """
    signature = get_tool_signature(tool_name)
    if not signature:
        return params, {}  # Can't introspect, return as-is
    
    # Split into accepted and ignored params
    filtered = {}
    ignored = {}
    for key, value in params.items():
        if key in signature:
            filtered[key] = value
        else:
            ignored[key] = value
    
    return filtered, ignored


def validate_and_filter_params(tool_name: str, params: dict) -> tuple[dict, dict, list]:
    """
    Validate and filter params with full error detection.
    
    Returns (valid_params, ignored_params, error_params).
    - valid_params: params that match signature
    - ignored_params: params not in signature (silently ignored)
    - error_params: params that caused errors (required but missing)
    """
    signature = get_tool_signature(tool_name)
    if not signature:
        return params, {}, []
    
    valid = {}
    ignored = {}
    errors = []
    
    for key, value in params.items():
        if key in signature:
            valid[key] = value
        else:
            ignored[key] = value
    
    # Check for missing required params
    for param_name, default in signature.items():
        if default is ... and param_name not in valid:
            errors.append(param_name)
    
    return valid, ignored, errors


# Add path for importing scout modules
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(REPO_ROOT))

# Import parameter registry for translating CLI params to tool params
from scout.parameter_registry import transform_params


# Executor params - handled by tool_loader wrapper, not passed to tools
EXECUTOR_PARAMS = {
    "success_condition",  # Natural language condition to evaluate for success
    "input_required",    # Whether tool should pause for user input
    "input_prompt",     # Prompt to show user
}


def _ensure_json_output(result: Any) -> dict:
    """
    Ensure tool result is always JSON with standardized format.
    
    All tools return: {"success": bool, "result": Any, "error": str|None, "metadata": dict}
    """
    # Handle ToolOutput objects specially - convert to dict first
    if hasattr(result, 'to_dict') and callable(result.to_dict):
        result = result.to_dict()
    
    # Already a dict - check if it has our standard format
    if isinstance(result, dict):
        if "success" in result:
            # Already has success field - ensure all fields present
            return {
                "success": bool(result.get("success", True)),
                "result": result.get("result", result.get("data", str(result))),
                "error": result.get("error"),
                "metadata": result.get("metadata", {})
            }
        else:
            # Dict but no success field - treat as successful result
            return {
                "success": True,
                "result": result,
                "error": None,
                "metadata": {}
            }
    
    # Non-dict result (string, etc)
    return {
        "success": True,
        "result": str(result) if result is not None else "",
        "error": None,
        "metadata": {}
    }


def _evaluate_success_condition(result: dict, condition: str, context: dict = None) -> bool:
    """
    Evaluate a natural language success condition against the result.
    
    Args:
        result: The tool execution result
        condition: Natural language condition (e.g., "file exists", "no errors", "output contains 'OK'")
        context: Execution context
    
    Returns:
        bool indicating if condition is met
    """
    if not condition:
        return result.get("success", True)
    
    condition_lower = condition.lower()
    result_str = str(result.get("result", "")).lower()
    error = result.get("error")
    
    # Built-in condition patterns
    if "no error" in condition_lower or "success" in condition_lower:
        return error is None and result.get("success", True)
    
    if "file exists" in condition_lower:
        import os
        # Check if result contains a file path
        return error is None
    
    if "contains" in condition_lower:
        # Extract the thing to check
        import re
        match = re.search(r"contains ['\"](.+?)['\"]", condition_lower)
        if match:
            target = match.group(1)
            return target in result_str
        return True  # Can't parse, assume success
    
    # Default: if no errors and has result, it's a success
    return error is None and result.get("success", True) != False


async def call_scout_tool(tool_name: str, params: dict = None) -> Any:
    """
    Call a Scout tool by name using native injection.
    
    Includes:
    - Parameter transformation via registry
    - Signature-based filtering (introspection)
    - Error handling with detailed feedback
    - Executor params handling (success_condition, input_required, etc)
    - Always returns JSON with success/result/error/metadata
    """
    if tool_name not in TOOL_MAP:
        return {"error": f"Unknown tool: {tool_name}", "success": False}

    if params is None:
        params = {}

    # Extract executor params BEFORE passing to tool (these aren't tool params)
    executor_params = {}
    for param in EXECUTOR_PARAMS:
        if param in params:
            executor_params[param] = params.pop(param)
    
    success_condition = executor_params.get("success_condition")

    # Step 1: Transform CLI params to tool params using registry
    transformed_params = transform_params(tool_name, params)

    # Step 2: Filter to only params accepted by the tool function (introspection)
    # Also track ignored params for debugging
    filtered_params, ignored_params = filter_params_to_signature(tool_name, transformed_params)
    
    # Log ignored params for debugging
    if ignored_params:
        _log_parameter_warning(tool_name, ignored_params)

    # Step 3: Execute with error handling
    module_path, func_name = TOOL_MAP[tool_name]
    try:
        raw_result = await load_and_call_tool(module_path, func_name, filtered_params)
        
        # Ensure JSON output
        result = _ensure_json_output(raw_result)
        
        # Evaluate success_condition if provided
        if success_condition:
            result["success"] = _evaluate_success_condition(result, success_condition)
        
        return result
        
    except TypeError as e:
        # Handle parameter errors gracefully
        error_msg = str(e)
        return {
            "error": f"Parameter error in {tool_name}: {error_msg}",
            "success": False,
            "tool": tool_name,
            "provided_params": list(filtered_params.keys()),
            "available_params": list(get_tool_signature(tool_name).keys()),
            "suggestion": "Check tool signature - some params may not be supported"
        }
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}", "tool": tool_name, "success": False}


def _log_parameter_warning(tool_name: str, ignored_params: dict) -> None:
    """Log warning about ignored parameters."""
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(
        f"Tool {tool_name} called with ignored parameters: {list(ignored_params.keys())}"
    )


def initialize_tool_registry() -> None:
    """
    Initialize the ToolSignatureRegistry by scanning all tool modules.
    
    Call this once at startup to populate the registry with all tool signatures.
    """
    try:
        from scout.registry import ToolSignatureRegistry, scan_and_register_tools
        
        registry = ToolSignatureRegistry()
        
        # Scan all known tool modules
        tool_modules = [
            "scout.tools.llm",
            "scout.tools.batch",
            "scout.tools.doc_gen",
            "scout.tools.git",
            "scout.tools.validation",
            "scout.tools.github",
            "scout.tools.file_ops",
            "scout.tools.admin",
        ]
        
        for module_path in tool_modules:
            try:
                scan_and_register_tools(module_path)
            except Exception:
                pass  # Module may not exist
        
    except ImportError:
        pass  # Registry not available, will use direct introspection


def list_available_tools() -> list[str]:
    """List all tools that can be called."""
    return list(TOOL_MAP.keys())


if __name__ == "__main__":
    # Test tool availability (FastMCP mock no longer needed)
    print("Checking tool availability...")
    for tool_name in ["scout_plan", "scout_git_status", "scout_lint"]:
        if tool_name in TOOL_MAP:
            module, func = TOOL_MAP[tool_name]
            can_load, err = check_tool_availability(module, func)
            status = "OK" if can_load else f"FAIL: {err}"
            print(f"  {tool_name}: {status}")
