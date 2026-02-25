from __future__ import annotations
"""Parser for extracting structured steps from plan markdown output."""

import json
import logging
import re
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# Track if we've already warned about prose parsing this session
_prose_warning_issued = False


def _try_parse_json_steps(plan_text: str) -> Optional[List[Dict[str, Any]]]:
    """Try to extract JSON steps from plan text.
    
    Returns None if no valid JSON steps found.
    """
    # Look for JSON code blocks first
    json_block_pattern = re.compile(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", re.MULTILINE)
    matches = json_block_pattern.findall(plan_text)
    
    for match in matches:
        try:
            data = json.loads(match)
            if "steps" in data and isinstance(data["steps"], list):
                return data["steps"]
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            continue
    
    # Try finding raw JSON at the start
    plan_stripped = plan_text.strip()
    if plan_stripped.startswith("{"):
        try:
            data = json.loads(plan_stripped)
            if "steps" in data and isinstance(data["steps"], list):
                return data["steps"]
            elif isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    
    return None


def parse_plan_steps_with_json_first(plan_text: str) -> list[dict]:
    """
    Parse plan markdown text into structured steps.
    
    Tries JSON format first (preferred), then falls back to prose parsing.
    If prose parsing is used, emits a DEPRECATION warning.
    
    The warning is only logged once per session to avoid spam.
    """
    global _prose_warning_issued
    
    # Try JSON first
    json_steps = _try_parse_json_steps(plan_text)
    if json_steps:
        # Convert to our step format
        steps = []
        for i, step in enumerate(json_steps):
            if isinstance(step, dict):
                command = step.get("command", "")
                
                # Handle browser_act command specially - preserve full WebStep schema
                if command == "browser_act":
                    args = step.get("args", {})
                    steps.append({
                        "id": step.get("id", i + 1),
                        "description": step.get("description", ""),
                        "command": "browser_act",
                        "action": args.get("action", "click"),  # Default to click
                        "target": args.get("target"),              # Optional
                        "value": args.get("value"),               # Optional  
                        "url": args.get("url"),                   # Optional
                        "condition": args.get("condition"),       # Optional
                        "depends_on": step.get("depends_on", []),
                    })
                else:
                    steps.append({
                        "id": step.get("id", i + 1),
                        "description": step.get("description", step.get("title", "")),
                        "command": command,
                        "args": step.get("args", []),
                        "depends_on": step.get("depends_on", []),
                    })
        if steps:
            return steps
    
    # Fall back to prose parsing with deprecation warning
    if not _prose_warning_issued:
        logger.warning(
            "DEPRECATED: Prose parsing will be removed in v2.0. "
            "Please emit JSON steps in your plans."
        )
        _prose_warning_issued = True
    
    # Use existing prose parser
    return parse_plan_steps(plan_text)


def parse_plan_steps(plan_text: str) -> list[dict]:
    """
    Parse plan markdown text into structured steps.

    Handles formats like:
    - ## Step 1: Do something
    - ### Step 1. Do something
    - 1. Do something
    - **Step 1** Do something
    
    Also extracts advanced execution metadata:
    - input_required: step pauses for user input
    - parallel_with: step runs in parallel with others
    - condition: conditional execution (if/else)
    - on_success/on_failure: flow control

    Returns:
        List of step dicts with keys:
        - id: step identifier
        - description: human-readable description
        - command: inferred scout command
        - args: command arguments
        - input_required: bool - pause for user input
        - input_prompt: str - prompt to show
        - parallel_with: list[str] - step IDs to run in parallel
        - condition: str - conditional expression
        - on_success: str - next step if success
        - on_failure: str - next step if failure
    """
    steps = []

    # Pattern for markdown headers with number only: ### 1. or ## 2:
    header_with_number_pattern = re.compile(r"^#{1,6}\s*(\d+)[\.:]\s*(.+)$", re.MULTILINE)

    # Pattern for markdown headers: ## Step N: or ### Step N.
    header_pattern = re.compile(r"^#{1,6}\s*Step\s*(\d+)[\.:]\s*(.+)$", re.MULTILINE)

    # Pattern for numbered lists: 1. Do something
    numbered_pattern = re.compile(r"^(\d+)\.\s*(.+)$", re.MULTILINE)

    # Pattern for bold: **Step 1** or **1.**
    bold_pattern = re.compile(r"^\*\*(?:Step\s*)?(\d+)[\.\:]\s*\*\*(.+)$", re.MULTILINE)

    # Pattern for execution metadata markers in description
    input_marker = re.compile(r"\[(?:input|prompt|getUserInput)\s*:\s*([^\]]+)\]", re.IGNORECASE)
    parallel_marker = re.compile(r"\[(?:parallel|parallel_with)\s*:\s*([^\]]+)\]", re.IGNORECASE)
    condition_marker = re.compile(r"\[(?:if|when|condition)\s*:\s*([^\]]+)\]", re.IGNORECASE)
    on_success_marker = re.compile(r"\[(?:on_success|if_success)\s*:\s*([^\]]+)\]", re.IGNORECASE)
    on_failure_marker = re.compile(r"\[(?:on_failure|if_error)\s*:\s*([^\]]+)\]", re.IGNORECASE)
    # New markers
    sub_plan_marker = re.compile(r"\[(?:sub_plan|subtask|spawn)\s*:\s*([^\]]+)\]", re.IGNORECASE)
    retry_marker = re.compile(r"\[(?:retry|retries)\s*:\s*(\d+)\]", re.IGNORECASE)
    timeout_marker = re.compile(r"\[(?:timeout)\s*:\s*(\d+)\s*(?:s|seconds?)?\]", re.IGNORECASE)

    # Try headers first
    matches = header_pattern.findall(plan_text)

    if matches:
        for match in matches:
            step_id = int(match[0])
            description = match[1].strip()
            step = _build_step(step_id, description, input_marker, parallel_marker, 
                             condition_marker, on_success_marker, on_failure_marker,
                             sub_plan_marker, retry_marker, timeout_marker)
            steps.append(step)
    else:
        # Try numbered list
        matches = numbered_pattern.findall(plan_text)
        for match in matches:
            step_id = int(match[0])
            description = match[1].strip()
            # Skip if it looks like a regular list item, not a step
            if len(description) > 5:
                step = _build_step(step_id, description, input_marker, parallel_marker,
                                 condition_marker, on_success_marker, on_failure_marker,
                                 sub_plan_marker, retry_marker, timeout_marker)
                steps.append(step)

    # Extract parallel groups from plan metadata section
    parallel_groups = _extract_parallel_groups(plan_text)
    if parallel_groups and steps:
        # Tag steps with their parallel group
        for group in parallel_groups:
            for idx in group:
                for step in steps:
                    if str(step["id"]) == str(idx) or step["id"] == idx:
                        step["parallel_with"] = [str(x) for x in group if str(x) != str(idx)]

    return steps


def _build_step(step_id: int, description: str, input_marker, parallel_marker,
                condition_marker, on_success_marker, on_failure_marker,
                sub_plan_marker=None, retry_marker=None, timeout_marker=None) -> dict:
    """Build a step dict with all metadata extracted."""
    command = _infer_command(description)
    args = _infer_args(description, command)
    
    step = {
        "id": step_id,
        "description": description,
        "command": command,
        "args": args,
    }
    
    # Extract input_required
    input_match = input_marker.search(description)
    if input_match:
        step["input_required"] = True
        step["input_prompt"] = input_match.group(1).strip()
        # Remove marker from description for clean display
        description = input_marker.sub("", description)
    
    # Extract parallel_with
    parallel_match = parallel_marker.search(description)
    if parallel_match:
        parallel_ids = [p.strip() for p in parallel_match.group(1).split(",")]
        step["parallel_with"] = parallel_ids
    
    # Extract condition
    condition_match = condition_marker.search(description)
    if condition_match:
        step["condition"] = condition_match.group(1).strip()
    
    # Extract on_success
    success_match = on_success_marker.search(description)
    if success_match:
        step["on_success"] = success_match.group(1).strip()
    
    # Extract on_failure  
    failure_match = on_failure_marker.search(description)
    if failure_match:
        step["on_failure"] = failure_match.group(1).strip()
    
    # Extract sub_plan (nested sub-plan request)
    if sub_plan_marker:
        sub_plan_match = sub_plan_marker.search(description)
        if sub_plan_match:
            step["sub_plan"] = sub_plan_match.group(1).strip()
    
    # Extract retry count
    if retry_marker:
        retry_match = retry_marker.search(description)
        if retry_match:
            step["retry"] = int(retry_match.group(1).strip())
    
    # Extract timeout
    if timeout_marker:
        timeout_match = timeout_marker.search(description)
        if timeout_match:
            step["timeout"] = int(timeout_match.group(1).strip())
    
    return step


def _extract_parallel_groups(plan_text: str) -> list[list]:
    """Extract parallel execution groups from plan metadata."""
    groups = []
    
    # Look for parallel section
    parallel_section = re.search(
        r"##\s*(?:Parallel|Parallel Execution|Parallelization)\s*\n(.*?)(?=\n##|\Z)",
        plan_text, re.IGNORECASE | re.DOTALL
    )
    
    if parallel_section:
        section_text = parallel_section.group(1)
        # Match patterns like "Group 1: 1, 2, 3" or "1, 2, 3 can run in parallel"
        group_patterns = [
            r"(?:Group\s*\d+|Step\s*\d+)\s*[:\-]\s*([\d,\s]+)",
            r"([\d,\s]+)\s+(?:can run|run together|parallel)",
        ]
        for pattern in group_patterns:
            matches = re.findall(pattern, section_text, re.IGNORECASE)
            for match in matches:
                group = [int(x.strip()) for x in match.split(",") if x.strip().isdigit()]
                if len(group) > 1:
                    groups.append(group)
    
    return groups


def _infer_command(description: str) -> str:
    """Infer the appropriate scout command from step description.

    Only returns commands supported by batch pipeline's COMMAND_BUILDERS.
    """
    desc_lower = description.lower()

    # File operations (highest priority - more specific)
    if any(
        word in desc_lower
        for word in ["create file", "write file", "new file", "add file"]
    ):
        return "write_file"
    if any(
        word in desc_lower
        for word in [
            "edit file",
            "modify file",
            "update file",
            "change file",
            "replace in",
        ]
    ):
        return "edit_file"
    if any(word in desc_lower for word in ["delete file", "remove file"]):
        return "delete_file"

    # Supported commands in batch pipeline
    if any(word in desc_lower for word in ["lint", "fix", "check"]):
        return "lint"
    if any(word in desc_lower for word in ["test", "spec", "run tests"]):
        return "run"
    if any(word in desc_lower for word in ["plan", "design", "architect"]):
        return "plan"
    if any(word in desc_lower for word in ["audit", "analyze", "check"]):
        return "audit"
    if any(word in desc_lower for word in ["roast", "critique", "review"]):
        return "roast"
    if any(word in desc_lower for word in ["validate", "verify"]):
        return "validate"
    if any(word in desc_lower for word in ["query", "search", "find"]):
        return "query"
    if any(word in desc_lower for word in ["index", "rebuild"]):
        return "index"
    if any(word in desc_lower for word in ["doc", "sync", "documentation"]):
        return "doc_sync"
    if any(word in desc_lower for word in ["env", "environment"]):
        return "env"
    if any(word in desc_lower for word in ["git", "branch", "diff", "log"]):
        return "git_status"  # Use git_status as safe default
    if any(word in desc_lower for word in ["brief", "summary"]):
        return "brief"
    if any(word in desc_lower for word in ["ci", "guard", "gate"]):
        return "ci_guard"

    # Default to nav (safest command - just navigates/explores)
    return "nav"


def _infer_args(description: str, command: str = "nav") -> dict:
    """Infer args from step description based on command type.

    Returns appropriate args for each supported command.
    """
    import re

    # write_file: Extract file path and optional content
    if command == "write_file":
        # Try to find file path - look for "create X" or "write to X" or just a path
        path_match = re.search(
            r"(?:create|write|file|to)\s+([^\s]+(?:\.py|\.txt|\.md|\.json|\.yaml|\.yml)[^\s]*)",
            description,
            re.IGNORECASE,
        )
        if path_match:
            file_path = path_match.group(1).strip("`\"'（）")
            return {"file_path": file_path}
        # Fallback: look for any .py/.txt/.md file
        path_match = re.search(r"([\w/]+\.(?:py|txt|md|json|yaml|yml))", description)
        if path_match:
            return {"file_path": path_match.group(1)}
        return {}

    # edit_file: Extract file path
    if command == "edit_file":
        path_match = re.search(
            r"(?:edit|modify|update|change)\s+([^\s]+(?:\.py|\.txt|\.md|\.json|\.yaml|\.yml)[^\s]*)",
            description,
            re.IGNORECASE,
        )
        if path_match:
            file_path = path_match.group(1).strip("`\"'（）")
            return {"file_path": file_path}
        # Fallback: look for any .py file
        path_match = re.search(r"([\w/]+\.py)", description)
        if path_match:
            return {"file_path": path_match.group(1)}
        return {}

    # delete_file: Extract file path
    if command == "delete_file":
        path_match = re.search(
            r"(?:delete|remove)\s+([^\s]+\.py)", description, re.IGNORECASE
        )
        if path_match:
            file_path = path_match.group(1).strip('`"\'（）')
            return {"file_path": file_path}
        # Fallback: look for any .py file
        path_match = re.search(r'([\w/]+\.py)', description)
        if path_match:
            return {"file_path": path_match.group(1)}
        return {}

    # nav uses --question
    if command == "nav":
        return {"question": description[:200]}
    # run takes module name as positional arg, not description
    if command == "run":
        return {"module": description.split()[0] if description else "pytest"}
    # audit/roast use --path or --target
    if command in ("audit", "roast"):
        paths = re.findall(r"[\w/]+\.py", description)
        return {"path": paths[0]} if paths else {}
    # lint uses --paths
    if command == "lint":
        paths = re.findall(r"[\w/]+\.py", description)
        return {"paths": paths} if paths else {}
    # query uses --query
    if command == "query":
        return {"query": description[:200]}

    # Default - no args
    return {}


def format_structured_plan(
    plan_text: str, tokens: int = 0, cost: float = 0.0, model: str = "minimax"
) -> dict:
    """Format a complete structured plan response."""
    return {
        "plan": plan_text,
        "steps": parse_plan_steps(plan_text),
        "tokens": tokens,
        "cost": cost,
        "model": model,
    }
