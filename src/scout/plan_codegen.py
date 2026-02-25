# TODO: This module is not yet integrated into the main application.
# It is planned for future use as part of the Plan Execution framework.
# See ADR-007 for design context.
from __future__ import annotations
#!/usr/bin/env python
"""
Plan Code Generator - Convert plan steps into executable Python code.

This module takes structured plan steps (from scout_plan with structured=True)
and generates Python code that executes them using Scout MCP tools.
"""

import json


def generate_executor(plan_steps: list[dict]) -> str:
    """
    Generate Python code that executes the given plan steps.
    Returns a string of Python code.

    Args:
        plan_steps: List of step dicts with keys: id, description, command, args

    Returns:
        String of Python code that can be executed to run the plan
    """
    lines = [
        '"""Auto-generated executor for plan steps."""',
        "import json",
        "",
        "async def execute_plan_steps(context: dict) -> list[dict]:",
        '    """Execute all plan steps and return results."""',
        "    results = []",
        "",
    ]

    for i, step in enumerate(plan_steps):
        step_id = step.get("id", i + 1)
        description = step.get("description", "")
        command = step.get("command", "")
        args = step.get("args", {})

        lines.append(f"    # Step {step_id}: {description}")
        lines.append(f"    # Command: {command}")
        lines.append(f"    step_{i}_args = {json.dumps(args, indent=8)}")

        code_lines = _generate_step_code(command, args, i)
        lines.extend(code_lines)
        lines.append("")

    lines.append("    return results")
    lines.append("")
    lines.append("# For direct execution:")
    lines.append("if __name__ == '__main__':")
    lines.append("    import asyncio")
    lines.append("    results = asyncio.run(execute_plan_steps({}))")
    lines.append("    print(json.dumps(results, indent=2))")

    return "\n".join(lines)


def _generate_step_code(command: str, args: dict, step_index: int) -> list[str]:
    """Generate code for a single step based on its command type."""
    code = []

    if command == "lint":
        path = args.get("path", ".")
        fix = args.get("fix", False)
        code.append(
            f'    result_{step_index} = await scout_lint(path="{path}", fix={fix})'
        )

    elif command == "write_file":
        file_path = args.get("file_path", args.get("file", ""))
        content = args.get("content", "")
        escaped_content = content.replace('"""', '\\"\\"\\"')
        code.append(f"    result_{step_index} = await scout_write_with_review(")
        code.append(f'        file_path="{file_path}",')
        code.append(f'        content="""{escaped_content}"""')
        code.append("    )")

    elif command == "edit_file":
        file_path = args.get("file_path", args.get("file", ""))
        old_string = args.get("old_string", args.get("old", ""))
        new_string = args.get("new_string", args.get("new", ""))
        reason = args.get("reason", "Executing plan step")
        escaped_old = old_string.replace('"""', '\\"\\"\\"')
        escaped_new = new_string.replace('"""', '\\"\\"\\"')
        code.append(f"    result_{step_index} = await scout_edit(")
        code.append(f'        file_path="{file_path}",')
        code.append(f'        old_string="""{escaped_old}""",')
        code.append(f'        new_string="""{escaped_new}""",')
        code.append(f'        reason="""{reason}"""')
        code.append("    )")

    elif command == "run_command" or command == "run":
        module = args.get("module", "")
        run_args = args.get("args", [])
        code.append(f"    result_{step_index} = await scout_run(")
        code.append(f'        module="{module}",')
        code.append(f"        args={run_args}")
        code.append("    )")

    elif command == "git_commit":
        message = args.get("message", "")
        allow_empty = args.get("allow_empty", False)
        escaped_msg = message.replace('"""', '\\"\\"\\"')
        code.append(f"    result_{step_index} = await scout_git_commit(")
        code.append(f'        message="""{escaped_msg}""",')
        code.append(f"        allow_empty={allow_empty})")

    elif command == "git_add":
        paths = args.get("paths", [])
        all_files = args.get("all_files", False)
        code.append(f"    result_{step_index} = await scout_git_add(")
        code.append(f"        paths={paths},")
        code.append(f"        all_files={all_files})")

    elif command == "delete_file":
        file_path = args.get("file_path", args.get("file", ""))
        reason = args.get("reason", "Executing plan step")
        code.append(f"    result_{step_index} = await scout_delete_with_review(")
        code.append(f'        file_path="{file_path}",')
        code.append(f'        reason="""{reason}"""')
        code.append("    )")

    elif command == "shell":
        shell_cmd = args.get("command", "")
        timeout = args.get("timeout", 30)
        code.append(f"    result_{step_index} = await scout_shell(")
        code.append(f'        command="""{shell_cmd}""",')
        code.append(f"        timeout={timeout})")

    elif command == "nav":
        task = args.get("task", "")
        query = args.get("query", "")
        code.append(f"    result_{step_index} = await scout_nav(")
        code.append(f'        task="""{task}""",')
        if query:
            code.append(f'        query="""{query}"""')
        else:
            code.append("        query=None")
        code.append("    )")

    elif command == "query":
        query_text = args.get("query", "")
        scope = args.get("scope")
        code.append(f"    result_{step_index} = await scout_query(")
        code.append(f'        query="""{query_text}""",')
        if scope:
            code.append(f'        scope="{scope}"')
        else:
            code.append("        scope=None")
        code.append("    )")

    elif command == "doc_sync":
        command_arg = args.get("command", "status")
        target = args.get("target")
        code.append(f"    result_{step_index} = await scout_doc_sync(")
        code.append(f'        command="{command_arg}",')
        if target:
            code.append(f'        target="{target}"')
        else:
            code.append("        target=None")
        code.append("    )")

    elif command == "validate_module":
        module_path = args.get("module_path", "")
        type_check = args.get("type_check", True)
        test = args.get("test", True)
        code.append(f"    result_{step_index} = await scout_validate_module(")
        code.append(f'        module_path="{module_path}",')
        code.append(f"        type_check={type_check},")
        code.append(f"        test={test})")

    else:
        code.append(f"    # Unknown command: {command}")
        code.append(f"    result_{step_index} = {{")
        code.append('        "status": "skipped",')
        code.append(f'        "error": "Unknown command: {command}"')
        code.append("    }}")

    # Add result tracking - single line to avoid linter issues
    code.append(
        '    results.append({"step": '
        + str(step_index)
        + ', "command": "'
        + command
        + '", "result": result_'
        + str(step_index)
        + "})"
    )

    return code


def generate_step_mapping() -> dict[str, str]:
    """Generate a mapping of command names to Scout tool function names."""
    return {
        "lint": "scout_lint",
        "write_file": "scout_write_with_review",
        "edit_file": "scout_edit",
        "run_command": "scout_run",
        "run": "scout_run",
        "git_commit": "scout_git_commit",
        "git_add": "scout_git_add",
        "delete_file": "scout_delete_with_review",
        "shell": "scout_shell",
        "nav": "scout_nav",
        "query": "scout_query",
        "doc_sync": "scout_doc_sync",
        "validate_module": "scout_validate_module",
    }


if __name__ == "__main__":
    sample_steps = [
        {
            "id": 1,
            "description": "Create auth middleware",
            "command": "write_file",
            "args": {
                "file_path": "auth/middleware.py",
                "content": "async def auth_middleware(request):\n    return True",
            },
        },
        {
            "id": 2,
            "description": "Run linter on new file",
            "command": "lint",
            "args": {"path": "auth/middleware.py"},
        },
        {
            "id": 3,
            "description": "Commit changes",
            "command": "git_commit",
            "args": {"message": "feat: add auth middleware"},
        },
    ]

    code = generate_executor(sample_steps)
    print(code)
