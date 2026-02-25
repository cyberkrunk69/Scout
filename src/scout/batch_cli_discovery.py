from __future__ import annotations
"""Auto-discovery of CLI tool interfaces for batch sub-batch execution."""
import asyncio
import re
from typing import Optional

from scout.config.defaults import CLI_DISCOVERY_TIMEOUT

# Cache of discovered CLI interfaces
_CLI_CACHE: dict[str, dict] = {}

# Commands that can be discovered
DISCOVERABLE_COMMANDS = [
    "lint", "audit", "run", "nav", "roast", "query", "validate",
    "index", "doc_sync", "env", "git_status", "git_branch", "git_diff",
    "git_log", "brief", "ci_guard", "status", "plan", "git_show"
]

# Map CLI module names to command names
CLI_MODULE_MAP = {
    "lint": "vivarium.scout.cli.lint",
    "audit": "vivarium.scout.cli.audit",
    "run": "vivarium.scout.cli.run",
    "nav": "vivarium.scout.cli.nav",
    "roast": "vivarium.scout.cli.roast",
    "query": "vivarium.scout.cli.query",
    "validate": "vivarium.scout.cli.validate",
    "index": "vivarium.scout.cli.index",
    "doc_sync": "vivarium.scout.cli.doc_sync",
    "env": "vivarium.scout.cli.env",
    "git_status": "vivarium.scout.cli.git",
    "git_branch": "vivarium.scout.cli.git",
    "git_diff": "vivarium.scout.cli.git",
    "git_log": "vivarium.scout.cli.git",
    "git_show": "vivarium.scout.cli.git",
    "brief": "vivarium.scout.cli.brief",
    "ci_guard": "vivarium.scout.cli.ci_guard",
    "status": "vivarium.scout.cli.status",
    "plan": "vivarium.scout.cli.plan",
}


async def discover_cli_interface(command: str, venv_python: str) -> dict:
    """
    Auto-discover CLI interface by running --help.
    
    Returns:
        {
            "command": str,
            "args": {"--flag": {"type": "bool|value", "required": bool}},
            "positional": [{"name": str, "optional": bool}]
        }
    """
    if command in _CLI_CACHE:
        return _CLI_CACHE[command]
    
    # Determine the CLI module
    module = CLI_MODULE_MAP.get(command, f"vivarium.scout.cli.{command}")
    
    # Run --help
    try:
        proc = await asyncio.create_subprocess_exec(
            venv_python, "-m", module, "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=CLI_DISCOVERY_TIMEOUT)
        help_text = stdout.decode() + stderr.decode()
    except Exception as e:
        return {"command": command, "error": str(e), "args": {}, "positional": []}
    
    # Parse the help text
    interface = _parse_help_text(command, help_text)
    _CLI_CACHE[command] = interface
    return interface


def _parse_help_text(command: str, help_text: str) -> dict:
    """Parse --help output to extract argument structure."""
    args = {}
    positional = []
    
    # Split into lines
    lines = help_text.split('\n')
    
    # Pattern for optional args: --flag TEXT or --flag=TEXT or -f, --flag
    opt_arg_pattern = re.compile(r'^\s*(-[\w],?\s*)?--(\w+)(?:[=\s]([\w]+))?\s*(.*)$')
    
    # Pattern for positional args
    pos_pattern = re.compile(r'^\s*(\w+)\s*(.*)$')
    
    in_options_section = False
    
    for line in lines:
        # Detect sections
        if 'options:' in line.lower() or 'optional arguments:' in line.lower():
            in_options_section = True
            continue
        if line.strip().startswith('positional arguments:') or 'commands:' in line.lower():
            in_options_section = False
            continue
            
        # Skip empty lines and section headers
        if not line.strip() or line.strip().startswith('='):
            continue
            
        # Parse optional args (--flag)
        if in_options_section or line.strip().startswith('-'):
            match = opt_arg_pattern.match(line)
            if match:
                short_flag = match.group(1)
                long_flag = match.group(2)
                arg_type = match.group(3)
                description = match.group(4) if match.group(4) else ""
                
                if long_flag:
                    # Determine if it's a boolean flag or takes a value
                    is_bool = 'store true' in description.lower() or 'store false' in description.lower() or not arg_type
                    args[f"--{long_flag}"] = {
                        "type": "bool" if is_bool else "value",
                        "required": "required" in description.lower(),
                        "description": description.strip()[:100]
                    }
        
        # Parse positional args
        elif not in_options_section and line.strip() and not line.strip().startswith('['):
            # Heuristic: if it looks like a positional arg name
            words = line.strip().split()
            if words and words[0].replace('_', '').isalpha():
                # Check if optional
                optional = '[optional]' in line.lower() or 'default:' in line.lower()
                positional.append({"name": words[0], "optional": optional})
    
    return {
        "command": command,
        "args": args,
        "positional": positional,
        "raw": help_text[:500]  # Keep first 500 chars for debugging
    }


def map_step_to_args(step_description: str, command: str, interface: dict) -> dict:
    """
    Intelligently map a step description to valid CLI arguments.
    
    Uses the discovered interface to determine what arguments to use.
    """
    args = {}
    available_args = interface.get("args", {})
    positional = interface.get("positional", [])
    desc_lower = step_description.lower()
    
    # Extract file paths from description
    file_paths = re.findall(r'[\w\-/]+\.py\b', step_description)
    
    # Try to match description to appropriate flags
    for flag, info in available_args.items():
        flag_name = flag.replace('--', '').replace('-', '_')
        
        # Skip help flags
        if flag_name in ('help', 'h'):
            continue
            
        # File/path related flags
        if any(w in flag_name for w in ('path', 'file', 'target')) and file_paths:
            args[flag] = file_paths[0]
            
        # Query/question related
        elif any(w in flag_name for w in ('query', 'question', 'q')):
            args[flag] = step_description[:200]
            
        # Module related
        elif 'module' in flag_name and ('test' in desc_lower or 'pytest' in desc_lower):
            args[flag] = 'pytest'
            
        # Limit/count
        elif any(w in flag_name for w in ('limit', 'count', 'n')):
            nums = re.findall(r'\d+', step_description)
            if nums:
                args[flag] = int(nums[0])
    
    # Handle positional args
    if positional and file_paths:
        # First positional arg often takes a path
        args[positional[0]['name']] = file_paths[0]
    elif positional:
        # Pass the whole description as first positional
        first_pos = positional[0]['name']
        if first_pos not in args:
            args[first_pos] = step_description.split()[0] if step_description.split() else "test"
    
    return args


async def preload_interfaces(commands: list[str], venv_python: str) -> dict:
    """Pre-load CLI interfaces for multiple commands in parallel."""
    tasks = [discover_cli_interface(cmd, venv_python) for cmd in commands]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return dict(zip(commands, results))


def get_command_argv_template(command: str, args: dict) -> list[str]:
    """Convert args dict to CLI argv for a command."""
    argv = []
    for flag, value in args.items():
        if value is True:
            argv.append(flag)
        elif value is not False and value is not None:
            argv.extend([flag, str(value)])
    return argv
