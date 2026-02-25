"""Safety guards for primitive tools."""

from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class SafetyViolation(Exception):
    """Raised when a safety check fails."""
    pass


class SafetyGuard:
    """Safety guards for workspace protection."""
    
    def __init__(self, workspace_root: Path, dry_run: bool = False):
        self.workspace_root = workspace_root.resolve()
        self.dry_run = dry_run
        
    def validate_path(self, path: str) -> Path:
        """Ensure path is within workspace boundary."""
        resolved = (self.workspace_root / path).resolve()
        if not str(resolved).startswith(str(self.workspace_root)):
            raise SafetyViolation(f"Path {path} outside workspace boundary")
        return resolved
        
    def validate_command(self, cmd: str, whitelist: List[str]) -> bool:
        """Check command against whitelist."""
        base_cmd = cmd.split()[0]
        if base_cmd not in whitelist:
            raise SafetyViolation(f"Command {base_cmd} not in whitelist: {whitelist}")
        return True
        
    def check_dry_run(self) -> bool:
        """If dry-run mode, simulate but don't execute."""
        return self.dry_run
        
    def check_depth(self, path: str, max_depth: int = 10) -> bool:
        """Check if path depth exceeds maximum."""
        depth = len(Path(path).parts)
        if depth > max_depth:
            raise SafetyViolation(f"Path depth {depth} exceeds maximum {max_depth}")
        return True


# Default command whitelist for scout_command
DEFAULT_COMMAND_WHITELIST = [
    "git", "npm", "python", "python3", "pip", "pip3", "uv", "docker",
    "ruff", "pytest", "python3 -m pytest", "node", "yarn", "pnpm"
]


# Primitive tool implementations

async def scout_mkdir(path: str, parents: bool = False, guard: SafetyGuard = None) -> dict:
    """Create directory."""
    if guard and guard.check_dry_run():
        return {"success": True, "path_created": path, "dry_run": True}
    
    from pathlib import Path
    target = guard.validate_path(path) if guard else Path(path)
    
    try:
        if parents:
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(exist_ok=False)
        return {"success": True, "path_created": str(target)}
    except FileExistsError:
        return {"success": True, "path_created": str(target), "note": "already exists"}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def scout_remove(path: str, recursive: bool = False, force: bool = False, guard: SafetyGuard = None) -> dict:
    """Remove file or directory."""
    if guard and guard.check_dry_run():
        return {"success": True, "path_removed": path, "dry_run": True}
    
    from pathlib import Path
    import shutil
    
    target = guard.validate_path(path) if guard else Path(path)
    
    if not target.exists():
        return {"success": False, "error": "Path does not exist"}
    
    try:
        if target.is_dir():
            if recursive:
                shutil.rmtree(target)
            else:
                target.rmdir()
        else:
            target.unlink()
        return {"success": True, "path_removed": str(target)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def scout_copy(source: str, destination: str, overwrite: bool = False, guard: SafetyGuard = None) -> dict:
    """Copy file or directory."""
    if guard and guard.check_dry_run():
        return {"success": True, "source": source, "destination": destination, "dry_run": True}
    
    from pathlib import Path
    import shutil
    
    src = guard.validate_path(source) if guard else Path(source)
    dst = guard.validate_path(destination) if guard else Path(destination)
    
    if not src.exists():
        return {"success": False, "error": "Source does not exist"}
    
    if dst.exists() and not overwrite:
        return {"success": False, "error": "Destination exists and overwrite=False"}
    
    try:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=overwrite)
        else:
            shutil.copy2(src, dst)
        return {"success": True, "source": str(src), "destination": str(dst)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def scout_move(source: str, destination: str, overwrite: bool = False, guard: SafetyGuard = None) -> dict:
    """Move/rename file or directory."""
    if guard and guard.check_dry_run():
        return {"success": True, "source": source, "destination": destination, "dry_run": True}
    
    from pathlib import Path
    import shutil
    
    src = guard.validate_path(source) if guard else Path(source)
    dst = guard.validate_path(destination) if guard else Path(destination)
    
    if not src.exists():
        return {"success": False, "error": "Source does not exist"}
    
    if dst.exists() and not overwrite:
        return {"success": False, "error": "Destination exists and overwrite=False"}
    
    try:
        shutil.move(str(src), str(dst))
        return {"success": True, "source": str(src), "destination": str(dst)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def scout_list(path: str = ".", pattern: str = None, recursive: bool = False, max_depth: int = 3, guard: SafetyGuard = None) -> dict:
    """List directory contents."""
    from pathlib import Path
    import fnmatch
    
    target = guard.validate_path(path) if guard else Path(path)
    
    if not target.exists():
        return {"success": False, "error": "Path does not exist"}
    
    if not target.is_dir():
        return {"success": False, "error": "Path is not a directory"}
    
    entries = []
    try:
        if recursive:
            for p in target.rglob("*"):
                if max_depth and len(p.relative_to(target).parts) > max_depth:
                    continue
                if pattern and not fnmatch.fnmatch(p.name, pattern):
                    continue
                entries.append({"path": str(p.relative_to(target)), "type": "dir" if p.is_dir() else "file"})
        else:
            for p in target.iterdir():
                if pattern and not fnmatch.fnmatch(p.name, pattern):
                    continue
                entries.append({"name": p.name, "type": "dir" if p.is_dir() else "file"})
        
        return {"success": True, "entries": entries, "count": len(entries)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def scout_read_file(path: str, encoding: str = "utf-8", max_size_kb: int = 1024, guard: SafetyGuard = None) -> dict:
    """Read file content."""
    from pathlib import Path
    
    target = guard.validate_path(path) if guard else Path(path)
    
    if not target.exists():
        return {"success": False, "error": "File does not exist"}
    
    if target.is_dir():
        return {"success": False, "error": "Path is a directory"}
    
    size = target.stat().st_size
    if size > max_size_kb * 1024:
        return {"success": False, "error": f"File exceeds max size of {max_size_kb}KB"}
    
    try:
        content = target.read_text(encoding=encoding)
        return {"success": True, "content": content, "size": size, "encoding": encoding}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def scout_write_file(path: str, content: str, append: bool = False, guard: SafetyGuard = None) -> dict:
    """Write file content."""
    if guard and guard.check_dry_run():
        return {"success": True, "path_written": path, "dry_run": True}
    
    from pathlib import Path
    
    target = guard.validate_path(path) if guard else Path(path)
    
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        
        if append:
            target.write_text(content, append=True)
        else:
            target.write_text(content)
        
        bytes_written = len(content.encode('utf-8'))
        return {"success": True, "path_written": str(target), "bytes_written": bytes_written}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def scout_command(command: str, args: List[str] = None, timeout: int = 30, cwd: str = None, guard: SafetyGuard = None) -> dict:
    """Run shell command."""
    import subprocess
    
    if guard:
        guard.validate_command(command, DEFAULT_COMMAND_WHITELIST)
    
    full_cmd = [command] + (args or [])
    
    try:
        result = subprocess.run(
            full_cmd,
            capture_output=True,
            timeout=timeout,
            cwd=cwd,
            text=True
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"success": False, "error": str(e), "exit_code": -1}


async def scout_check_command(command: str) -> dict:
    """Verify command exists."""
    import shutil
    
    cmd_path = shutil.which(command)
    if cmd_path:
        return {"success": True, "available": True, "path": cmd_path}
    return {"success": True, "available": False}


async def scout_wait(seconds: int = None, condition: str = None, condition_params: dict = None, max_wait: int = 60) -> dict:
    """Wait until condition is met."""
    import asyncio
    import time
    
    if seconds:
        await asyncio.sleep(min(seconds, max_wait))
        return {"success": True, "elapsed": min(seconds, max_wait), "condition_met": True}
    
    # TODO: Implement condition-based waiting
    return {"success": True, "elapsed": 0, "condition_met": False, "note": "condition waiting not implemented"}


async def scout_condition(type: str, params: dict) -> dict:
    """Evaluate simple condition."""
    from pathlib import Path
    
    if type == "file_exists":
        path = params.get("path")
        exists = Path(path).exists() if path else False
        return {"success": True, "result": exists, "details": f"file_exists: {path}"}
    
    if type == "file_contains":
        path = params.get("path")
        pattern = params.get("pattern", "")
        if not path or not Path(path).exists():
            return {"success": False, "result": False, "details": "file not found"}
        
        content = Path(path).read_text()
        found = pattern in content
        return {"success": True, "result": found, "details": f"pattern '{pattern}' {'found' if found else 'not found'}"}
    
    return {"success": False, "result": False, "details": f"unknown condition type: {type}"}
