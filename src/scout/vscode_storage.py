"""VS Code Extension Storage Utilities.

Provides utilities for locating and accessing VS Code extension storage paths
for various AI assistant extensions (Copilot, Cline, Continue).
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from scout.config import (
    VSCODE_GLOBAL_STORAGE,
    VSCODE_WORKSPACE_STORAGE,
    PARSER_MAX_RETRIES,
    PARSER_RETRY_DELAY,
)

logger = logging.getLogger(__name__)


def normalise_path(path: str | Path) -> Path:
    """Normalize a path to a canonical Path object.

    Resolves relative paths relative to home directory,
    expands user ~, and makes absolute.

    Args:
        path: Path string or Path object

    Returns:
        Normalized absolute Path object
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.is_absolute():
        path = path.expanduser()

    return path.resolve()


def locate_vscode_storage() -> Dict[str, Optional[Path]]:
    """Locate VS Code global and workspace storage directories.

    Returns:
        Dict with 'globalStorage' and 'workspaceStorage' paths.
        Paths may be None if not found.
    """
    result = {
        "globalStorage": None,
        "workspaceStorage": None,
    }

    # Find VS Code data directory
    # macOS: ~/Library/Application Support/Code
    # Linux: ~/.config/Code
    # Windows: %APPDATA%\Code
    if os.name == "darwin":
        base = Path.home() / "Library" / "Application Support" / "Code"
    elif os.name == "nt":
        base = Path(os.environ.get("APPDATA", "")) / "Code"
    else:
        base = Path.home() / ".config" / "Code"

    global_storage = base / VSCODE_GLOBAL_STORAGE
    workspace_storage = base / VSCODE_WORKSPACE_STORAGE

    if global_storage.exists():
        result["globalStorage"] = global_storage

    if workspace_storage.exists():
        result["workspaceStorage"] = workspace_storage

    logger.debug(f"VS Code storage: global={result['globalStorage']}, workspace={result['workspaceStorage']}")
    return result


def locate_workspace_storage(workspace_hash: Optional[str] = None) -> Optional[Path]:
    """Locate a specific workspace storage directory.

    Args:
        workspace_hash: Optional hash to locate specific workspace.
                       If None, returns first available workspace.

    Returns:
        Path to workspace storage or None if not found.
    """
    storage = locate_vscode_storage()
    workspace_storage = storage.get("workspaceStorage")

    if not workspace_storage or not workspace_storage.exists():
        return None

    if workspace_hash:
        # Look for specific workspace by hash
        target = workspace_storage / workspace_hash
        if target.exists():
            return target
        return None

    # Return first available workspace
    workspaces = sorted(workspace_storage.iterdir())
    if workspaces:
        return workspaces[0]

    return None


def parse_json_file(file_path: Path, retries: int = PARSER_MAX_RETRIES) -> Optional[Dict[str, Any]]:
    """Parse a JSON file with retry logic and error logging.

    Args:
        file_path: Path to JSON file
        retries: Number of retry attempts

    Returns:
        Parsed JSON as dict, or None if failed
    """
    last_error: Optional[Exception] = None

    for attempt in range(retries):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"JSON decode error in {file_path} (attempt {attempt + 1}/{retries}): {e}")
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except Exception as e:
            last_error = e
            logger.warning(f"Error reading {file_path} (attempt {attempt + 1}/{retries}): {e}")

        if attempt < retries - 1:
            import time
            time.sleep(PARSER_RETRY_DELAY * (attempt + 1))

    logger.error(f"Failed to parse {file_path} after {retries} attempts: {last_error}")
    return None


def hash_workspace_path(workspace_path: str | Path) -> str:
    """Generate a hash identifier for a workspace path.

    Args:
        workspace_path: Path to workspace folder

    Returns:
        Hash string suitable for VS Code workspace storage directory
    """
    workspace_path = normalise_path(workspace_path)
    # VS Code uses a specific hashing scheme, but a simple MD5 works for lookup
    return hashlib.md5(str(workspace_path).encode()).hexdigest()


def get_extension_storage(extension_id: str, global_only: bool = False) -> Optional[Path]:
    """Get storage path for a specific VS Code extension.

    Args:
        extension_id: Extension identifier (e.g., 'github.copilot')
        global_only: If True, only check global storage

    Returns:
        Path to extension storage or None
    """
    storage = locate_vscode_storage()

    # Check global storage first
    global_storage = storage.get("globalStorage")
    if global_storage:
        ext_path = global_storage / extension_id
        if ext_path.exists():
            return ext_path

    if global_only:
        return None

    # Check workspace storage
    workspace_storage = storage.get("workspaceStorage")
    if workspace_storage:
        for workspace in workspace_storage.iterdir():
            if workspace.is_dir():
                ext_path = workspace / extension_id
                if ext_path.exists():
                    return ext_path

    return None
