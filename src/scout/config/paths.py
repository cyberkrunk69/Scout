"""Path resolution for Scout – single source of truth for repo root and venv."""

from pathlib import Path
import os

def get_repo_root() -> Path:
    """Return the absolute path to the repository root.
    
    Assumes this file lives in scout/config/paths.py, so repo root is four levels up.
    """
    return Path(__file__).parent.parent.parent.parent

def get_venv_python() -> Path:
    """Return the path to the Python executable in the virtual environment.
    
    Checks environment variable SCOUT_VENV_PYTHON first; otherwise assumes
    <repo_root>/venv/bin/python.
    """
    env_path = os.environ.get("SCOUT_VENV_PYTHON")
    if env_path:
        return Path(env_path)
    return get_repo_root() / "venv" / "bin" / "python"

# Pre‑compute for performance
REPO_ROOT = get_repo_root()
VENV_PYTHON = get_venv_python()
