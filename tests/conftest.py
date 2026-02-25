"""Pytest configuration for scout tests."""
import sys
from pathlib import Path

# Add src to path for the tests - conftest is in tests/, so parent.parent is project root
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"

# Insert at the very beginning to override any other paths
sys.path.insert(0, str(src_path))
