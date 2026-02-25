from __future__ import annotations
"""Path validator for ensuring planner suggestions match existing files."""
from pathlib import Path
from typing import Optional
import re

class PathValidator:
    """
    Validate suggested paths against actual filesystem state.
    
    Prevents planner from suggesting files that already exist.
    """
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
    
    def validate_suggested_path(self, suggested_path: str) -> dict:
        """
        Check if a suggested path already exists.
        
        Returns:
            {
                "path": str,
                "exists": bool,
                "suggestion": str or None  # Alternative path if conflict
            }
        """
        # Normalize the path
        normalized = self._normalize_path(suggested_path)
        full_path = self.repo_root / normalized
        
        exists = full_path.exists()
        
        result = {
            "path": normalized,
            "exists": exists,
            "suggestion": None
        }
        
        if exists:
            # Find alternative
            result["suggestion"] = self._find_alternative(normalized)
        
        return result
    
    def validate_paths_batch(self, paths: list[str]) -> list[dict]:
        """Validate multiple paths at once."""
        return [self.validate_suggested_path(p) for p in paths]
    
    def extract_paths_from_plan(self, plan_text: str) -> list[str]:
        """
        Extract file path suggestions from plan text.
        
        Looks for patterns like:
        - "Create X at path/to/file.py"
        - "File: path/to/file.py"
        - "path/to/file.py" (in code blocks)
        """
        paths = []
        
        # Pattern for quoted paths
        quote_pattern = re.compile(r'["\']([a-zA-Z0-9_/.-]+.py)["\']')
        paths.extend(quote_pattern.findall(plan_text))
        
        # Pattern for "at path/to/file.py"
        at_pattern = re.compile(r'at\s+([a-zA-Z0-9_/.-]+.py)')
        paths.extend(at_pattern.findall(plan_text))
        
        # Pattern for "path/to/file.py" in code blocks
        code_pattern = re.compile(r'```\w*\n([a-zA-Z0-9_/.-]+.py)\n```')
        paths.extend(code_pattern.findall(plan_text))
        
        # Dedupe
        return list(set(paths))
    
    def validate_plan_paths(self, plan_text: str) -> dict:
        """
        Validate all suggested paths in a plan.
        
        Returns:
            {
                "all_valid": bool,
                "conflicts": [list of conflicting paths],
                "validated": [list of validation results]
            }
        """
        suggested_paths = self.extract_paths_from_plan(plan_text)
        validations = self.validate_paths_batch(suggested_paths)
        
        conflicts = [v for v in validations if v["exists"]]
        
        return {
            "all_valid": len(conflicts) == 0,
            "conflicts": [c["path"] for c in conflicts],
            "validated": validations
        }
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path separators and leading slashes."""
        # Remove leading slash
        path = path.lstrip('/')
        # Normalize separators
        path = path.replace('\\', '/')
        return path
    
    def _find_alternative(self, original: str) -> str:
        """Find an alternative path that doesn't exist."""
        # Try adding _v2, _new, etc.
        stem = Path(original).stem
        suffix = Path(original).suffix
        parent = str(Path(original).parent)
        
        for suffix_str in ['_v2', '_new', '_alt']:
            alt_name = f"{stem}{suffix_str}{suffix}"
            alt_path = f"{parent}/{alt_name}" if parent else alt_name
            if not (self.repo_root / alt_path).exists():
                return alt_path
        
        # Fallback: just note it exists
        return f"{original} (EXISTS - use existing or specify new name)"


def validate_plan_paths(plan_text: str, repo_root: str = ".") -> dict:
    """Convenience function to validate paths in a plan."""
    validator = PathValidator(Path(repo_root))
    return validator.validate_plan_paths(plan_text)
