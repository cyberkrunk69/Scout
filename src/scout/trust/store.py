"""Trust store stub for scout-core.

This is a stub implementation to allow plan_store to import.
The full trust system should be extracted in a future track.
"""

from typing import Any, Dict, List, Optional
from pathlib import Path


class TrustStore:
    """Stub trust store for scout-core.
    
    This class provides a minimal interface that plan_store expects.
    The full trust system should be extracted from Vivarium in a future track.
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self._storage_path = storage_path or Path.home() / ".scout" / "trust"
        self._data: Dict[str, Any] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the trust store."""
        return self._data.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the trust store."""
        self._data[key] = value
    
    def delete(self, key: str) -> bool:
        """Delete a value from the trust store."""
        if key in self._data:
            del self._data[key]
            return True
        return False
    
    def list(self, pattern: str = "*") -> List[str]:
        """List keys matching a pattern."""
        import fnmatch
        return [k for k in self._data.keys() if fnmatch.fnmatch(k, pattern)]
    
    def save(self) -> None:
        """Persist the trust store to disk."""
        pass
    
    def load(self) -> None:
        """Load the trust store from disk."""
        pass
