"""Browser automation placeholder.

This module is a stub. The full browser implementation will be added later.
To use browser features, install with: pip install scout-core[browser]
"""

def __getattr__(name):
    raise ImportError(
        f"Browser support is not yet implemented. "
        f"Install with 'pip install scout-core[browser]' to get the placeholder, "
        f"but full functionality is coming soon."
    )

# Make it clear this is a stub
__all__ = []
