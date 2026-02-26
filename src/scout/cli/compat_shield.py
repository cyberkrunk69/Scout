#!/usr/bin/env python
"""
Compat Shield - Make Python 3.10+ code work on Python 3.9

Adds 'from __future__ import annotations' and fixes isinstance/issubclass with | syntax.
"""

import os
import re
from pathlib import Path


def apply_compat_shield(directory: str, dry_run: bool = False):
    """
    Programmatically converts 3.10+ code to be 3.9 compatible.
    1. Adds 'from __future__ import annotations' to all files.
    2. Converts 'isinstance(x, (A, B))' to 'isinstance(x, (A, B))'.
    """
    print(f"🛡️ Applying Compat Shield to: {directory}")
    
    fixed_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                path = Path(root) / file
                with open(path, "r") as f:
                    content = f.read()
                
                original = content
                
                # --- STEP 1: Inject Future Annotations ---
                if "from __future__ import annotations" not in content:
                    content = "from __future__ import annotations\n" + content
                
                # --- STEP 2: Fix Runtime Type Checks ---
                # Converts 'int | str' to '(int, str)' ONLY inside isinstance/issubclass
                # Pattern: isinstance(var, (Type, Type, Type))
                def fix_union_types(match):
                    func_name = match.group(1)
                    var = match.group(2)
                    types = match.group(3)
                    # Split by | and rebuild as tuple
                    type_list = [t.strip() for t in types.split('|')]
                    tuple_str = "(" + ", ".join(type_list) + ")"
                    return f"{func_name}({var}, {tuple_str})"
                
                content = re.sub(
                    r"(isinstance|issubclass)\(([^,]+),\s*([A-Za-z_][A-Za-z0-9_\s|]+)\)",
                    fix_union_types,
                    content
                )
                
                if content != original:
                    if not dry_run:
                        with open(path, "w") as f:
                            f.write(content)
                    print(f"  ✅ Fixed: {path}")
                    fixed_count += 1
    
    print(f"\n🛡️ Done! Fixed {fixed_count} files.")
    if dry_run:
        print("  (dry run - no files modified)")


if __name__ == "__main__":
    apply_compat_shield("vivarium/scout")
