"""Environment variable validation for Scout LLM providers.

Provides utilities to check if required environment variables are set
for the multi-provider LLM router.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvValidationResult:
    """Result of environment variable validation."""
    
    deepseek_key_set: bool
    minimax_key_set: bool
    scout_llm_mode: str
    is_valid: bool
    missing_required: list[str]
    warnings: list[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "deepseek_configured": self.deepseek_key_set,
            "minimax_configured": self.minimax_key_set,
            "scout_llm_mode": self.scout_llm_mode,
            "is_valid": self.is_valid,
            "missing_required": self.missing_required,
            "warnings": self.warnings,
        }


def validate_environment() -> EnvValidationResult:
    """
    Validate that required environment variables are set for LLM providers.
    
    Returns:
        EnvValidationResult with configuration status and any warnings
    """
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    minimax_key = os.environ.get("MINIMAX_API_KEY", "")
    scout_llm_mode = os.environ.get("SCOUT_LLM_MODE", "auto").lower()
    
    missing_required = []
    warnings = []
    
    # Check mode-specific requirements
    if scout_llm_mode == "free":
        if not deepseek_key:
            missing_required.append("DEEPSEEK_API_KEY (required for free mode)")
    elif scout_llm_mode == "paid":
        if not minimax_key:
            missing_required.append("MINIMAX_API_KEY (required for paid mode)")
    elif scout_llm_mode == "auto":
        if not deepseek_key and not minimax_key:
            missing_required.append("At least one of DEEPSEEK_API_KEY or MINIMAX_API_KEY")
            warnings.append("Auto mode needs at least one provider configured")
    else:
        warnings.append(f"Unknown SCOUT_LLM_MODE: {scout_llm_mode}. Using 'auto'.")
    
    # Warn about unused keys
    if deepseek_key and scout_llm_mode == "paid":
        warnings.append("DEEPSEEK_API_KEY is set but SCOUT_LLM_MODE=paid (DeepSeek unused)")
    if minimax_key and scout_llm_mode == "free":
        warnings.append("MINIMAX_API_KEY is set but SCOUT_LLM_MODE=free (MiniMax unused)")
    
    is_valid = len(missing_required) == 0
    
    return EnvValidationResult(
        deepseek_key_set=bool(deepseek_key),
        minimax_key_set=bool(minimax_key),
        scout_llm_mode=scout_llm_mode,
        is_valid=is_valid,
        missing_required=missing_required,
        warnings=warnings,
    )


def get_router_status() -> dict:
    """
    Get current router status for diagnostics.
    
    Returns a dictionary with:
    - mode: Current operating mode
    - deepseek_configured: Whether DeepSeek API key is available
    - minimax_configured: Whether MiniMax API key is available
    - deepseek_rpm_used: Current minute request count
    - deepseek_rpm_limit: Rate limit for DeepSeek
    - deepseek_reset_in: Seconds until rate limit resets
    """
    result = validate_environment()
    
    # Import rate limit state from router (if available)
    deepseek_rpm_used = 0
    deepseek_reset_in = 0
    
    try:
        from scout.router import _deepseek_calls, _deepseek_reset, DEEPSEEK_RPM
        import time
        now = time.time()
        elapsed = now - _deepseek_reset
        if elapsed < 60:
            deepseek_rpm_used = _deepseek_calls
            deepseek_reset_in = int(60 - elapsed)
    except (ImportError, AttributeError):
        pass
    
    return {
        "mode": result.scout_llm_mode,
        "deepseek_configured": result.deepseek_key_set,
        "minimax_configured": result.minimax_key_set,
        "is_valid": result.is_valid,
        "missing_required": result.missing_required,
        "warnings": result.warnings,
        "deepseek_rpm_used": deepseek_rpm_used,
        "deepseek_rpm_limit": 50,
        "deepseek_reset_in": deepseek_reset_in,
    }


if __name__ == "__main__":
    # CLI for quick diagnostics
    result = validate_environment()
    
    print("=== Scout LLM Router Environment ===")
    print(f"Mode: {result.scout_llm_mode}")
    print(f"DeepSeek: {'✓' if result.deepseek_key_set else '✗'}")
    print(f"MiniMax: {'✓' if result.minimax_key_set else '✗'}")
    print(f"Valid: {'✓' if result.is_valid else '✗'}")
    
    if result.missing_required:
        print("\nMissing Required:")
        for m in result.missing_required:
            print(f"  - {m}")
    
    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
