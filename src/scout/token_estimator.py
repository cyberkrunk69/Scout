"""Token estimation utilities for LLM requests.

Estimates token counts for prompts and responses to enable
token-based rate limiting.
"""

from __future__ import annotations

import os
import re
from typing import Optional


# Rough token estimation ratios (tokens ≈ chars / ratio)
TOKENS_PER_CHAR_ENGLISH = 4  # Conservative estimate for English
TOKENS_PER_CHAR_CODE = 3     # Code is more dense


def estimate_tokens(text: str, is_code: bool = False) -> int:
    """
    Estimate token count for text.
    
    Uses character-based approximation. For more accurate counts,
    use a proper tokenizer like tiktoken.
    
    Args:
        text: Input text
        is_code: Whether text is code (uses different ratio)
    
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    
    ratio = TOKENS_PER_CHAR_CODE if is_code else TOKENS_PER_CHAR_ENGLISH
    return max(1, len(text) // ratio)


def estimate_tokens_for_prompt(
    prompt: str,
    system: Optional[str] = None,
    messages: Optional[list] = None,
) -> int:
    """
    Estimate total tokens for a complete prompt.
    
    Includes prompt, system message, and conversation history.
    
    Args:
        prompt: Main user prompt
        system: System message (if any)
        messages: Conversation history as [{"role": ..., "content": ...}]
    
    Returns:
        Estimated total token count
    """
    total = 0
    
    # System message (typically ~50 tokens overhead + content)
    if system:
        total += estimate_tokens(system) + 50
    
    # Conversation history (each message has ~4 token overhead)
    if messages:
        for msg in messages:
            content = msg.get("content", "")
            total += estimate_tokens(content) + 4
    
    # Main prompt
    total += estimate_tokens(prompt)
    
    return total


def count_messages_tokens(messages: list) -> int:
    """Count tokens in a message list."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        # Estimate based on whether it looks like code
        is_code = bool(re.search(r"```|def |class |import ", content))
        total += estimate_tokens(content, is_code=is_code)
    return total


class TokenEstimator:
    """More accurate token estimation using tiktoken if available."""
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.model = model
        self._encoder = None
        self._load_encoder()
    
    def _load_encoder(self) -> None:
        """Try to load tiktoken encoder."""
        try:
            import tiktoken
            self._encoder = tiktoken.encoding_for_model(self.model)
        except ImportError:
            logger.warning("tiktoken not available, using approximation")
        except KeyError:
            # Fall back to cl100k_base
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                pass
    
    def estimate(self, text: str) -> int:
        """Estimate tokens using tiktoken if available."""
        if self._encoder:
            return len(self._encoder.encode(text))
        return estimate_tokens(text)
    
    def estimate_messages(self, messages: list) -> int:
        """Estimate tokens for messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            total += self.estimate(content)
            # Add overhead for role
            total += 4
        return total


import logging

logger = logging.getLogger(__name__)


# Default estimator instance
_default_estimator: Optional[TokenEstimator] = None


def get_token_estimator() -> TokenEstimator:
    """Get default token estimator."""
    global _default_estimator
    if _default_estimator is None:
        model = os.environ.get("SCOUT_TOKEN_ESTIMATOR_MODEL", "gpt-3.5-turbo")
        _default_estimator = TokenEstimator(model)
    return _default_estimator
