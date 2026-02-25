"""
MiniMax LLM provider — Anthropic-compatible API for cheap lint fixes.

Uses MINIMAX_API_KEY. Base URL: https://api.minimax.io/anthropic
Pricing: ~$0.30/M input, $1.20/M output (Feb 2026).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

MINIMAX_INPUT_PER_M = 0.30
MINIMAX_OUTPUT_PER_M = 1.20


@dataclass
class MiniMaxResult:
    """Result from a MiniMax API call with full cost details."""
    response_text: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


async def call_minimax_async(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    model: Optional[str] = None,
) -> tuple[str, float]:
    """
    Call MiniMax via Anthropic-compatible API.

    Args:
        prompt: The prompt to send to the model.
        system: Optional system prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Temperature for generation (0.0-1.0).
        model: Model to use. Defaults to MINIMAX_MODEL env var or "MiniMax-M2".
               Available: MiniMax-M2.5, MiniMax-M2.5-highspeed,
               MiniMax-M2.1, MiniMax-M2.1-highspeed, MiniMax-M2.

    Returns (response_text, cost_usd).
    """
    result = await call_minimax_async_detailed(
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
    )
    return result.response_text, result.cost_usd


async def call_minimax_async_detailed(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    model: Optional[str] = None,
) -> MiniMaxResult:
    """
    Call MiniMax via Anthropic-compatible API with full cost details.

    Args:
        prompt: The prompt to send to the model.
        system: Optional system prompt.
        max_tokens: Maximum tokens to generate.
        temperature: Temperature for generation (0.0-1.0).
        model: Model to use. Defaults to MINIMAX_MODEL env var or "MiniMax-M2".
               Available: MiniMax-M2.5, MiniMax-M2.5-highspeed,
               MiniMax-M2.1, MiniMax-M2.1-highspeed, MiniMax-M2.

    Returns MiniMaxResult with full cost breakdown.
    """
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MINIMAX_API_KEY missing. Set it in .env or environment."
        )

    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise RuntimeError(
            "anthropic required for MiniMax. pip install anthropic"
        )

    base_url = os.environ.get(
        "MINIMAX_BASE_URL", "https://api.minimax.io/anthropic"
    )
    default_model = os.environ.get("MINIMAX_MODEL", "MiniMax-M2")
    effective_model = model if model else default_model

    client = AsyncAnthropic(
        base_url=base_url,
        api_key=api_key,
    )

    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {
        "model": effective_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system

    message = await client.messages.create(**kwargs)

    text = ""
    thinking_text = None
    
    if message.content:
        for block in message.content:
            if hasattr(block, "thinking"):
                thinking_text = block.thinking
            if hasattr(block, "text"):
                text += block.text
            elif isinstance(block, dict):
                if "thinking" in block:
                    thinking_text = block["thinking"]
                if "text" in block:
                    text += block["text"]
    
    if not text and thinking_text:
        import re
        quoted_words = re.findall(r'"([^"]+)"', thinking_text)
        
        if quoted_words:
            text = quoted_words[-1].strip()
            text = text.rstrip('.')

    usage = getattr(message, "usage", None)
    if usage:
        input_t = getattr(usage, "input_tokens", None) or (
            usage.get("input_tokens") if isinstance(usage, dict) else 0
        )
        output_t = getattr(usage, "output_tokens", None) or (
            usage.get("output_tokens") if isinstance(usage, dict) else 0
        )
    else:
        input_t = len(prompt) // 4
        output_t = len(text) // 4

    cost = (input_t / 1_000_000) * MINIMAX_INPUT_PER_M + (
        output_t / 1_000_000
    ) * MINIMAX_OUTPUT_PER_M

    return MiniMaxResult(
        response_text=text.strip(),
        cost_usd=cost,
        input_tokens=input_t,
        output_tokens=output_t,
        model=effective_model,
    )
