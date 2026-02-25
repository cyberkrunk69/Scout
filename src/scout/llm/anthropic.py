"""
Anthropic Claude provider for Scout.

Supports Claude models via the Anthropic Messages API.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from scout.llm.pricing import estimate_cost_usd

CLAUDE_MODELS = {
    "claude-3-5-sonnet-20241022": {
        "context": 200_000,
        "output": 8192,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
    },
    "claude-3-5-sonnet": {
        "context": 200_000,
        "output": 8192,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
    },
    "claude-3-opus": {
        "context": 200_000,
        "output": 4096,
        "cost_per_1k_input": 0.015,
        "cost_per_1k_output": 0.075,
    },
    "claude-3-sonnet": {
        "context": 200_000,
        "output": 4096,
        "cost_per_1k_input": 0.003,
        "cost_per_1k_output": 0.015,
    },
    "claude-3-haiku": {
        "context": 200_000,
        "output": 4096,
        "cost_per_1k_input": 0.00025,
        "cost_per_1k_output": 0.00125,
    },
}


def _get_anthropic_api_key() -> Optional[str]:
    """Get Anthropic API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY")


async def call_anthropic_async(
    prompt: str,
    system: Optional[str] = None,
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> tuple[str, float]:
    """
    Call Anthropic Claude API asynchronously.
    
    Args:
        prompt: The user prompt
        system: Optional system prompt
        model: Model name (default: claude-3-5-sonnet-20241022)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Tuple of (response_text, cost_usd)
        
    Raises:
        ValueError: If API key is missing or model is unsupported
        httpx.HTTPError: On API errors
    """
    api_key = _get_anthropic_api_key()
    if not api_key:
        raise ValueError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable."
        )
    
    model_config = CLAUDE_MODELS.get(model)
    if not model_config:
        matching = [k for k in CLAUDE_MODELS.keys() if k.startswith(model)]
        if matching:
            model = matching[0]
            model_config = CLAUDE_MODELS[model]
        else:
            raise ValueError(
                f"Unsupported model: {model}. Supported models: {list(CLAUDE_MODELS.keys())}"
            )
    
    messages = [{"role": "user", "content": prompt}]
    
    url = "https://api.anthropic.com/v1/messages"
    
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": messages,
    }
    
    if system:
        payload["system"] = system
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        
        if response.status_code == 401:
            raise ValueError("Invalid Anthropic API key")
        if response.status_code == 429:
            raise ValueError("Anthropic API rate limit exceeded")
        if response.status_code == 400:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
            raise ValueError(f"Anthropic API error: {error_msg}")
        if response.status_code != 200:
            raise httpx.HTTPError(
                f"Anthropic API error {response.status_code}: {response.text}"
            )
        
        data = response.json()
    
    if "content" not in data or not data["content"]:
        raise ValueError(f"Empty response from Anthropic: {data}")
    
    text_content = None
    for block in data["content"]:
        if block.get("type") == "text":
            text_content = block["text"]
            break
    
    if not text_content:
        raise ValueError(f"No text content in Anthropic response: {data}")
    
    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    
    cost = estimate_cost_usd(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )
    
    return text_content, cost


async def call_anthropic_async_with_retry(
    prompt: str,
    system: Optional[str] = None,
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_retries: int = 3,
    context: "LLMCallContext" = None,
    estimated_cost: float = 0.0,
) -> tuple[str, float]:
    """
    Call Anthropic API with unified retry wrapper.
    
    Args:
        prompt: The user prompt
        system: Optional system prompt
        model: Model name (default: claude-3-5-sonnet-20241022)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        max_retries: Max retry attempts
        context: LLMCallContext for budget/audit/correlation (optional)
        estimated_cost: Cost estimate per attempt
    
    Returns:
        Tuple of (response_text, cost_usd)
    """
    from scout.llm.retry import call_with_retries, LLMCallContext
    
    async def _do_call():
        return await call_anthropic_async(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    if context is None:
        context = LLMCallContext(
            model=model,
            provider="anthropic",
            operation="call_anthropic",
        )
    
    return await call_with_retries(
        _do_call,
        context=context,
        estimated_cost=estimated_cost,
        max_retries=max_retries,
        cost_extractor=lambda r: r[1] if isinstance(r, tuple) else 0.0,
    )
