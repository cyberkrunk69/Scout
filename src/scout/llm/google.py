"""
Google Gemini provider for Scout.

Supports Gemini models via the Google Generative Language API.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx

from scout.llm.pricing import estimate_cost_usd

GEMINI_MODELS = {
    "gemini-2.0-flash": {
        "context": 1_000_000,
        "output": 8_192,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
    },
    "gemini-2.5-flash": {
        "context": 1_000_000,
        "output": 8_192,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
    },
    "gemini-2.5-pro": {
        "context": 2_000_000,
        "output": 8_192,
        "cost_per_1k_input": 0.00125,
        "cost_per_1k_output": 0.005,
    },
    "gemini-flash-latest": {
        "context": 1_000_000,
        "output": 8_192,
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
    },
}


def _get_google_api_key() -> Optional[str]:
    """Get Google API key from environment."""
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


async def call_google_async(
    prompt: str,
    system: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> tuple[str, float]:
    """
    Call Google Gemini API asynchronously.
    
    Args:
        prompt: The user prompt
        system: Optional system prompt
        model: Model name (default: gemini-2.0-flash)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Tuple of (response_text, cost_usd)
        
    Raises:
        ValueError: If API key is missing or model is unsupported
        httpx.HTTPError: On API errors
    """
    api_key = _get_google_api_key()
    if not api_key:
        raise ValueError(
            "Google API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
        )
    
    model_config = GEMINI_MODELS.get(model)
    if not model_config:
        raise ValueError(
            f"Unsupported model: {model}. Supported models: {list(GEMINI_MODELS.keys())}"
        )
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    
    payload = {
        "contents": [{"role": msg["role"], "parts": [{"text": msg["content"]}]} for msg in messages],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
            "topK": 40,
        },
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload, params=params)
        
        if response.status_code == 401:
            raise ValueError("Invalid Google API key")
        if response.status_code == 429:
            raise ValueError("Google API rate limit exceeded")
        if response.status_code != 200:
            raise httpx.HTTPError(
                f"Google API error {response.status_code}: {response.text}"
            )
        
        data = response.json()
    
    if "candidates" not in data or not data["candidates"]:
        raise ValueError(f"Empty response from Gemini: {data}")
    
    content = data["candidates"][0]["content"]["parts"][0]["text"]
    
    from scout.llm.cost import MODEL_COSTS
    
    usage = data.get("usageMetadata", {})
    input_tokens = usage.get("promptTokenCount", 0)
    output_tokens = usage.get("candidatesTokenCount", 0)
    
    model_config = MODEL_COSTS.get(model, {})
    cost_per_1k_input = model_config.get("cost_per_1k_input", 0.0)
    cost_per_1k_output = model_config.get("cost_per_1k_output", 0.0)
    
    cost = (input_tokens / 1000) * cost_per_1k_input + (output_tokens / 1000) * cost_per_1k_output
    
    return content, cost


async def call_google_async_with_retry(
    prompt: str,
    system: Optional[str] = None,
    model: str = "gemini-2.0-flash",
    max_tokens: int = 2048,
    temperature: float = 0.0,
    max_retries: int = 3,
    context: "LLMCallContext" = None,
    estimated_cost: float = 0.0,
) -> tuple[str, float]:
    """
    Call Google Gemini API with unified retry wrapper.
    
    Args:
        prompt: The user prompt
        system: Optional system prompt
        model: Model name (default: gemini-2.0-flash)
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
        return await call_google_async(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    if context is None:
        context = LLMCallContext(
            model=model,
            provider="google",
            operation="call_google",
        )
    
    return await call_with_retries(
        _do_call,
        context=context,
        estimated_cost=estimated_cost,
        max_retries=max_retries,
        cost_extractor=lambda r: r[1] if isinstance(r, tuple) else 0.0,
    )
