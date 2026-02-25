"""Example provider for Scout - dummy provider for testing the provider guide."""

import os
import logging
from typing import Optional

import httpx

from scout.llm import LLMResponse
from scout.llm.pricing import estimate_cost_usd

logger = logging.getLogger(__name__)


async def call_exampleprovider_async(
    prompt: str,
    model: str = "example-model",
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: Optional[float] = None,
) -> LLMResponse:
    """
    Call the Example Provider API (dummy implementation).

    This is a mock provider that returns a predefined response for testing
    the provider guide. In a real provider, this would call an actual API.

    Args:
        prompt: User prompt
        model: Model identifier (e.g., 'example-model')
        system: Optional system prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        LLMResponse with content, cost, tokens, model
    """
    api_key = os.environ.get("EXAMPLEPROVIDER_API_KEY")
    if not api_key:
        # For testing purposes, allow a dummy key
        api_key = os.environ.get("EXAMPLEPROVIDER_API_KEY", "dummy-key-for-testing")

    url = os.environ.get(
        "EXAMPLEPROVIDER_API_URL", 
        "https://api.exampleprovider.com/v1/chat/completions"
    )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=60.0)) as client:
            response = await client.post(
                url,
                json=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPStatusError as e:
        # For testing: if we get an error (expected since this is a fake endpoint),
        # return a mock response instead
        logger.warning(f"ExampleProvider API error (expected for dummy): {e}")
        return _mock_response(model, prompt)
    except Exception as e:
        # For testing: return a mock response for any connection error
        logger.warning(f"ExampleProvider connection error (expected for dummy): {e}")
        return _mock_response(model, prompt)

    # Extract response data
    content = data["choices"][0]["message"]["content"]

    # Calculate costs using the pricing module
    input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = data.get("usage", {}).get("completion_tokens", 0)

    cost_usd = estimate_cost_usd(
        model_id=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )

    return LLMResponse(
        content=content,
        cost_usd=cost_usd,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _mock_response(model: str, prompt: str) -> LLMResponse:
    """Generate a mock response for testing purposes."""
    return LLMResponse(
        content=f"Mock response from exampleprovider for prompt: {prompt[:50]}...",
        cost_usd=0.0001,
        model=model,
        input_tokens=10,
        output_tokens=20,
    )


# Supported models for this provider
EXAMPLEPROVIDER_MODELS = {
    "example-model": {
        "context": 100_000,
        "output": 4_096,
    },
}
