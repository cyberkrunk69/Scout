"""Anthropic provider for ProviderRegistry.

Provides multi-key support with health tracking.
"""

import os
import logging
from typing import Any, Optional

from scout.llm.providers import ProviderClient, ProviderResult, registry, is_permanent_error

logger = logging.getLogger(__name__)


async def _call_anthropic(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    api_key: str = None,
    **kwargs,
) -> ProviderResult:
    """
    Wrapper for Anthropic API that returns ProviderResult.
    
    Args:
        model: Model name (e.g., 'claude-3-5-sonnet-20241022')
        prompt: User prompt
        system: Optional system prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        api_key: API key (ignored - we use the key from the registry)
        
    Returns:
        ProviderResult with response, cost, tokens, model, provider
    """
    from scout.llm.anthropic import call_anthropic_async
    
    provider = registry.get("anthropic")
    
    try:
        response_text, cost_usd = await call_anthropic_async(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        if provider.keys:
            provider.record_key_success(provider.keys[0].key)
        
        input_tokens = len(prompt) // 4
        output_tokens = len(response_text) // 4
        
        return ProviderResult(
            response_text=response_text,
            cost_usd=cost_usd,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
            provider="anthropic",
        )
    except Exception as e:
        if provider.keys:
            provider.record_key_failure(provider.keys[0].key, permanent=is_permanent_error(e))
        raise


_anthropic_client = ProviderClient(
    name="anthropic",
    call=_call_anthropic,
    env_key_name="ANTHROPIC_API_KEYS",
    env_single_key_name="ANTHROPIC_API_KEY",
    default_cooldown=60.0,
    max_failures_before_cooldown=3,
    max_key_attempts=3,
)

registry.register("anthropic", _anthropic_client)
logger.info("Registered anthropic provider with %d keys", len(_anthropic_client.keys))
