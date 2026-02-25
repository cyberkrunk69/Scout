"""Google Gemini provider for ProviderRegistry.

Provides multi-key support with health tracking.
"""

import os
import logging
from typing import Any, Optional

from scout.llm.providers import ProviderClient, ProviderResult, registry, is_permanent_error

logger = logging.getLogger(__name__)


async def _call_google(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    api_key: str = None,
    **kwargs,
) -> ProviderResult:
    """
    Wrapper for Google Gemini API that returns ProviderResult.
    
    Args:
        model: Model name (e.g., 'gemini-2.0-flash')
        prompt: User prompt
        system: Optional system prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        api_key: API key (ignored - we use the key from the registry)
        
    Returns:
        ProviderResult with response, cost, tokens, model, provider
    """
    from scout.llm.google import call_google_async
    
    provider = registry.get("google")
    
    try:
        response_text, cost_usd = await call_google_async(
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
            provider="google",
        )
    except Exception as e:
        if provider.keys:
            provider.record_key_failure(provider.keys[0].key, permanent=is_permanent_error(e))
        raise


_google_client = ProviderClient(
    name="google",
    call=_call_google,
    env_key_name="GOOGLE_API_KEYS",
    env_single_key_name="GEMINI_API_KEY",
    default_cooldown=60.0,
    max_failures_before_cooldown=3,
    max_key_attempts=3,
)

registry.register("google", _google_client)
logger.info("Registered google provider with %d keys", len(_google_client.keys))
