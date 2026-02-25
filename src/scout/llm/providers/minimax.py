"""MiniMax provider for ProviderRegistry.

Provides multi-key support with health tracking.
"""

import os
import logging
from typing import Any, Optional

from scout.llm.providers import ProviderClient, ProviderResult, registry, is_permanent_error

logger = logging.getLogger(__name__)


async def _call_minimax(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    api_key: str = None,
    **kwargs,
) -> ProviderResult:
    """
    Wrapper for MiniMax API that returns ProviderResult.
    
    Args:
        model: Model name (e.g., 'MiniMax-M2.5')
        prompt: User prompt
        system: Optional system prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        api_key: API key (ignored - we use the key from the registry)
        
    Returns:
        ProviderResult with response, cost, tokens, model, provider
    """
    from scout.llm.minimax import call_minimax_async_detailed
    
    provider = registry.get("minimax")
    
    try:
        result = await call_minimax_async_detailed(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )
        
        if provider.keys:
            provider.record_key_success(provider.keys[0].key)
        
        return ProviderResult(
            response_text=result.response_text,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            model=result.model,
            provider="minimax",
        )
    except Exception as e:
        if provider.keys:
            provider.record_key_failure(provider.keys[0].key, permanent=is_permanent_error(e))
        raise


_minimax_client = ProviderClient(
    name="minimax",
    call=_call_minimax,
    env_key_name="MINIMAX_API_KEYS",
    env_single_key_name="MINIMAX_API_KEY",
    default_cooldown=60.0,
    max_failures_before_cooldown=3,
    max_key_attempts=3,
)

registry.register("minimax", _minimax_client)
logger.info("Registered minimax provider with %d keys", len(_minimax_client.keys))
