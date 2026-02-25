"""Example provider for ProviderRegistry.

Provides multi-key support with health tracking.
This is a dummy provider for testing the provider guide.
"""

import logging
from typing import Optional

from scout.llm.providers import ProviderClient, ProviderResult, registry, is_permanent_error

logger = logging.getLogger(__name__)


async def _call_exampleprovider(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    api_key: str = None,
    **kwargs,
) -> ProviderResult:
    """
    Wrapper for Example Provider API that returns ProviderResult.
    
    Args:
        model: Model name (e.g., 'example-model')
        prompt: User prompt
        system: Optional system prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        api_key: API key (ignored - we use the key from the registry)
        
    Returns:
        ProviderResult with response, cost, tokens, model, provider
    """
    # Import inside function to avoid circular import
    from scout.llm.exampleprovider import call_exampleprovider_async
    from scout.llm import LLMResponse
    
    # Get provider for key rotation tracking
    provider = registry.get("exampleprovider")
    
    # Call the actual Example Provider function
    try:
        response: LLMResponse = await call_exampleprovider_async(
            prompt=prompt,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Success - reset failure count for the key being used
        if provider.keys:
            provider.record_key_success(provider.keys[0].key)
        
        return ProviderResult(
            response_text=response.content,
            cost_usd=response.cost_usd,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model,
            provider="exampleprovider",
        )
    except Exception as e:
        # Record failure for the provider
        if provider.keys:
            provider.record_key_failure(
                provider.keys[0].key,
                permanent=is_permanent_error(e)
            )
        raise


# Create provider client
_exampleprovider_client = ProviderClient(
    name="exampleprovider",
    call=_call_exampleprovider,
    env_key_name="EXAMPLEPROVIDER_API_KEYS",      # Comma-separated keys: "key1,key2,key3"
    env_single_key_name="EXAMPLEPROVIDER_API_KEY", # Single key fallback
    default_cooldown=60.0,
    max_failures_before_cooldown=3,
    max_key_attempts=3,
)

# Register on import
registry.register("exampleprovider", _exampleprovider_client)
logger.info("Registered exampleprovider provider with %d keys", len(_exampleprovider_client.keys))
