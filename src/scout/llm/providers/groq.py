"""Groq provider for ProviderRegistry.

Provides multi-key support with health tracking.
"""

import logging
from typing import Optional

from scout.llm.providers import ProviderClient, ProviderResult, registry, is_permanent_error

logger = logging.getLogger(__name__)


async def _call_groq(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    api_key: str = None,
    **kwargs,
) -> ProviderResult:
    """
    Wrapper for Groq API that returns ProviderResult.
    
    Args:
        model: Model name (e.g., 'llama-3.1-8b-instant')
        prompt: User prompt
        system: Optional system prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        api_key: API key (ignored - we use the key from the registry)
        
    Returns:
        ProviderResult with response, cost, tokens, model, provider
    """
    # Import inside function to avoid circular import
    from scout.llm import call_groq_async, LLMResponse
    
    # Get provider for key rotation tracking
    provider = registry.get("groq")
    
    # Call the actual Groq function
    try:
        response: LLMResponse = await call_groq_async(
            prompt=prompt,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Success - reset failure count for the key being used
        # Note: We don't track which specific key was used since call_groq_async 
        # handles key internally, but we mark the provider as healthy
        provider.record_key_success(provider.keys[0].key if provider.keys else "")
        
        return ProviderResult(
            response_text=response.content,
            cost_usd=response.cost_usd,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            model=response.model,
            provider="groq",
        )
    except Exception as e:
        # Record failure for the provider
        if provider.keys:
            provider.record_key_failure(provider.keys[0].key, permanent=is_permanent_error(e))
        raise


# Create provider client
_groq_client = ProviderClient(
    name="groq",
    call=_call_groq,
    env_key_name="GROQ_API_KEYS",      # Comma-separated keys: "key1,key2,key3"
    env_single_key_name="GROQ_API_KEY", # Single key fallback
    default_cooldown=60.0,
    max_failures_before_cooldown=3,
    max_key_attempts=3,
)

# Register on import
registry.register("groq", _groq_client)
logger.info("Registered groq provider with %d keys", len(_groq_client.keys))
