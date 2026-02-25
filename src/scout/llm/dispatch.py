"""
Unified LLM Dispatcher for Scout.

Routes model calls to the appropriate provider based on model name patterns.
Supports Groq, Google Gemini, MiniMax, and Anthropic models.
"""

from __future__ import annotations

from typing import Optional

from scout.llm import call_groq_async
from scout.llm.google import call_google_async, GEMINI_MODELS
from scout.llm.minimax import call_minimax_async
from scout.llm.anthropic import call_anthropic_async, CLAUDE_MODELS
from scout.llm.router import call_deepseek_async
from scout.llm.exampleprovider import call_exampleprovider_async
from scout.audit import AuditLog

_audit = AuditLog()


PROVIDER_MAP = {
    "deepseek": ("deepseek", call_deepseek_async),
    "llama": ("groq", call_groq_async),
    "mixtral": ("groq", call_groq_async),
    "gemma": ("groq", call_groq_async),
    "gemini": ("google", call_google_async),
    "claude": ("anthropic", call_anthropic_async),
    "minimax": ("minimax", call_minimax_async),
    "abab6": ("minimax", call_minimax_async),
    # Example provider for testing
    "example": ("exampleprovider", call_exampleprovider_async),
}


DEFAULT_PROVIDER = ("groq", call_groq_async)


def get_provider_for_model(model: str) -> tuple[str, callable]:
    """
    Get the provider name and function for a given model.
    
    Args:
        model: Model name (e.g., 'gemini-1.5-flash', 'llama-3.1-8b-instant')
        
    Returns:
        Tuple of (provider_name, provider_function)
        
    Raises:
        ValueError: If no provider is found for the model
    """
    model_lower = model.lower()
    
    for pattern, provider in PROVIDER_MAP.items():
        if pattern in model_lower:
            if provider[1] is None:
                raise ValueError(
                    f"Provider for pattern '{pattern}' not implemented yet"
                )
            return provider
    
    return DEFAULT_PROVIDER


async def call_llm_async(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> tuple[str, float]:
    """
    Unified LLM dispatcher that routes to the appropriate provider.
    
    Args:
        model: Model name (e.g., 'gemini-1.5-flash', 'llama-3.1-8b-instant', 'MiniMax-M2.5')
        prompt: The user prompt
        system: Optional system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        
    Returns:
        Tuple of (response_text, cost_usd)
        
    Raises:
        ValueError: If no provider is found or provider is not implemented
    """
    from scout.llm import LLMResponse as NavResponse
    
    _audit.log("llm_request", model=model, prompt_length=len(prompt), max_tokens=max_tokens)
    
    provider_name, provider_func = get_provider_for_model(model)
    
    if provider_name == "groq":
        response: NavResponse = await provider_func(
            prompt=prompt,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        _audit.log("llm_response", model=model, cost=response.cost_usd, response_length=len(response.content))
        return response.content, response.cost_usd
    
    elif provider_name == "google":
        result = await provider_func(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_text, cost = result
        _audit.log("llm_response", model=model, cost=cost, response_length=len(response_text))
        return result
    
    elif provider_name == "minimax":
        result = await provider_func(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model=model,
        )
        response_text, cost = result
        _audit.log("llm_response", model=model, cost=cost, response_length=len(response_text))
        return result
    
    elif provider_name == "anthropic":
        result = await provider_func(
            prompt=prompt,
            system=system,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_text, cost = result
        _audit.log("llm_response", model=model, cost=cost, response_length=len(response_text))
        return result
    
    elif provider_name == "exampleprovider":
        response: NavResponse = await provider_func(
            prompt=prompt,
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        _audit.log("llm_response", model=model, cost=response.cost_usd, response_length=len(response.content))
        return response.content, response.cost_usd
    
    else:
        raise ValueError(f"Unknown provider: {provider_name}")


async def call_llm_async_with_fallback(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    fallback_models: Optional[list[str]] = None,
) -> tuple[str, float]:
    """
    Unified LLM dispatcher with fallback to alternative models.
    
    If the primary model fails, tries fallback models in order.
    
    Args:
        model: Primary model name
        prompt: The user prompt
        system: Optional system prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        fallback_models: List of fallback model names to try on failure
        
    Returns:
        Tuple of (response_text, cost_usd)
        
    Raises:
        ValueError: If all models fail
    """
    models_to_try = [model] + (fallback_models or [])
    last_error = None
    
    for try_model in models_to_try:
        try:
            return await call_llm_async(
                model=try_model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            last_error = e
            continue
    
    raise last_error or ValueError(f"All models failed: {models_to_try}")


def get_provider_info(model: str) -> dict:
    """
    Get provider information for a given model.
    
    Args:
        model: Model name
        
    Returns:
        Dictionary with provider_name and supported boolean
    """
    try:
        provider_name, provider_func = get_provider_for_model(model)
        return {
            "provider": provider_name,
            "supported": provider_func is not None,
            "model": model,
        }
    except ValueError as e:
        return {
            "provider": None,
            "supported": False,
            "model": model,
            "error": str(e),
        }
