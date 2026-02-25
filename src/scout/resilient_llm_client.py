"""LLM client wrapper with timeout, retry, and circuit breaker support.

Provides a unified client that wraps the multi-provider router with
additional resilience features.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from scout.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    get_circuit_breaker_manager,
)
from scout.retry import RetryConfig, retry_sync as retry
from scout.sliding_window import (
    MultiProviderRateLimiter,
    get_rate_limiter,
)
from scout.timeout_config import TimeoutConfig, get_timeout_config

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM client."""
    content: str
    cost_usd: float
    tokens: int
    provider: str
    model: str


class ResilientLLMClient:
    """LLM client with built-in resilience features."""
    
    def __init__(
        self,
        retry_config: Optional[RetryConfig] = None,
        timeout_config: Optional[TimeoutConfig] = None,
        circuit_config: Optional[CircuitBreakerConfig] = None,
    ):
        self.retry_config = retry_config or RetryConfig()
        self.timeout_config = timeout_config or get_timeout_config()
        
        # Get shared circuit breaker and rate limiter
        self._circuit_manager = get_circuit_breaker_manager()
        self._rate_limiter = get_rate_limiter()
    
    async def call(
        self,
        prompt: str,
        system: Optional[str] = None,
        provider: str = "auto",
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """
        Make an LLM call with full resilience features.
        
        Args:
            prompt: User prompt
            system: System message
            provider: Provider to use ("auto", "deepseek", "minimax")
            model: Model name (provider-specific)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
        
        Returns:
            LLMResponse with content and metadata
        """
        # Get circuit breaker for provider
        if provider == "auto":
            circuit_name = "llm_auto"
        else:
            circuit_name = f"llm_{provider}"
        
        circuit = self._circuit_manager.get_breaker(circuit_name)
        
        # Get rate limiter
        limiter = self._rate_limiter.get_limiter(provider if provider != "auto" else "deepseek")
        
        # Estimate tokens for rate limiting
        from scout.token_estimator import estimate_tokens_for_prompt
        estimated_tokens = estimate_tokens_for_prompt(prompt, system)
        
        # Acquire rate limit permission
        await limiter.acquire(tokens=estimated_tokens)
        
        # Get timeout for provider
        connect_timeout, read_timeout = self.timeout_config.for_provider(
            provider if provider != "auto" else "deepseek"
        )
        
        # Execute through circuit breaker with retry
        result = await circuit.call(
            self._call_with_timeout,
            prompt=prompt,
            system=system,
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )
        
        return result
    
    async def _call_with_timeout(
        self,
        prompt: str,
        system: Optional[str],
        provider: str,
        model: Optional[str],
        max_tokens: int,
        temperature: float,
        connect_timeout: float,
        read_timeout: float,
    ) -> LLMResponse:
        """Execute the actual LLM call with timeout."""
        try:
            from scout.router import call_llm
        except ImportError:
            call_llm = None
        
        # Define a fallback if router is not available
        async def fallback_call():
            raise NotImplementedError("LLM router not available - please extract scout.router module")
        
        _call_llm = call_llm or fallback_call
        
        # Wrap in timeout
        try:
            async with asyncio.timeout(read_timeout):
                try:
                    # First try with retry
                    result_text, cost = await self._call_with_retry(
                        prompt=prompt,
                        system=system,
                        provider=provider,
                        model=model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                except RouterRateLimitError:
                    # Fall back to other provider if in auto mode
                    if provider == "auto":
                        fallback_provider = "minimax" if provider != "minimax" else "deepseek"
                        result_text, cost = await self._call_with_retry(
                            prompt=prompt,
                            system=system,
                            provider=fallback_provider,
                            model=model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                    else:
                        raise
                
                # Estimate token count
                from scout.token_estimator import estimate_tokens
                tokens = estimate_tokens(result_text)
                
                return LLMResponse(
                    content=result_text,
                    cost_usd=cost,
                    tokens=tokens,
                    provider=provider if provider != "auto" else "deepseek",
                    model=model or "default",
                )
                
        except asyncio.TimeoutError:
            from scout.retry import RetryStats
            class ProviderError(Exception):
                """Provider error for timeouts."""
                pass
            raise ProviderError(f"Request timeout after {read_timeout}s")
    
    async def _call_with_retry(
        self,
        prompt: str,
        system: Optional[str],
        provider: str,
        model: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> tuple[str, float]:
        """Execute LLM call with retry logic."""
        try:
            from scout.router import call_llm
        except ImportError:
            call_llm = None
        
        try:
            from scout.retry import retry_sync as retry
        except ImportError:
            retry = None
        
        # Fallback if modules not available
        if call_llm is None or retry is None:
            async def fallback_call():
                raise NotImplementedError("Required modules not available")
            return await fallback_call(), 0.0
        
        @retry(
            max_attempts=self.retry_config.max_attempts,
            base_delay=self.retry_config.base_delay,
            max_delay=self.retry_config.max_delay,
            exponential_base=self.retry_config.exponential_base,
        )
        async def _call():
            return await call_llm(
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                model=model,
            )
        
        return await _call()
    
    def get_status(self) -> dict:
        """Get client status."""
        return {
            "circuit_breakers": self._circuit_manager.get_all_status(),
            "rate_limiters": self._rate_limiter.get_all_status(),
        }


# Global client instance
_client: Optional[ResilientLLMClient] = None


def get_resilient_client() -> ResilientLLMClient:
    """Get global resilient LLM client."""
    global _client
    if _client is None:
        _client = ResilientLLMClient()
    return _client
