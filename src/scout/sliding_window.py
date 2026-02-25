"""Sliding window rate limiter for token-based limiting.

Implements a sliding window algorithm for tracking both request
and token rates.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RateLimitConfig:
    """Configuration for rate limits."""
    
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000  # 100k TPM default
    
    @classmethod
    def for_deepseek(cls) -> "RateLimitConfig":
        """DeepSeek free tier limits."""
        return cls(requests_per_minute=50, tokens_per_minute=2_000_000)
    
    @classmethod
    def for_minimax(cls) -> "MiniMaxRateLimitConfig":
        """MiniMax limits (varies by tier)."""
        return MiniMaxRateLimitConfig()


@dataclass
class MiniMaxRateLimitConfig(RateLimitConfig):
    """MiniMax-specific rate limits."""
    
    requests_per_minute: int = 100
    tokens_per_minute: int = 200_000


@dataclass
class TokenUsage:
    """Token usage record."""
    
    timestamp: float
    tokens: int


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for both RPM and TPM."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        
        # Request tracking
        self._request_times: deque[float] = deque()
        
        # Token tracking
        self._token_usage: deque[TokenUsage] = deque()
        
        self._lock = asyncio.Lock()
    
    async def acquire(
        self,
        tokens: int = 0,
        wait: bool = True,
    ) -> bool:
        """
        Acquire permission to make a request.
        
        Args:
            tokens: Number of tokens for this request
            wait: Whether to wait if rate limited
        
        Returns:
            True if acquired, False if rate limited (and wait=False)
        
        Raises:
            RateLimitError: If rate limited (and wait=True)
        """
        async with self._lock:
            now = time.time()
            window_start = now - 60
            
            # Clean old entries
            while self._request_times and self._request_times[0] < window_start:
                self._request_times.popleft()
            
            while self._token_usage and self._token_usage[0].timestamp < window_start:
                self._token_usage.popleft()
            
            # Check request limit
            current_rpm = len(self._request_times)
            
            if current_rpm >= self.config.requests_per_minute:
                if not wait:
                    return False
                
                # Calculate wait time
                oldest = self._request_times[0]
                wait_time = oldest + 60 - now
                
                # Release lock and wait
                self._lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._lock.acquire()
                
                # Retry after wait
                return await self.acquire(tokens, wait)
            
            # Check token limit
            if tokens > 0:
                current_tokens = sum(u.tokens for u in self._token_usage)
                
                if current_tokens + tokens > self.config.tokens_per_minute:
                    if not wait:
                        return False
                    
                    # Calculate wait time based on oldest token usage
                    if self._token_usage:
                        oldest = self._token_usage[0].timestamp
                        wait_time = oldest + 60 - now
                        
                        self._lock.release()
                        try:
                            await asyncio.sleep(wait_time)
                        finally:
                            await self._lock.acquire()
                        
                        return await self.acquire(tokens, wait)
            
            # Record this request
            self._request_times.append(now)
            if tokens > 0:
                self._token_usage.append(TokenUsage(timestamp=now, tokens=tokens))
            
            return True
    
    def get_status(self) -> dict:
        """Get current rate limit status."""
        now = time.time()
        window_start = now - 60
        
        # Count current usage
        current_requests = sum(1 for t in self._request_times if t >= window_start)
        current_tokens = sum(
            u.tokens for u in self._token_usage if u.timestamp >= window_start
        )
        
        return {
            "requests_this_minute": current_requests,
            "requests_limit": self.config.requests_per_minute,
            "requests_remaining": self.config.requests_per_minute - current_requests,
            "tokens_this_minute": current_tokens,
            "tokens_limit": self.config.tokens_per_minute,
            "tokens_remaining": self.config.tokens_per_minute - current_tokens,
        }
    
    def reset(self) -> None:
        """Reset the rate limiter."""
        self._request_times.clear()
        self._token_usage.clear()


class MultiProviderRateLimiter:
    """Rate limiter that manages multiple providers."""
    
    def __init__(self):
        self._limiters: dict[str, SlidingWindowRateLimiter] = {}
        self._default_config = RateLimitConfig.for_deepseek()
    
    def get_limiter(self, provider: str) -> SlidingWindowRateLimiter:
        """Get or create limiter for a provider."""
        if provider not in self._limiters:
            if provider == "deepseek":
                config = RateLimitConfig.for_deepseek()
            elif provider == "minimax":
                config = RateLimitConfig.for_minimax()
            else:
                config = self._default_config
            
            self._limiters[provider] = SlidingWindowRateLimiter(config)
        
        return self._limiters[provider]
    
    def get_all_status(self) -> dict:
        """Get status of all limiters."""
        return {
            provider: limiter.get_status()
            for provider, limiter in self._limiters.items()
        }
    
    def reset_all(self) -> None:
        """Reset all limiters."""
        for limiter in self._limiters.values():
            limiter.reset()


# Global instance
_rate_limiter: Optional[MultiProviderRateLimiter] = None


def get_rate_limiter() -> MultiProviderRateLimiter:
    """Get global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = MultiProviderRateLimiter()
    return _rate_limiter


class RateLimitError(Exception):
    """Rate limit exceeded."""
    pass
