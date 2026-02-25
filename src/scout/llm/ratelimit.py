"""Rate Limiter for OpenRouter Free Tier.

Provides:
- OpenRouterRateLimiter: Tracks RPM/RPD limits with async-safe operations
"""

import time
import asyncio
from typing import Optional

class OpenRouterRateLimiter:
    """Rate limiter for OpenRouter free tier: 20 RPM, 50 RPD (default).
    
    Uses asyncio.Lock for async-safe operations.
    """
    
    def __init__(self, rpm_limit: int = 20, rpd_limit: int = 50):
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        
        self._rpm_count = 0
        self._rpm_reset = 0.0
        self._rpd_count = 0
        self._rpd_reset = time.time()
        self._lock = asyncio.Lock()
    
    async def check_limits(self) -> tuple[bool, str]:
        """Check if under limits. Returns (allowed, reason)."""
        async with self._lock:
            return self._check_limits_sync()
    
    def _check_limits_sync(self) -> tuple[bool, str]:
        """Synchronous check (called within lock)."""
        now = time.time()
        
        # Reset RPM counter
        if now - self.rpm_reset > 60:
            self._rpm_count = 0
            self.rpm_reset = now
        
        # Reset RPD counter
        if now - self._rpd_reset > 86400:  # 24 hours
            self._rpd_count = 0
            self._rpd_reset = now
        
        # Check RPM
        if self._rpm_count >= self.rpm_limit:
            wait_time = int(60 - (now - self.rpm_reset))
            return False, f"RPM limit ({self.rpm_limit}) reached, wait {wait_time}s"
        
        # Check RPD
        if self._rpd_count >= self.rpd_limit:
            wait_time = int(86400 - (now - self._rpd_reset))
            return False, f"RPD limit ({self.rpd_limit}) reached, wait {wait_time}s"
        
        return True, "ok"
    
    async def increment(self):
        """Increment usage counters after a successful call."""
        async with self._lock:
            self._rpm_count += 1
            self._rpd_count += 1
    
    @property
    async def usage(self) -> dict:
        async with self._lock:
            return {
                "rpm_used": self._rpm_count,
                "rpm_limit": self.rpm_limit,
                "rpd_used": self._rpd_count,
                "rpd_limit": self.rpd_limit,
            }

# Global instance
rate_limiter = OpenRouterRateLimiter()
