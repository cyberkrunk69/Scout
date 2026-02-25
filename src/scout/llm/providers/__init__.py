"""LLM Provider Registry - Multi-key support with health tracking.

Provides:
- ProviderRegistry: Central registry for LLM providers
- ProviderClient: Holds provider call method and key pool
- KeyState: Tracks individual API key health (failures, cooldowns, permanent failure)
- is_permanent_error(): Detect errors that should stop key rotation
"""

from dataclasses import dataclass, field
from typing import Callable, Awaitable, Optional
import time
import logging

# Load .env if available (for direct imports outside CLI)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on system env

logger = logging.getLogger(__name__)

# =============================================================================
# Provider Result
# =============================================================================

@dataclass
class ProviderResult:
    """Standardized result from any provider."""
    response_text: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    model: str
    provider: str


# =============================================================================
# Key State - Individual API Key Health Tracking
# =============================================================================

@dataclass
class KeyState:
    """Tracks health of a single API key."""
    key: str
    failures: int = 0
    last_failure: float = 0.0
    cooldown_until: float = 0.0
    permanently_failed: bool = False
    
    # Class-level cap to prevent unbounded cooldowns
    MAX_COOLDOWN_SECONDS: float = 3600.0  # 1 hour max
    
    def is_healthy(self) -> bool:
        """Check if key is healthy (not in cooldown, not permanently failed)."""
        if self.permanently_failed:
            return False
        return time.time() > self.cooldown_until
    
    def record_failure(self, permanent: bool = False, cooldown: float = 60.0):
        """Record a failure and apply exponential backoff cooldown."""
        self.failures += 1
        self.last_failure = time.time()
        if permanent:
            self.permanently_failed = True
            logger.warning(f"Key marked permanently failed")
        else:
            # Exponential backoff with cap
            actual_cooldown = min(
                cooldown * (2 ** min(self.failures, 10)),  # Cap exponent at 10
                self.MAX_COOLDOWN_SECONDS
            )
            self.cooldown_until = time.time() + actual_cooldown
            logger.debug(f"Key cooldown until {self.cooldown_until} ({actual_cooldown}s)")
    
    def record_success(self):
        """Reset failure count on success."""
        self.failures = 0
        self.permanently_failed = False


# =============================================================================
# Circuit Breaker - import from shared module
# =============================================================================

# Import circuit breaker from shared module
from scout.llm.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    get_breaker,
    is_provider_available,
    record_success,
    record_failure,
    CircuitOpenError,
)

# Backward compatibility aliases
def get_circuit_breaker(provider_name: str) -> CircuitBreaker:
    """Get or create circuit breaker for a provider. (Backward compatibility)"""
    return get_breaker(provider_name)


# =============================================================================
# Provider Client - Holds call method and key pool
# =============================================================================

@dataclass
class ProviderClient:
    """Holds provider's call method and metadata with multi-key support."""
    name: str
    call: Callable[..., Awaitable[ProviderResult]]
    cleanup: Optional[Callable] = None
    keys: list[KeyState] = field(default_factory=list)
    
    # Provider-level config
    env_key_name: str = ""  # e.g., "GROQ_API_KEYS"
    env_single_key_name: str = ""  # e.g., "GROQ_API_KEY" (fallback)
    default_cooldown: float = 60.0  # seconds
    max_failures_before_cooldown: int = 3
    max_key_attempts: int = 3  # Cap key rotation attempts to avoid infinite loops
    
    def add_key(self, key: str):
        """Add a key to the pool."""
        self.keys.append(KeyState(key=key))
        logger.info(f"Added key to {self.name} pool (total: {len(self.keys)})")
    
    def _parse_keys_from_env(self):
        """Parse keys from environment variables.
        
        Precedence:
        1. ENV_KEY_NAME (_KEYS) - comma-separated list takes priority
        2. ENV_SINGLE_KEY_NAME (_KEY) - single key as fallback
        """
        import os
        
        # Priority: _KEYS takes precedence over _KEY
        if self.env_key_name:
            keys_str = os.environ.get(self.env_key_name, "")
            if keys_str:
                for key in keys_str.split(","):
                    key = key.strip()
                    if key:
                        self.add_key(key)
                return  # _KEYS takes priority
        
        # Fallback to single key
        if self.env_single_key_name:
            key = os.environ.get(self.env_single_key_name, "")
            if key:
                self.add_key(key)
    
    def get_working_key(self) -> Optional[str]:
        """Get the best available key (not in cooldown, lowest failures)."""
        if not self.keys:
            return None
        
        # Filter healthy keys
        healthy = [k for k in self.keys if k.is_healthy()]
        if not healthy:
            return None
        
        # Return key with lowest failure count
        return min(healthy, key=lambda k: k.failures).key
    
    def has_keys(self) -> bool:
        """Check if provider has any keys configured."""
        return len([k for k in self.keys if not k.permanently_failed]) > 0
    
    def record_key_failure(self, key: str, permanent: bool = False):
        """Record failure for a specific key."""
        for k in self.keys:
            if k.key == key:
                k.record_failure(
                    permanent=permanent,
                    cooldown=self.default_cooldown
                )
                break
    
    def record_key_success(self, key: str):
        """Record success for a specific key."""
        for k in self.keys:
            if k.key == key:
                k.record_success()
                break


# =============================================================================
# Provider Registry
# =============================================================================

class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, ProviderClient] = {}
    
    def register(self, name: str, client: ProviderClient):
        self._providers[name] = client
        # Auto-parse keys from env
        client._parse_keys_from_env()
        logger.info(f"Registered provider: {name} with {len(client.keys)} keys")
    
    def get(self, name: str) -> ProviderClient:
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        return self._providers[name]
    
    def available(self, name: str) -> bool:
        if name not in self._providers:
            return False
        return self._providers[name].has_keys()
    
    def list_providers(self) -> list[str]:
        return list(self._providers.keys())


# Global registry instance
registry = ProviderRegistry()


# =============================================================================
# Permanent Error Detection
# =============================================================================

# Permanent error signatures - string patterns
PERMANENT_ERROR_PATTERNS = {
    "geo_blocked": [
        "user location not supported",
        "not available in your country",
        "geo_restricted",
    ],
    "invalid_key": [
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "api key not found",
    ],
    "quota_exceeded": [
        # Generic patterns
        "quota exceeded",
        "insufficient credits",
        "billing required",
        "credit limit",
        "no credits",
        "credits exhausted",
        "exceeded your credits",
        # OpenRouter specific
        "credits exceeded",
        "no credits left",
        "insufficient quota",
        "out of credits",
        # MiniMax specific
        "balance insufficient",
        "account balance",
        # Groq specific (rate limit is separate, but quota errors can occur)
        "quota",
        "usage limit",
    ],
    # Rate limit - should retry with backoff, not permanent fallback
    "rate_limit": [
        "rate limit exceeded",
        "too many requests",
        "rate_limit",
    ],
}

# Exception types that indicate permanent failures (checked by type)
PERMANENT_ERROR_TYPES: tuple = ()

def _load_permanent_error_types() -> tuple:
    """Lazily load exception types for permanent error detection."""
    types = []
    try:
        from openai import AuthenticationError, PermissionDeniedError
        types.extend([AuthenticationError, PermissionDeniedError])
    except ImportError:
        pass
    try:
        from anthropic import AuthenticationError, PermissionError
        types.extend([AuthenticationError, PermissionError])
    except ImportError:
        pass
    try:
        from google.api_core.exceptions import Forbidden, NotFound
        types.extend([Forbidden, NotFound])
    except ImportError:
        pass
    return tuple(types)


def is_permanent_error(error: Exception) -> bool:
    """Detect if error is permanent (don't retry with another key).
    
    Checks:
    1. Exception type (e.g., AuthenticationError)
    2. Error message patterns
    """
    global PERMANENT_ERROR_TYPES
    
    # Check exception type first (more reliable)
    if not PERMANENT_ERROR_TYPES:
        PERMANENT_ERROR_TYPES = _load_permanent_error_types()
    
    if isinstance(error, PERMANENT_ERROR_TYPES):
        logger.warning(f"Permanent error by type: {type(error).__name__}")
        return True
    
    # Check error string patterns
    error_str = str(error).lower()
    for category, patterns in PERMANENT_ERROR_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in error_str:
                logger.warning(f"Permanent error detected: {category} - {error}")
                return True
    
    return False


# Import provider modules to register them with the registry
# Each module registers itself on import
from scout.llm.providers import groq  # noqa: F401
from scout.llm.providers import google  # noqa: F401
from scout.llm.providers import minimax  # noqa: F401
from scout.llm.providers import anthropic  # noqa: F401
