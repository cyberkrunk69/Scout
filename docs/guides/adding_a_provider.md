# Adding a New LLM Provider to Scout-Core

This guide walks you through adding a new LLM provider to scout-core. The architecture is designed to be modular, allowing you to add providers with minimal changes to the core system.

## Overview of the Provider Architecture

Scout-core uses a multi-layered architecture for LLM providers:

| Component | Location | Purpose |
|-----------|----------|---------|
| **ProviderRegistry** | `scout/llm/providers/__init__.py` | Central registry that tracks all providers and their keys |
| **ProviderClient** | `scout/llm/providers/__init__.py` | Holds the provider's call method and manages a pool of API keys |
| **KeyState** | `scout/llm/providers/__init__.py` | Tracks health of individual API keys (failures, cooldowns, permanent failures) |
| **Provider wrapper** | `scout/llm/providers/{provider}.py` | Thin wrapper that connects the registry to the actual API implementation |
| **Base API implementation** | `scout/llm/{provider}.py` | Raw API calls to the provider's endpoints |

The flow is:
1. `call_llm()` (in `scout/llm/router.py`) routes requests to a provider
2. The provider wrapper in `scout/llm/providers/{provider}.py` is called
3. The wrapper calls the actual API implementation in `scout/llm/{provider}.py`
4. The response is standardized into a `ProviderResult` and returned

## Step-by-Step: Adding a New Provider

### Step 1: Create the Base API Implementation

Create a new file at `src/scout/llm/{provider}.py`. This handles the raw API communication.

```python
"""ProviderName provider for Scout."""

import os
import logging
from typing import Optional

from scout.llm import LLMResponse
from scout.llm.pricing import estimate_cost_usd

logger = logging.getLogger(__name__)

async def call_provider_async(
    prompt: str,
    model: str = "default-model",
    system: Optional[str] = None,
    max_tokens: int = 500,
    temperature: Optional[float] = None,
) -> LLMResponse:
    """
    Call the Provider API.
    
    Args:
        prompt: User prompt
        model: Model identifier
        system: Optional system prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
    
    Returns:
        LLMResponse with content, cost, tokens, model
    """
    api_key = os.environ.get("PROVIDER_API_KEY")
    if not api_key:
        raise EnvironmentError("PROVIDER_API_KEY not set")
    
    # Build your API request here
    url = os.environ.get("PROVIDER_API_URL", "https://api.provider.com/v1/chat/completions")
    
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    
    # Make the API call using httpx
    import httpx
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
    
    # Extract response data
    content = data["choices"][0]["message"]["content"]
    
    # Calculate costs using the pricing module
    input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = data.get("usage", {}).get("completion_tokens", 0)
    cost_usd = estimate_cost_usd(
        provider="provider",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    
    return LLMResponse(
        content=content,
        cost_usd=cost_usd,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
```

### Step 2: Create the Provider Wrapper

Create `src/scout/llm/providers/{provider}.py`. This connects your implementation to the registry:

```python
"""ProviderName provider for ProviderRegistry.

Provides multi-key support with health tracking.
"""

import logging
from typing import Optional

from scout.llm.providers import ProviderClient, ProviderResult, registry, is_permanent_error

logger = logging.getLogger(__name__)


async def _call_provider(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    api_key: str = None,  # Ignored - we use keys from registry
    **kwargs,
) -> ProviderResult:
    """
    Wrapper for Provider API that returns ProviderResult.
    
    Args:
        model: Model name (e.g., 'model-id')
        prompt: User prompt
        system: Optional system prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature
        api_key: API key (ignored - we use the key from the registry)
        
    Returns:
        ProviderResult with response, cost, tokens, model, provider
    """
    # Import inside function to avoid circular import
    from scout.llm import call_provider_async, LLMResponse
    
    # Get provider for key rotation tracking
    provider = registry.get("providername")
    
    # Call the actual provider function
    try:
        response: LLMResponse = await call_provider_async(
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
            provider="providername",
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
_provider_client = ProviderClient(
    name="providername",
    call=_call_provider,
    env_key_name="PROVIDER_API_KEYS",      # Comma-separated: "key1,key2,key3"
    env_single_key_name="PROVIDER_API_KEY", # Single key fallback
    default_cooldown=60.0,                  # Seconds to wait after failure
    max_failures_before_cooldown=3,         # Failures before cooldown kicks in
    max_key_attempts=3,                      # Max key rotation attempts
)

# Register on import
registry.register("providername", _provider_client)
logger.info("Registered providername provider with %d keys", len(_provider_client.keys))
```

### Step 3: Register the Provider Module

Add your provider to the imports in `src/scout/llm/providers/__init__.py` at the bottom of the file:

```python
# Import provider modules to register them with the registry
# Each module registers itself on import
from scout.llm.providers import groq  # noqa: F401
from scout.llm.providers import google  # noqa: F401
from scout.llm.providers import minimax  # noqa: F401
from scout.llm.providers import anthropic  # noqa: F401
from scout.llm.providers import yourprovider  # noqa: F401  # Add this line
```

### Step 4: Add Pricing Information

If your provider uses new models, add pricing to `src/scout/llm/pricing.py`:

```python
# Add to the PRICING dictionary
PROVIDER_NAME = {
    "model-id-1": {"input": 0.0001, "output": 0.0002},  # per 1K tokens
    "model-id-2": {"input": 0.0002, "output": 0.0004},
}
```

## Multi-Key Support

The registry natively supports multiple API keys with automatic health tracking:

### Environment Variable Configuration

```bash
# Single key
export PROVIDER_API_KEY="sk-..."

# Multiple keys (comma-separated)
export PROVIDER_API_KEYS="sk-key1,sk-key2,sk-key3"
```

The `_parse_keys_from_env()` method in `ProviderClient` handles precedence:
1. `_KEYS` (comma-separated) takes priority
2. Falls back to `_KEY` (single key)

### How Key Rotation Works

1. `ProviderClient.get_working_key()` returns a healthy key with the lowest failure count
2. On failure: `record_key_failure()` applies exponential backoff cooldown
3. On success: `record_key_success()` resets the failure count
4. Keys marked permanently failed (e.g., invalid API key) are excluded from rotation

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_cooldown` | 60.0 | Base seconds to wait after a failure |
| `max_failures_before_cooldown` | 3 | Failures before cooldown is applied |
| `max_key_attempts` | 3 | Max keys to try before giving up |
| `MAX_COOLDOWN_SECONDS` | 3600 | Maximum cooldown cap (1 hour) |

## Cost Tracking and Audit

### Automatic Cost Tracking

Costs are calculated in the base API implementation using `estimate_cost_usd()`:

```python
from scout.llm.pricing import estimate_cost_usd

cost_usd = estimate_cost_usd(
    provider="providername",
    model=model,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
)
```

### Audit Logging

To log LLM calls to the audit system:

```python
from scout.audit import get_audit

audit = get_audit()
audit.log(
    "nav",  # Event type
    cost=0.0005,
    model="model-id",
    input_t=100,
    output_t=50,
    reason="navigation",
)
```

Available event types include: `nav`, `brief`, `cascade`, `validation_fail`, `budget`, `llm_retry`, `llm_error`, etc.

### Circuit Breaker Integration

The system includes circuit breaker protection. Record successes and failures:

```python
from scout.llm.providers import record_success, record_failure, is_provider_available

# Check if provider is available
if not is_provider_available("providername"):
    raise CircuitOpenError("Provider circuit is open")

# After API call
record_success("providername")
# or
record_failure("providername", permanent=is_permanent_error(e))
```

## Testing Your Provider

### Unit Tests

Create tests in `tests/scout/llm/test_provider.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch

from scout.llm.providers import registry, ProviderClient


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("TESTPROVIDER_API_KEY", "test-key-123")


async def test_provider_registers(mock_env):
    """Test that provider registers correctly."""
    from scout.llm.providers import testprovider  # noqa: F401
    
    assert registry.available("testprovider")
    client = registry.get("testprovider")
    assert len(client.keys) == 1
    assert client.keys[0].key == "test-key-123"


async def test_provider_call():
    """Test provider call returns ProviderResult."""
    from scout.llm.providers import testprovider
    
    with patch("scout.llm.testprovider.call_testprovider_async") as mock_call:
        from scout.llm import LLMResponse
        mock_call.return_value = LLMResponse(
            content="Test response",
            cost_usd=0.0001,
            model="test-model",
            input_tokens=10,
            output_tokens=20,
        )
        
        client = registry.get("testprovider")
        result = await client.call(
            model="test-model",
            prompt="Hello",
        )
        
        assert result.response_text == "Test response"
        assert result.provider == "testprovider"
```

### Integration Tests

Test actual API calls with mocked responses:

```python
@pytest.mark.integration
async def test_provider_integration():
    """Test actual API call (requires API key)."""
    # Skip if no API key
    import os
    if not os.environ.get("TESTPROVIDER_API_KEY"):
        pytest.skip("No API key")
    
    from scout.llm.providers import testprovider
    client = registry.get("testprovider")
    
    result = await client.call(
        model="test-model",
        prompt="Say 'test'",
    )
    
    assert result.response_text
    assert result.cost_usd >= 0
```

## Configuration Reference

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{PROVIDER}_API_KEY` | Single API key | `GROQ_API_KEY=sk-...` |
| `{PROVIDER}_API_KEYS` | Comma-separated keys | `GROQ_API_KEYS=sk-1,sk-2` |
| `{PROVIDER}_API_URL` | Custom endpoint | `GROQ_API_URL=https://...` |

### ProviderClient Configuration

When creating a `ProviderClient`, these parameters are available:

```python
client = ProviderClient(
    name="providername",
    call=_call_provider,
    env_key_name="PROVIDER_API_KEYS",
    env_single_key_name="PROVIDER_API_KEY",
    default_cooldown=60.0,
    max_failures_before_cooldown=3,
    max_key_attempts=3,
)
```

## Common Patterns

### Handling Rate Limits

```python
async def _call_provider(...) -> ProviderResult:
    try:
        response = await call_provider_async(...)
        return response
    except Exception as e:
        error_str = str(e).lower()
        if "rate limit" in error_str or "too many requests" in error_str:
            # Record non-permanent failure (will retry with backoff)
            provider.record_key_failure(key, permanent=False)
            raise  # Let caller retry
        else:
            # Permanent error
            provider.record_key_failure(key, permanent=True)
            raise
```

### Custom Error Detection

The `is_permanent_error()` function detects permanent failures. You can extend patterns in `providers/__init__.py`:

```python
PERMANENT_ERROR_PATTERNS.update({
    "your_category": [
        "your error pattern",
    ],
})
```

## Troubleshooting

### Provider Not Registering

1. Check that the import exists in `providers/__init__.py`
2. Verify the module is being imported
3. Check logs for registration messages

### Keys Not Loading

1. Verify environment variables are set
2. Check naming: `PROVIDER_API_KEYS` vs `PROVIDER_API_KEY`
3. Ensure keys don't have extra whitespace (use `.strip()`)

### Key Rotation Not Working

1. Verify `ProviderClient` is being used, not direct API calls
2. Check that `record_key_failure()` and `record_key_success()` are called
3. Review logs for cooldown timing

---

For more details, see:
- `src/scout/llm/providers/__init__.py` - Full provider infrastructure
- `src/scout/llm/providers/groq.py` - Simple provider example
- `src/scout/llm/providers/anthropic.py` - Complex provider with more features
- `src/scout/llm/pricing.py` - Cost calculation
- `src/scout/audit.py` - Audit logging
