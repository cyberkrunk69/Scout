# Adding a New LLM Provider to Scout-Core

This guide walks you through adding a new LLM provider to scout-core. The architecture is designed to be modular, allowing you to add providers with minimal changes to the core system.

## Prerequisites

Before adding a new provider, you should be familiar with:

- **Python async/await**: The codebase uses `asyncio` for concurrent operations
- **httpx**: HTTP client library used for API calls
- **Environment variables**: Provider configuration via env vars

You will need:

- API credentials for the provider
- Pricing information (input/output costs per million tokens)
- Understanding of which model IDs the provider supports

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
1. `call_llm_async()` (in `scout/llm/dispatch.py`) routes requests based on model name
2. The dispatcher looks up the provider in `PROVIDER_MAP`
3. The provider wrapper in `scout/llm/providers/{provider}.py` is called
4. The wrapper calls the actual API implementation in `scout/llm/{provider}.py`
5. The response is standardized into a `ProviderResult` and returned

## Step-by-Step: Adding a New Provider

### Step 1: Create the Base API Implementation

Create a new file at `src/scout/llm/{provider}.py`. This handles the raw API communication. Here's a complete example based on `groq.py`:

```python
"""Groq provider for Scout."""

import os
import logging
from typing import Optional

import httpx

from scout.llm import LLMResponse
from scout.llm.pricing import estimate_cost_usd
from scout.llm.timeout_config import get_timeout_config

logger = logging.getLogger(__name__)


async def call_groq_async(
    prompt: str,
    model: str = "llama-3.1-8b-instant",
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: Optional[float] = None,
) -> LLMResponse:
    """
    Call the Groq API.

    Args:
        prompt: User prompt
        model: Model identifier (e.g., 'llama-3.1-8b-instant')
        system: Optional system prompt
        max_tokens: Max tokens to generate
        temperature: Sampling temperature

    Returns:
        LLMResponse with content, cost, tokens, model
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set")

    url = os.environ.get("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")

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

    # Use configurable timeouts
    timeout_config = get_timeout_config()
    connect_timeout, read_timeout = timeout_config.for_provider("groq")

    async with httpx.AsyncClient(timeout=httpx.Timeout(connect_timeout, read=read_timeout)) as client:
        response = await client.post(
            url,
            json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        data = response.json()

    # Extract response data
    content = data["choices"][0]["message"]["content"]

    # Calculate costs using the pricing module
    input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = data.get("usage", {}).get("completion_tokens", 0)

    # Correct signature: estimate_cost_usd(model_id, input_tokens, output_tokens)
    cost_usd = estimate_cost_usd(
        model_id=model,
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

**Important**: The `estimate_cost_usd` function signature is:

```python
def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
```

Note: It takes `model_id` (not separate `provider` and `model` parameters).

### Step 2: Create the Provider Wrapper

Create `src/scout/llm/providers/{provider}.py`. This connects your implementation to the registry. Here's a complete example based on `groq.py`:

```python
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
        if provider.keys:
            provider.record_key_success(provider.keys[0].key)

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
            provider.record_key_failure(
                provider.keys[0].key,
                permanent=is_permanent_error(e)
            )
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

### Step 4: Add Model-to-Provider Routing

Add your provider to the `PROVIDER_MAP` in `src/scout/llm/dispatch.py`. This maps model name prefixes to providers:

```python
PROVIDER_MAP = {
    "deepseek": ("deepseek", call_deepseek_async),
    "llama": ("groq", call_groq_async),
    "mixtral": ("groq", call_groq_async),
    "gemma": ("groq", call_groq_async),
    "gemini": ("google", call_google_async),
    "claude": ("anthropic", call_anthropic_async),
    "minimax": ("minimax", call_minimax_async),
    "abab6": ("minimax", call_minimax_async),
    # Add your provider here:
    "yourmodelprefix": ("yourprovider", call_yourprovider_async),
}
```

The key is the model prefix (lowercase). When a model name contains this prefix, it's routed to the specified provider. The dispatcher checks prefixes in order and falls back to `DEFAULT_PROVIDER` if no match is found.

### Step 5: Add Pricing Information

Add pricing for your models to `src/scout/llm/pricing.py`. The pricing dictionary format is:

```python
PRICING = {
    "model-id": {
        "input_per_million": 0.05,   # Cost per million input tokens (USD)
        "output_per_million": 0.08,   # Cost per million output tokens (USD)
        "nickname": "optional",       # Human-readable name (optional)
    },
    # Add more models...
}
```

Example:

```python
# Add to PRICING dictionary in src/scout/llm/pricing.py
PRICING = {
    # ... existing entries ...

    # Your provider's models
    "your-model-id": {
        "input_per_million": 0.10,
        "output_per_million": 0.20,
        "nickname": "your_nickname",
    },
}
```

### Step 6: Define Model List (Optional but Recommended)

For validation and documentation purposes, define a model list in your base API implementation file. This helps track supported models and their capabilities:

```python
# In src/scout/llm/{provider}.py

YOUR_PROVIDER_MODELS = {
    "model-id-1": {
        "context": 1_000_000,        # Context window size
        "output": 8_192,              # Max output tokens
        "cost_per_1k_input": 0.0001,  # Legacy field (use pricing.py)
        "cost_per_1k_output": 0.0002,
    },
    "model-id-2": {
        "context": 2_000_000,
        "output": 16_384,
        "cost_per_1k_input": 0.0002,
        "cost_per_1k_output": 0.0004,
    },
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

# Correct signature: takes model_id, input_tokens, output_tokens
cost_usd = estimate_cost_usd(
    model_id=model,
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

## Circuit Breaker Integration

The system includes circuit breaker protection to prevent cascading failures. The circuit breaker is implemented in `scout/llm/circuit_breaker.py`.

### Checking Circuit State

```python
from scout.llm.circuit_breaker import is_provider_available, record_success, record_failure, CircuitOpenError

# Check if provider is available before making a call
if not is_provider_available("providername"):
    raise CircuitOpenError("Provider circuit is open - provider is temporarily unavailable")

# After successful API call
record_success("providername")

# After failed API call
record_failure("providername", permanent=False)  # Temporary failure - may retry
record_failure("providername", permanent=True)   # Permanent failure (e.g., auth error)
```

### Circuit Breaker States

- **CLOSED**: Normal operation, requests allowed
- **OPEN**: Provider failing, requests blocked (after 5 failures)
- **HALF_OPEN**: Testing if provider recovered (after 300 seconds cooldown)

### Circuit Breaker Configuration

The default thresholds are defined in `CircuitBreaker`:

```python
class CircuitBreaker:
    FAILURE_THRESHOLD = 5   # Failures before opening circuit
    COOLDOWN_SECONDS = 300   # Seconds before attempting recovery
```

## Retry Logic

The retry system provides exponential backoff with budget and audit support. It's implemented in `scout/llmtry.py`.

###/re Using call_with_retries

```python
from scout.llm.retry import call_with_retries, LLMCallContext

# Simple usage
result = await call_with_retries(
    your_provider_call,
    arg1, arg2,
    estimated_cost=0.01,
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
)

# With budget and audit context
from scout.llm.budget import BudgetService
from scout.audit import get_audit

budget_service = BudgetService(config, audit)
reservation = budget_service.reserve(estimated_cost=0.01, operation="your_operation")

context = LLMCallContext(
    budget_service=budget_service,
    reservation_id=reservation.id,
    audit_log=audit,
    model="your-model",
    provider="yourprovider",
    operation="your_operation",
)

result = await call_with_retries(
    your_provider_call,
    context=context,
    estimated_cost=0.01,
    max_retries=3,
)
```

### LLMCallContext Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `budget_service` | BudgetService | Optional budget service for cost management |
| `reservation_id` | str | Reservation ID from budget_service.reserve() |
| `audit_log` | AuditLog | Optional audit logger |
| `model` | str | Model name for logging |
| `provider` | str | Provider name for logging |
| `operation` | str | Operation name for tracking |
| `cost_extractor` | callable | Function to extract cost from response |

### Retry Exceptions

By default, retries are performed on:
- `TimeoutError`
- `ConnectionError`
- `httpx.HTTPStatusError`
- `httpx.ConnectError`
- OpenAI: `APIError`, `RateLimitError`, `APITimeoutError`
- Anthropic: `APIError`, `RateLimitError`

You can specify custom exceptions:

```python
result = await call_with_retries(
    your_provider_call,
    retry_on=(TimeoutError, ConnectionError, CustomError),
)
```

## Budget Service

The budget service provides centralized cost management with reservation semantics. It's implemented in `src/scout/llm/budget.py`.

### Checking Budget Before a Call

```python
from scout.llm.budget import BudgetService, InsufficientBudgetError

budget_service = BudgetService(config, audit)

# Check if operation can proceed
if not budget_service.check(estimated_cost=0.01, operation="my_operation"):
    raise InsufficientBudgetError(requested=0.01, available=budget_service.get_remaining(), operation="my_operation")
```

### Using Reservations

```python
# Reserve budget before LLM call
reservation = budget_service.reserve(
    estimated_cost=0.01,
    operation="my_operation",
    timeout_seconds=30,
)

# Use context manager for automatic commit/rollback
with reservation as res:
    result = await your_provider_call(...)
    # On success: budget is committed automatically
    # On exception: budget is rolled back automatically
```

### Budget Service Methods

| Method | Description |
|--------|-------------|
| `check(estimated_cost, operation)` | Check if budget allows operation |
| `reserve(estimated_cost, operation)` | Reserve budget, returns context manager |
| `commit(reservation, actual_cost)` | Commit actual cost |
| `rollback(reservation)` | Release reserved budget |
| `get_remaining()` | Get remaining hourly budget |

### Budget Configuration

Configure in your config file or environment:

```yaml
limits:
  hourly_budget: 1.00  # USD per hour
  hard_safety_cap: 10.00

budget:
  reservation_timeout_seconds: 30
  allow_overage_percent: 10
```

Environment variables:
- `SCOUT_HOURLY_BUDGET` - Hourly budget in USD

## Timeout Configuration

Timeout configuration is managed in `src/scout/llm/timeout_config.py`.

### Default Timeouts

```python
from scout.llm.timeout_config import get_timeout_config

config = get_timeout_config()

# Default timeouts (in seconds)
# connect_timeout: 10.0
# read_timeout: 60.0
```

### Provider-Specific Timeouts

```python
# Get timeouts for a specific provider
connect_timeout, read_timeout = config.for_provider("yourprovider")

# Environment variable overrides
# SCOUT_CONNECT_TIMEOUT=10.0
# SCOUT_READ_TIMEOUT=60.0
# SCOUT_DEEPSEEK_CONNECT_TIMEOUT=10.0
# SCOUT_DEEPSEEK_READ_TIMEOUT=60.0
```

### Setting Provider-Specific Timeouts

To add provider-specific timeouts, extend `TimeoutConfig` in `src/scout/llm/timeout_config.py`:

```python
@dataclass
class TimeoutConfig:
    connect_timeout: float = 10.0
    read_timeout: float = 60.0

    # Add your provider
    yourprovider_connect: float = 10.0
    yourprovider_read: float = 90.0

    def for_provider(self, provider: str) -> tuple[float, float]:
        if provider == "yourprovider":
            return self.yourprovider_connect, self.yourprovider_read
        # ... other providers
        else:
            return self.connect_timeout, self.read_timeout
```

## Fallback Providers

For resilience, you can define fallback chains that try alternative models if the primary fails.

### Using call_llm_async_with_fallback

```python
from scout.llm.dispatch import call_llm_async_with_fallback

# Primary model with fallbacks
response, cost = await call_llm_async_with_fallback(
    model="expensive-model",
    prompt="Your prompt",
    system="Optional system prompt",
    max_tokens=2048,
    temperature=0.0,
    fallback_models=[
        "cheap-model-1",
        "cheap-model-2",  # Last resort
    ],
)
```

The function tries each model in order until one succeeds. If all fail, it raises the last exception.

### Defining Fallback Chains in Your Provider

You can also implement custom fallback logic:

```python
async def call_with_fallback(prompt: str, model: str = "primary-model"):
    models = [model, "fallback-1", "fallback-2"]

    for attempt_model in models:
        try:
            return await call_llm_async(prompt=prompt, model=attempt_model, ...)
        except Exception as e:
            logger.warning(f"Model {attempt_model} failed: {e}")
            continue

    raise AllModelsFailedError(f"All models failed: {models}")
```

## Testing Your Provider

### Unit Tests

Create tests in `tests/scout/llm/test_provider.py`:

```python
import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock

from scout.llm.providers import registry, ProviderClient


@pytest.fixture
def mock_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("TESTPROVIDER_API_KEY", "test-key-123")


@pytest.fixture
def mock_provider_module(mock_env):
    """Mock provider module for testing."""
    with patch.dict(os.environ, {"TESTPROVIDER_API_KEY": "test-key-123"}):
        # Force reimport to pick up mocked env
        import importlib
        if "scout.llm.providers.testprovider" in sys.modules:
            del sys.modules["scout.llm.providers.testprovider"]
        yield


async def test_provider_registers():
    """Test that provider registers correctly."""
    # Import to trigger registration
    from scout.llm.providers import testprovider  # noqa: F401

    assert registry.available("testprovider")
    client = registry.get("testprovider")
    assert len(client.keys) == 1
    assert client.keys[0].key == "test-key-123"


async def test_provider_call_returns_result():
    """Test provider call returns ProviderResult."""
    from scout.llm.providers import testprovider
    from scout.llm import LLMResponse

    with patch("scout.llm.testprovider.call_testprovider_async") as mock_call:
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
        assert result.cost_usd == 0.0001


async def test_provider_key_rotation():
    """Test that failed keys are rotated."""
    from scout.llm.providers import testprovider

    client = registry.get("testprovider")
    initial_key = client.keys[0].key

    # Simulate failure
    client.record_key_failure(initial_key, permanent=False)

    # Should get same key (not permanent)
    assert client.get_working_key() is not None


async def test_estimate_cost_usd_signature():
    """Test that estimate_cost_usd works with correct signature."""
    from scout.llm.pricing import estimate_cost_usd

    # Correct: model_id, input_tokens, output_tokens
    cost = estimate_cost_usd(
        model_id="test-model",
        input_tokens=1000,
        output_tokens=500,
    )

    assert isinstance(cost, float)
    assert cost >= 0
```

### Integration Tests

Test actual API calls with mocked responses:

```python
import pytest
import os

@pytest.mark.integration
async def test_provider_integration():
    """Test actual API call (requires API key)."""
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


@pytest.mark.integration
async def test_pricing_lookup():
    """Test that pricing is correctly configured."""
    from scout.llm.pricing import PRICING

    # Verify at least one model has pricing
    assert len(PRICING) > 0

    # Verify pricing structure
    for model_id, pricing in PRICING.items():
        assert "input_per_million" in pricing
        assert "output_per_million" in pricing
        assert isinstance(pricing["input_per_million"], (int, float))
```

### Mocking External API Calls

For testing without making actual API calls:

```python
from unittest.mock import AsyncMock, patch

async def test_with_mocked_api():
    """Test provider with mocked API responses."""
    mock_response = {
        "choices": [{"message": {"content": "Mocked response"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = AsyncMock(
            raise_for_status=lambda: None,
            json=lambda: mock_response,
        )

        from scout.llm import call_yourprovider_async
        result = await call_yourprovider_async(prompt="test")

        assert result.content == "Mocked response"
```

## Configuration Reference

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `{PROVIDER}_API_KEY` | Single API key | `GROQ_API_KEY=sk-...` |
| `{PROVIDER}_API_KEYS` | Comma-separated keys | `GROQ_API_KEYS=sk-1,sk-2` |
| `{PROVIDER}_API_URL` | Custom endpoint | `GROQ_API_URL=https://...` |
| `SCOUT_HOURLY_BUDGET` | Hourly budget (USD) | `SCOUT_HOURLY_BUDGET=5.00` |
| `SCOUT_CONNECT_TIMEOUT` | Default connect timeout | `SCOUT_CONNECT_TIMEOUT=15.0` |
| `SCOUT_READ_TIMEOUT` | Default read timeout | `SCOUT_READ_TIMEOUT=120.0` |

### ProviderClient Configuration

When creating a `ProviderClient`:

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

### Model Not Routing to Provider

1. Check `PROVIDER_MAP` in `dispatch.py`
2. Verify model name prefix matches (case-insensitive)
3. Ensure fallback to `DEFAULT_PROVIDER` is working

### Cost Calculation Issues

1. Verify `estimate_cost_usd` signature: `(model_id, input_tokens, output_tokens)`
2. Check pricing entry exists in `PRICING` dictionary
3. Verify token counts are extracted from API response

---

For more details, see:

- `src/scout/llm/providers/__init__.py` - Full provider infrastructure
- `src/scout/llm/providers/groq.py` - Simple provider example
- `src/scout/llm/providers/anthropic.py` - Complex provider with more features
- `src/scout/llm/pricing.py` - Cost calculation
- `src/scout/llm/dispatch.py` - Model routing (PROVIDER_MAP)
- `src/scout/llm/retry.py` - Retry logic with budget support
- `src/scout/llm/circuit_breaker.py` - Circuit breaker implementation
- `src/scout/llm/budget.py` - Budget service
- `src/scout/llm/timeout_config.py` - Timeout configuration
- `src/scout/audit.py` - Audit logging
