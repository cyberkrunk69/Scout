# Technical Report: Provider Registration Guide (Track I7)

## Executive Summary

This report provides a senior-level technical review of the provider registration guide created for Track I7. The guide (`docs/guides/adding_a_provider.md`) provides documentation for adding new LLM providers to scout-core, but analysis reveals several significant gaps, inaccuracies, and areas requiring improvement.

**Verdict: NOT FULLY ASSED** — While the guide provides a reasonable starting point, it contains technical inaccuracies, missing critical components, and incomplete coverage that would prevent a developer from successfully adding a new provider without additional support.

---

## 1. What Was Actually Done

### 1.1 Codebase Exploration
- Analyzed `src/scout/llm/providers/__init__.py` (ProviderRegistry, ProviderClient, KeyState)
- Analyzed `src/scout/llm/providers/groq.py` (example provider wrapper)
- Reviewed `src/scout/audit.py` (audit logging)
- Reviewed parts of `src/scout/llm/__init__.py` (LLMResponse, call_groq_async)

### 1.2 Deliverable Created
- Created `docs/guides/adding_a_provider.md` (490 lines)
- Directory structure: `docs/guides/`
- Committed to git: `1122e2b`

---

## 2. Critical Errors and Inaccuracies

### 2.1 WRONG: Cost Estimation Function Signature

**In the guide:**
```python
cost_usd = estimate_cost_usd(
    provider="providername",
    model=model,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
)
```

**Actual code in `src/scout/llm/pricing.py:66`:**
```python
def estimate_cost_usd(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate actual USD cost from token counts + hardcoded 2026 pricing."""
    pricing = PRICING.get(model_id, PRICING["llama-3.1-8b-instant"])
    return (input_tokens / 1_000_000) * pricing["input_per_million"] + \
           (output_tokens / 1_000_000) * pricing["output_per_million"]
```

**Impact:** Developer would get runtime errors trying to use the documented API.

**Severity:** HIGH — This is a functional blocker.

---

### 2.2 MISSING: Model-to-Provider Routing

**Not documented:** Adding a new provider requires registration in `PROVIDER_MAP` in `src/scout/llm/dispatch.py`:

```22:31:src/scout/llm/dispatch.py
PROVIDER_MAP = {
    "deepseek": ("deepseek", call_deepseek_async),
    "llama": ("groq", call_groq_async),
    "mixtral": ("groq", call_groq_async),
    "gemma": ("groq", call_groq_async),
    "gemini": ("google", call_google_async),
    "claude": ("anthropic", call_anthropic_async),
    "minimax": ("minimax", call_minimax_async),
    "abab6": ("minimax", call_minimax_async),
}
```

**What the guide implies:** The provider wrapper is called directly.

**Reality:** Model names are pattern-matched to providers. A developer would need to add their model prefix to `PROVIDER_MAP`.

**Impact:** Without this, the provider won't be automatically selected when users specify model names.

**Severity:** HIGH — Incomplete implementation path.

---

### 2.3 MISSING: Model List Definition

**Not documented:** Many providers define explicit model lists (e.g., `GEMINI_MODELS`, `CLAUDE_MODELS` in their respective modules). These serve as:
- Validation that the model is supported
- Documentation for users
- Potential routing logic

**Severity:** MEDIUM — Could cause confusion during implementation.

---

### 2.4 MISUNDERSTOOD: Pricing Architecture

**In the guide:** Implies pricing is provider-based:
```python
PROVIDER_NAME = {
    "model-id-1": {"input": 0.0001, "output": 0.0002},
}
```

**Actual architecture:** Pricing is model-based in `PRICING` dictionary keyed by exact model ID:
```11:63:src/scout/llm/pricing.py
PRICING = {
    "gemini-2.5-pro": {
        "input_per_million": 1.25,
        "output_per_million": 10.00,
        "nickname": "galaxy_brain",
    },
    "llama-3.1-8b-instant": {
        "input_per_million": 0.05,
        "output_per_million": 0.08,
        "nickname": "router",
    },
    # ... more models
}
```

**Impact:** Developer would add pricing incorrectly.

**Severity:** MEDIUM — Would cause incorrect cost tracking.

---

## 3. Significant Gaps in Coverage

### 3.1 MISSING: Router/Dispatch Integration

The guide doesn't mention:
- `src/scout/llm/router.py` — the main `call_llm()` function
- `src/scout/llm/dispatch.py` — the dispatcher that routes models to providers
- How fallback providers work (when one fails, tries another)
- The `DEFAULT_PROVIDER` fallback mechanism

### 3.2 MISSING: Circuit Breaker Deep Dive

What's in the guide:
- Basic mention of circuit breaker
- Import statement examples

What's missing:
- How `CircuitBreaker` class works (`src/scout/llm/circuit_breaker.py`)
- Circuit states (CLOSED, OPEN, HALF_OPEN)
- How to check circuit status before calling
- The `CircuitOpenError` exception handling

### 3.3 MISSING: Retry Logic

Not covered:
- `src/scout/llm/retry.py` — `call_with_retries()` function
- Retry context (`RetryContext`)
- How retries interact with key rotation
- Max token budget during retries

### 3.4 MISSING: Budget Service Integration

Not covered:
- `src/scout/llm/budget.py` — BudgetService
- How providers interact with budget limits
- Pre-call budget checking
- Budget reservation/commit pattern

### 3.5 MISSING: Timeout Configuration

Not covered:
- Provider-specific timeout settings
- How timeouts interact with retries
- Timeout configuration in `src/scout/config.py`

---

## 4. Code Quality Issues in the Guide

### 4.1 Placeholder Code

The guide uses generic placeholders like:
- `providername` — inconsistent with actual naming (e.g., "groq", "google")
- `PROVIDER_API_KEY` — should be actual key format (e.g., `GROQ_API_KEY`)
- `ProviderName` — mixing naming conventions

### 4.2 Incomplete Error Handling

Example code in guide:
```python
except Exception as e:
    provider.record_key_failure(...)
    raise
```

Missing:
- Specific exception types to catch
- Logging of errors
- Distinguishing between retriable and permanent errors beyond basic `is_permanent_error()`

### 4.3 Magic Numbers

Guide uses magic numbers without explanation:
- `max_tokens: int = 2048`
- `temperature: float = 0.0`
- `timeout=60.0`

No guidance on what values are appropriate.

### 4.4 Hardcoded Assumptions

- Assumes httpx is available (should mention as dependency)
- Assumes async/await pattern (doesn't mention sync alternatives)
- Assumes OpenAI-compatible API format (not all providers use this)

---

## 5. Areas Not Covered by Tests

### 5.1 Provider Registry Tests

Looking at the test suite, there are tests for:
- `tests/scout/llm/test_minimax.py` — provider registration tests
- `tests/scout/llm/test_google.py` — provider registration tests
- `tests/scout/llm/test_router.py` — router tests

What's missing from guide:
- How to write registration tests
- How to mock provider responses
- Integration test patterns

### 5.2 No Test File Location Guidance

Guide doesn't mention:
- Where provider tests should live
- Naming conventions (`test_{provider}.py`)
- Required test fixtures

---

## 6. Documentation Gaps

### 6.1 Not Covered

| Topic | Location in Codebase | Covered? |
|-------|---------------------|----------|
| Model prefixes | `dispatch.py` | NO |
| Pricing dictionary | `pricing.py` | INCORRECT |
| Circuit breaker | `circuit_breaker.py` | PARTIAL |
| Retry logic | `retry.py` | NO |
| Budget service | `budget.py` | NO |
| Timeout config | `config.py` | NO |
| Provider fallback | `dispatch.py` | NO |
| Intent detection | `router.py` | NO |
| Quality loop | `router.py` | NO |

### 6.2 Outdated/Deprecated Patterns

Guide references:
- `call_groq_async` — marked as DEPRECATED in actual code
- Should direct to `call_llm` or `generate_with_quality_loop`

---

## 7. Punted Items

The following were identified as out of scope but would be needed for a "full assed" solution:

1. **Complete API reference** — Full parameter documentation for all provider functions
2. **Migration guide** — How to convert existing direct API calls to use the registry
3. **Performance benchmarks** — Key rotation overhead, connection pooling
4. **Security considerations** — Key storage, environment variable security
5. **Rate limit handling** — Detailed provider-specific rate limit strategies
6. **Model availability** — How to handle providers that remove models
7. **Error code reference** — Comprehensive list of error types and handling
8. **CI/CD integration** — How to test providers in CI

---

## 8. Code Left Unused / Simplifications Made

### 8.1 Not Used

- The guide doesn't reference `src/scout/llm/cost.py` at all, which contains:
  - `get_provider_for_model()`
  - `is_free_model()`
  - Cost estimation utilities

- The guide doesn't explain the relationship between:
  - `dispatch.py` (low-level routing)
  - `router.py` (high-level with retries, fallback)
  - `providers/` (multi-key support)

### 8.2 Simplifications Made

1. **Single-provider view** — Ignores the cascading/fallback system where multiple providers can be tried
2. **Linear flow** — Doesn't explain the quality loop or when to escalate to more expensive models
3. **Basic error handling** — Doesn't cover advanced patterns like:
   - Partial success handling
   - Token budget exhaustion mid-stream
   - Context window limits

---

## 9. Maintainability and Scalability Assessment

### 9.1 Concerns

1. **Documentation drift risk** — The guide uses example code that could diverge from actual implementations
2. **No version coupling** — No mention of matching guide version to code version
3. **Missing API stability guarantee** — Functions like `estimate_cost_usd` could change signature

### 9.2 Strengths

1. **Modular structure** — Good separation of concerns in actual code
2. **Clear interfaces** — `ProviderResult`, `ProviderClient` are well-defined
3. **Multi-key built-in** — Good foundation for scaling

### 9.3 Scalability

The actual provider architecture supports:
- Multiple keys per provider ✅
- Key health tracking ✅
- Circuit breakers ✅
- Exponential backoff ✅

But the guide doesn't adequately explain how to leverage these for scale.

---

## 10. Recommendations

### HIGH PRIORITY (Blockers)

1. **Fix `estimate_cost_usd` signature** — Use actual function signature
2. **Add PROVIDER_MAP documentation** — Explain model-to-provider routing
3. **Document pricing dictionary format** — Show actual PRICING structure

### MEDIUM PRIORITY

4. Add circuit breaker deep dive
5. Add retry logic section
6. Add budget service integration
7. Document dispatch.py routing
8. Explain fallback chains

### LOW PRIORITY

9. Add security considerations
10. Add performance benchmarks
11. Add migration guide for existing code
12. Add version coupling mechanism

---

## 11. Conclusion

The guide provides a reasonable **conceptual introduction** to the provider architecture but is **functionally incomplete** for a developer to successfully add a new provider. The critical errors (wrong function signature, missing routing) would cause immediate failures.

**Can a developer follow it?** Partially — they'd understand the high-level concepts but would need to:
1. Fix the cost estimation call themselves
2. Discover the PROVIDER_MAP on their own
3. Understand the pricing format through trial and error

**Is this "full assed"?** No. It's approximately 50-60% complete for a production-ready guide.

**Effort to complete:** Estimated 2-4 additional hours to address HIGH and MEDIUM priorities.

---

## Appendix: Files Reviewed

| File | Purpose | Understanding |
|------|---------|---------------|
| `src/scout/llm/providers/__init__.py` | Registry, Client, KeyState | Complete |
| `src/scout/llm/providers/groq.py` | Example wrapper | Complete |
| `src/scout/audit.py` | Audit logging | Surface |
| `src/scout/llm/dispatch.py` | Model routing | Incomplete (not in guide) |
| `src/scout/llm/router.py` | Main LLM caller | Incomplete (not in guide) |
| `src/scout/llm/pricing.py` | Cost estimation | Incomplete (misunderstood) |
| `src/scout/llm/circuit_breaker.py` | Circuit breaker | Not reviewed in detail |
| `src/scout/llm/retry.py` | Retry logic | Not reviewed |
| `src/scout/llm/budget.py` | Budget service | Not reviewed |
