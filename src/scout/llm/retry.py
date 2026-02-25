"""Unified Retry Wrapper for LLM Providers.

Provides:
- LLMCallContext: Context object for safeguard parameters
- call_with_retries(): Add exponential backoff retry with budget/audit support
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Awaitable, TypeVar, Optional, Any

T = TypeVar('T')
logger = logging.getLogger(__name__)


@dataclass
class LLMCallContext:
    """Context object for LLM calls with safeguard parameters.
    
    Groups related parameters to keep function signatures clean.
    """
    budget_service: Any = None
    reservation_id: str = None
    audit_log: Any = None
    model: str = None
    provider: str = None
    operation: str = "llm_call"
    cost_extractor: Callable = None
    request_id: str = None
    
    def __post_init__(self):
        if self.budget_service and not self.reservation_id:
            raise ValueError(
                "budget_service requires reservation_id. "
                "Call budget_service.reserve() before passing context."
            )


RETRY_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
)

def _get_retry_exceptions() -> tuple:
    exceptions = list(RETRY_EXCEPTIONS)
    try:
        import httpx
        exceptions.append(httpx.HTTPStatusError)
        exceptions.append(httpx.ConnectError)
    except ImportError:
        pass
    try:
        from openai import APIError, RateLimitError, APITimeoutError
        exceptions.extend([APIError, RateLimitError, APITimeoutError])
    except ImportError:
        pass
    try:
        from anthropic import APIError, RateLimitError
        exceptions.extend([APIError, RateLimitError])
    except ImportError:
        pass
    return tuple(exceptions)


class BudgetExhaustedError(Exception):
    """Raised when budget check fails mid-retry."""
    pass


async def call_with_retries(
    provider_call: Callable[..., Awaitable[T]],
    *args,
    context: LLMCallContext = None,
    estimated_cost: float = 0.0,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_on: Optional[tuple] = None,
    **kwargs
) -> T:
    """Wrapper with retry, budget, and audit support.
    
    Args:
        provider_call: Async function to call
        *args: Positional args for provider_call
        context: LLMCallContext - REQUIRED if budget/audit/correlation needed.
        estimated_cost: Cost per attempt (fallback if not in context)
        max_retries: Max retry attempts
        base_delay: Base delay for exponential backoff
        max_delay: Max delay cap
        retry_on: Tuple of exceptions to retry on
        **kwargs: Additional kwargs for provider_call
    
    Raises:
        BudgetExhaustedError: If budget check fails during retry loop
    """
    budget_service = context.budget_service if context else None
    audit_log = context.audit_log if context else None
    model = context.model if context else None
    provider = context.provider if context else None
    operation = context.operation if context else "llm_call"
    reservation_id = context.reservation_id if context else None
    request_id = context.request_id if context else None
    
    if budget_service and not context:
        raise ValueError(
            "budget_service requires LLMCallContext. "
            "Build context with budget_service.reserve() before calling."
        )
    
    cost_extractor = None
    if context and context.cost_extractor:
        cost_extractor = context.cost_extractor
    else:
        def default_extractor(result):
            if hasattr(result, 'cost_usd'):
                return result.cost_usd
            elif isinstance(result, tuple) and len(result) >= 2:
                return result[1] if isinstance(result[1], (int, float)) else 0.0
            return 0.0
        cost_extractor = default_extractor
    
    retry_exceptions = retry_on or _get_retry_exceptions()
    last_error = None
    attempts_made = 0
    
    for attempt in range(max_retries):
        if budget_service and not budget_service.check(estimated_cost, operation=operation):
            if audit_log:
                audit_log.log(
                    "budget_exceeded",
                    operation=operation,
                    attempt=attempt+1,
                    max_retries=max_retries,
                    estimated=estimated_cost,
                    reservation_id=reservation_id,
                    action="rollback",
                    request_id=request_id
                )
            if budget_service and reservation_id:
                budget_service.rollback(reservation_id)
            raise BudgetExhaustedError(f"Budget insufficient for {operation}")
        
        try:
            result = await provider_call(*args, **kwargs)
            actual_cost = cost_extractor(result)
            if budget_service and actual_cost > 0:
                budget_service.commit(reservation_id, actual_cost)
                if audit_log:
                    audit_log.log(
                        "reservation_commit",
                        operation=operation,
                        reservation_id=reservation_id,
                        actual_cost=actual_cost,
                        request_id=request_id
                    )
            elif budget_service and actual_cost == 0 and estimated_cost > 0:
                logger.warning(
                    f"Cost tracking missing for {operation} (provider may not return cost). "
                    f"Expected ~{estimated_cost}, got 0."
                )
            return result
        except retry_exceptions as e:
            last_error = e
            attempts_made += 1
            
            if audit_log:
                delay = min(base_delay * (2 ** attempt), max_delay)
                audit_log.log(
                    "llm_retry",
                    attempt=attempt+1,
                    max_retries=max_retries,
                    delay=delay,
                    error=str(e),
                    model=model,
                    provider=provider,
                    estimated_cost=estimated_cost,
                    request_id=request_id,
                )
            
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {delay}s: {e}"
                )
                await asyncio.sleep(delay)
            continue
        except Exception as e:
            last_error = e
            break
    
    if budget_service and reservation_id:
        budget_service.rollback(reservation_id)
        if audit_log:
            audit_log.log(
                "llm_error",
                operation=operation,
                reservation_id=reservation_id,
                reason="retries_exhausted",
                attempts=attempts_made,
                error=str(last_error),
                error_type=type(last_error).__name__ if last_error else None,
                request_id=request_id
            )
    
    if last_error:
        raise last_error
    
    raise RuntimeError("Retry loop exited without result or error")
