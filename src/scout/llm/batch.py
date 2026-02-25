"""Batching utility for efficient LLM calls."""

import asyncio
from typing import Callable, List, Awaitable, Union, Optional, Any

# Type alias for rate limiter - will import from .ratelimit when available
RateLimiter = Any


async def batch_process(
    prompts: List[str],
    func: Callable[[str], Awaitable[Any]],
    max_concurrent: int = 5,
    rate_limiter: Optional[RateLimiter] = None,
    return_exceptions: bool = False,
) -> List[Union[Any, Exception]]:
    """
    Process a list of prompts with concurrency control.

    Args:
        prompts: List of prompt strings.
        func: Async function that takes a prompt and returns a value.
        max_concurrent: Max number of simultaneous calls.
        rate_limiter: Optional RateLimiter instance with wait_if_needed() method.
        return_exceptions: If True, return exceptions instead of raising.

    Returns:
        List of results (or Exceptions) in the same order as prompts.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded(prompt: str) -> Any:
        async with semaphore:
            if rate_limiter is not None:
                await rate_limiter.wait_if_needed()
            return await func(prompt)

    tasks = [bounded(p) for p in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
    return list(results)
