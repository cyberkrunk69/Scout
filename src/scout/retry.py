"""
Retry logic with exponential backoff for self-healing batch execution.

Features:
- Configurable retry policies
- Exponential backoff with jitter
- Progress reporting integration
- Circuit breaker integration
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Optional, Type, Union

from scout.progress import ProgressReporter, Status, ProgressEvent


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay_ms: float = 1000.0
    max_delay_ms: float = 30000.0
    backoff_multiplier: float = 2.0
    jitter: float = 0.1  # 10% jitter
    retryable_exceptions: tuple = (
        ConnectionError,
        TimeoutError,
        asyncio.TimeoutError,
        OSError,
    )
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)


@dataclass
class RetryStats:
    """Statistics for retry attempts."""
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    total_delay_ms: float = 0.0
    last_error: Optional[Exception] = None


def calculate_backoff(attempt: int, config: RetryConfig) -> float:
    """
    Calculate delay for given attempt with exponential backoff and jitter.
    
    Formula: min(base * (multiplier ^ attempt) + jitter, max_delay)
    """
    delay = config.base_delay_ms * (config.backoff_multiplier ** attempt)
    delay = min(delay, config.max_delay_ms)
    
    # Add jitter
    jitter_range = delay * config.jitter
    delay += random.uniform(-jitter_range, jitter_range)
    
    return max(0, delay)


def is_retryable(error: Exception, config: RetryConfig) -> bool:
    """Check if error is retryable."""
    if isinstance(error, config.retryable_exceptions):
        return True
    
    # Check for HTTP status code in error message
    if hasattr(error, 'status_code'):
        return error.status_code in config.retryable_status_codes
    
    error_str = str(error).lower()
    retryable_keywords = ['timeout', 'connection', 'network', 'temporarily']
    return any(kw in error_str for kw in retryable_keywords)


class RetryContext:
    """Context for retry operations."""
    
    def __init__(
        self,
        task_id: str,
        config: RetryConfig,
        reporter: Optional[ProgressReporter] = None,
    ):
        self.task_id = task_id
        self.config = config
        self.reporter = reporter
        self.stats = RetryStats()
        self._cancelled = False
    
    @property
    def can_retry(self) -> bool:
        return (
            not self._cancelled 
            and self.stats.attempts < self.config.max_retries
        )
    
    @property
    def attempt(self) -> int:
        return self.stats.attempts
    
    def cancel(self) -> None:
        """Cancel retry attempts."""
        self._cancelled = True
    
    async def wait_before_retry(self) -> float:
        """Wait with backoff and emit progress event."""
        if not self.can_retry:
            return 0.0
        
        delay_ms = calculate_backoff(self.stats.attempts, self.config)
        self.stats.total_delay_ms += delay_ms
        
        if self.reporter:
            event = ProgressEvent(
                task_id=self.task_id,
                status=Status.RETRY,
                message=f"Retry attempt {self.stats.attempts + 1}/{self.config.max_retries}",
                metadata={
                    "retry_count": self.stats.attempts + 1,
                    "max_retries": self.config.max_retries,
                    "backoff_ms": int(delay_ms),
                    "error_type": type(self.stats.last_error).__name__ if self.stats.last_error else None,
                }
            )
            await self.reporter.emit_async(event)
        
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        
        return delay_ms


async def with_retry_async(
    func: Callable,
    *args,
    task_id: str = "unknown",
    config: Optional[RetryConfig] = None,
    reporter: Optional[ProgressReporter] = None,
    **kwargs,
) -> Any:
    """
    Execute async function with retry logic.
    
    Args:
        func: Async function to execute
        *args: Positional arguments for func
        task_id: Identifier for progress reporting
        config: Retry configuration
        reporter: Progress reporter
        **kwargs: Keyword arguments for func
    
    Returns:
        Result of func(*args, **kwargs)
    
    Raises:
        Last exception if all retries exhausted
    """
    config = config or RetryConfig()
    ctx = RetryContext(task_id, config, reporter)
    
    while ctx.can_retry:
        ctx.stats.attempts += 1
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            ctx.stats.successes += 1
            
            if ctx.stats.attempts > 1 and reporter:
                # Report recovery after retry
                event = ProgressEvent(
                    task_id=task_id,
                    status=Status.SUCCESS,
                    message=f"Succeeded after {ctx.stats.attempts} attempts",
                    metadata={
                        "retry_count": ctx.stats.attempts - 1,
                        "total_delay_ms": int(ctx.stats.total_delay_ms),
                    }
                )
                await reporter.emit_async(event)
            
            return result
            
        except Exception as e:
            ctx.stats.last_error = e
            
            if not is_retryable(e, config):
                ctx.stats.failures += 1
                raise
            
            if not ctx.can_retry:
                ctx.stats.failures += 1
                raise
            
            await ctx.wait_before_retry()
    
    # All retries exhausted
    ctx.stats.failures += 1
    
    if reporter:
        event = ProgressEvent(
            task_id=task_id,
            status=Status.FAILURE,
            message=f"Failed after {ctx.stats.attempts} attempts",
            metadata={
                "retry_count": ctx.stats.attempts,
                "error_type": type(ctx.stats.last_error).__name__ if ctx.stats.last_error else "unknown",
                "total_delay_ms": int(ctx.stats.total_delay_ms),
            }
        )
        await reporter.emit_async(event)
    
    raise ctx.stats.last_error


def with_retry_sync(
    func: Callable,
    *args,
    task_id: str = "unknown",
    config: Optional[RetryConfig] = None,
    **kwargs,
) -> Any:
    """Synchronous version of with_retry."""
    import time
    
    config = config or RetryConfig()
    stats = RetryStats()
    
    while stats.attempts < config.max_retries:
        stats.attempts += 1
        
        try:
            result = func(*args, **kwargs)
            stats.successes += 1
            return result
            
        except Exception as e:
            stats.last_error = e
            
            if not is_retryable(e, config):
                stats.failures += 1
                raise
            
            if stats.attempts >= config.max_retries:
                stats.failures += 1
                raise
            
            delay_ms = calculate_backoff(stats.attempts, config)
            stats.total_delay_ms += delay_ms
            time.sleep(delay_ms / 1000.0)
    
    raise stats.last_error


def retry_async(
    task_id: Optional[str] = None,
    config: Optional[RetryConfig] = None,
):
    """Decorator for async functions with retry logic."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tid = task_id or func.__name__
            return await with_retry_async(
                func, *args,
                task_id=tid,
                config=config,
                **kwargs
            )
        return wrapper
    return decorator


def retry_sync(
    task_id: Optional[str] = None,
    config: Optional[RetryConfig] = None,
):
    """Decorator for sync functions with retry logic."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tid = task_id or func.__name__
            return with_retry_sync(
                func, *args,
                task_id=tid,
                config=config,
                **kwargs
            )
        return wrapper
    return decorator
