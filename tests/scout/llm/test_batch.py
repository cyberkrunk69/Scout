"""Tests for the batch processing utility."""

import pytest
import asyncio
from scout.llm.batch import batch_process


@pytest.mark.asyncio
async def test_batch_process_basic():
    """Test that batch_process returns results in order."""
    async def double(prompt: str) -> str:
        return prompt * 2

    prompts = ["a", "b", "c"]
    results = await batch_process(prompts, double)
    assert results == ["aa", "bb", "cc"]


@pytest.mark.asyncio
async def test_batch_process_limits_concurrency():
    """Test that concurrency limit is respected."""
    active_count = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def slow_func(prompt: str) -> str:
        nonlocal active_count, max_concurrent
        async with lock:
            active_count += 1
            max_concurrent = max(max_concurrent, active_count)
        await asyncio.sleep(0.05)
        async with lock:
            active_count -= 1
        return prompt

    prompts = ["a"] * 10
    results = await batch_process(prompts, slow_func, max_concurrent=3)

    assert max_concurrent <= 3
    assert results == prompts


@pytest.mark.asyncio
async def test_batch_process_with_rate_limiter_mock():
    """Test that rate limiter integration works when provided."""
    class MockRateLimiter:
        def __init__(self):
            self.wait_called = False

        async def wait_if_needed(self):
            self.wait_called = True

    rate_limiter = MockRateLimiter()

    async def dummy(prompt: str) -> str:
        return prompt

    results = await batch_process(
        ["a", "b"], dummy, rate_limiter=rate_limiter
    )

    assert results == ["a", "b"]
    assert rate_limiter.wait_called


@pytest.mark.asyncio
async def test_batch_process_without_rate_limiter():
    """Test that rate limiter is optional."""
    async def dummy(prompt: str) -> str:
        return prompt

    results = await batch_process(["a"], dummy)
    assert results == ["a"]


@pytest.mark.asyncio
async def test_batch_process_return_exceptions():
    """Test return_exceptions=True returns exceptions instead of raising."""

    async def failing_func(prompt: str) -> str:
        if prompt == "fail":
            raise ValueError("intentional error")
        return prompt

    results = await batch_process(
        ["ok", "fail", "also_ok"],
        failing_func,
        return_exceptions=True
    )

    assert results[0] == "ok"
    assert isinstance(results[1], ValueError)
    assert results[2] == "also_ok"


@pytest.mark.asyncio
async def test_batch_process_raises_on_exception():
    """Test that exceptions are raised by default."""
    async def failing_func(prompt: str) -> str:
        raise ValueError("intentional error")

    with pytest.raises(ValueError):
        await batch_process(["fail"], failing_func)


@pytest.mark.asyncio
async def test_batch_process_empty_list():
    """Test that empty list returns empty list."""
    async def dummy(prompt: str) -> str:
        return prompt

    results = await batch_process([], dummy)
    assert results == []
