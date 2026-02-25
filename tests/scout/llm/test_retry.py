"""Tests for retry logic."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from scout.llm.retry import (
    LLMCallContext,
    call_with_retries,
    BudgetExhaustedError,
)


def test_llm_call_context_creation():
    """Test LLMCallContext initialization."""
    context = LLMCallContext(
        model="test-model",
        provider="test-provider",
        operation="test-op",
    )
    assert context.model == "test-model"
    assert context.provider == "test-provider"
    assert context.operation == "test-op"


def test_llm_call_context_budget_requires_reservation():
    """Test that budget_service requires reservation_id."""
    with pytest.raises(ValueError):
        LLMCallContext(
            budget_service="fake_service",
            reservation_id=None,
        )


@pytest.mark.asyncio
async def test_call_with_retries_success():
    """Test successful call with no retries."""
    async def successful_call():
        return "success"
    
    result = await call_with_retries(
        successful_call,
        max_retries=3,
    )
    assert result == "success"


@pytest.mark.asyncio
async def test_call_with_retries_exhausted():
    """Test retries exhausted."""
    call_count = 0
    
    async def failing_call():
        nonlocal call_count
        call_count += 1
        raise ConnectionError("Network error")
    
    with pytest.raises(ConnectionError):
        await call_with_retries(
            failing_call,
            max_retries=3,
        )
    
    assert call_count == 3


@pytest.mark.asyncio
async def test_call_with_retries_succeeds_on_retry():
    """Test successful retry after initial failure."""
    call_count = 0
    
    async def eventually_successful():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Temporary error")
        return "success"
    
    result = await call_with_retries(
        eventually_successful,
        max_retries=3,
        base_delay=0.01,  # Fast for testing
    )
    
    assert result == "success"
    assert call_count == 2


@pytest.mark.asyncio
async def test_call_with_retries_cost_extractor():
    """Test cost extraction from result."""
    result_obj = type('Result', (), {'cost_usd': 0.05})()
    
    async def call_with_cost():
        return result_obj
    
    context = LLMCallContext(
        model="test",
        provider="test",
    )
    
    result = await call_with_retries(
        call_with_cost,
        context=context,
        max_retries=1,
    )
    
    assert result == result_obj
