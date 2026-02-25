"""Tests for LLM router."""

import pytest
from unittest.mock import AsyncMock, patch
from scout.llm.router import (
    LLMResult,
    LLMResponse,
    get_router_status,
    get_tier_for_task,
    get_tier_for_model,
    get_next_tier,
    is_quota_error,
    PROVIDER_FALLBACKS,
    TierFallbackManager,
)


def test_llm_result_creation():
    """Test LLMResult dataclass."""
    result = LLMResult(
        content="Test response",
        cost_usd=0.01,
        model="test-model",
        provider="test-provider",
        input_tokens=100,
        output_tokens=50,
    )
    
    assert result.content == "Test response"
    assert result.cost_usd == 0.01
    assert result.model == "test-model"
    assert result.provider == "test-provider"
    assert result.input_tokens == 100
    assert result.output_tokens == 50


def test_llm_response_creation():
    """Test LLMResponse dataclass."""
    response = LLMResponse(
        success=True,
        result="Test result",
        error=None,
        metadata={"model": "test"},
    )
    
    assert response.success is True
    assert response.result == "Test result"
    assert response.metadata["model"] == "test"


def test_get_tier_for_task():
    """Test tier lookup for task types."""
    tier = get_tier_for_task("simple")
    assert tier in ["fast", "medium", "large"]


def test_get_tier_for_model():
    """Test tier lookup for models."""
    tier = get_tier_for_model("llama-3.1-8b-instant")
    assert tier is not None


def test_get_next_tier():
    """Test next tier calculation."""
    assert get_next_tier("fast") == "medium"
    assert get_next_tier("medium") == "large"
    assert get_next_tier("large") is None


def test_get_router_status():
    """Test router status retrieval."""
    status = get_router_status()
    
    assert "mode" in status
    assert "providers" in status
    assert "rate_limiter" in status


def test_is_quota_error():
    """Test quota error detection."""
    assert is_quota_error(Exception("quota exceeded")) is True
    assert is_quota_error(Exception("insufficient credits")) is True
    assert is_quota_error(Exception("no credits left")) is True
    assert is_quota_error(Exception("some other error")) is False


def test_tier_fallback_manager():
    """Test TierFallbackManager."""
    manager = TierFallbackManager()
    
    # Record failures (max_failures = 3, so 3 failures makes it unavailable)
    manager.record_failure("provider1")
    manager.record_failure("provider1")
    manager.record_failure("provider1")
    assert manager.is_available("provider1") is False
    
    # Record success
    manager.record_success("provider1")
    assert manager.is_available("provider1") is True
    
    # Unknown provider
    assert manager.is_available("unknown") is True


def test_provider_fallbacks_config():
    """Test provider fallback configuration."""
    assert "groq" in PROVIDER_FALLBACKS
    assert "minimax" in PROVIDER_FALLBACKS
    assert "openrouter" in PROVIDER_FALLBACKS
