"""Tests for MiniMax provider."""

import pytest
from unittest.mock import AsyncMock, patch
from scout.llm.providers import registry
from dataclasses import dataclass


@pytest.fixture
def mock_minimax_result():
    """Mock MiniMax result."""
    @dataclass
    class MockMiniMaxResult:
        response_text: str
        cost_usd: float
        input_tokens: int
        output_tokens: int
        model: str
    return MockMiniMaxResult(
        response_text="Test response",
        cost_usd=0.002,
        input_tokens=100,
        output_tokens=50,
        model="MiniMax-M2"
    )


@pytest.mark.asyncio
async def test_call_minimax_returns_provider_result(mock_minimax_result):
    """Test that _call_minimax returns ProviderResult."""
    from scout.llm.providers.minimax import _call_minimax
    
    with patch('scout.llm.minimax.call_minimax_async_detailed', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_minimax_result
        
        result = await _call_minimax(
            model="MiniMax-M2",
            prompt="Test prompt",
            system="You are helpful.",
            max_tokens=100,
            temperature=0.0,
        )
        
        assert result.provider == "minimax"
        assert result.model == "MiniMax-M2"
        assert result.response_text == "Test response"
        assert result.cost_usd == 0.002


def test_minimax_provider_registered():
    """Test that MiniMax provider is registered in the registry."""
    providers = registry.list_providers()
    assert "minimax" in providers


def test_minimax_provider_has_keys():
    """Test MiniMax provider has correct configuration."""
    provider = registry.get("minimax")
    assert provider.name == "minimax"
    assert provider.env_key_name == "MINIMAX_API_KEYS"
    assert provider.env_single_key_name == "MINIMAX_API_KEY"
