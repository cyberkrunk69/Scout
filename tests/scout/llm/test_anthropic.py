"""Tests for Anthropic provider."""

import pytest
from unittest.mock import AsyncMock, patch
from scout.llm.providers import registry


@pytest.mark.asyncio
async def test_call_anthropic_returns_provider_result():
    """Test that _call_anthropic returns ProviderResult."""
    from scout.llm.providers.anthropic import _call_anthropic
    
    with patch('scout.llm.anthropic.call_anthropic_async', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("Test response", 0.005)
        
        result = await _call_anthropic(
            model="claude-3-5-sonnet-20241022",
            prompt="Test prompt",
            system="You are helpful.",
            max_tokens=100,
            temperature=0.0,
        )
        
        assert result.provider == "anthropic"
        assert result.model == "claude-3-5-sonnet-20241022"
        assert result.response_text == "Test response"
        assert result.cost_usd == 0.005


def test_anthropic_provider_registered():
    """Test that Anthropic provider is registered in the registry."""
    providers = registry.list_providers()
    assert "anthropic" in providers


def test_anthropic_provider_has_keys():
    """Test Anthropic provider has correct configuration."""
    provider = registry.get("anthropic")
    assert provider.name == "anthropic"
    assert provider.env_key_name == "ANTHROPIC_API_KEYS"
    assert provider.env_single_key_name == "ANTHROPIC_API_KEY"
