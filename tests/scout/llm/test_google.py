"""Tests for Google Gemini provider."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from scout.llm.providers import registry


@pytest.fixture
def mock_google_response():
    """Mock Google API response."""
    return {
        "candidates": [{
            "content": {
                "parts": [{"text": "Test response"}]
            }
        }],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 50
        }
    }


@pytest.mark.asyncio
async def test_call_google_returns_provider_result(mock_google_response):
    """Test that _call_google returns ProviderResult."""
    from scout.llm.providers.google import _call_google
    
    with patch('scout.llm.google.call_google_async', new_callable=AsyncMock) as mock_call:
        mock_call.return_value = ("Test response", 0.001)
        
        result = await _call_google(
            model="gemini-2.0-flash",
            prompt="Test prompt",
            system="You are helpful.",
            max_tokens=100,
            temperature=0.0,
        )
        
        assert result.provider == "google"
        assert result.model == "gemini-2.0-flash"
        assert result.response_text == "Test response"
        assert result.cost_usd == 0.001


def test_google_provider_registered():
    """Test that Google provider is registered in the registry."""
    providers = registry.list_providers()
    assert "google" in providers


def test_google_provider_has_keys():
    """Test Google provider has correct configuration."""
    provider = registry.get("google")
    assert provider.name == "google"
    assert provider.env_key_name == "GOOGLE_API_KEYS"
    assert provider.env_single_key_name == "GEMINI_API_KEY"
