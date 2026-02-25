"""Tests for Groq LLM client."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.mark.filterwarnings("ignore::DeprecationWarning")


class MockResponse:
    """Mock httpx response."""
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
        self.request = MagicMock()
    
    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.mark.asyncio
async def test_call_groq_async_with_mock(monkeypatch):
    """Test call_groq_async with mocked HTTP client."""
    from scout.llm import call_groq_async, LLMResponse
    
    # Mock the HTTP client
    mock_response = MockResponse(
        status_code=200,
        json_data={
            "choices": [{
                "message": {"content": "Hello, world!"}
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5
            }
        }
    )
    
    async def mock_post(*args, **kwargs):
        return mock_response
    
    # Mock the audit to avoid file I/O
    mock_audit = MagicMock()
    mock_audit.log = MagicMock()
    
    # Set up environment
    import os
    monkeypatch.setenv("GROQ_API_KEY", "test_key_123")
    monkeypatch.setenv("GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
    
    with patch("scout.llm.get_audit", return_value=mock_audit):
        with patch("httpx.AsyncClient.post", side_effect=mock_post):
            # Need to also mock the client context manager
            with patch("httpx.AsyncClient") as MockClient:
                instance = AsyncMock()
                instance.post = mock_post
                instance.__aenter__ = AsyncMock(return_value=instance)
                instance.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = instance
                
                response = await call_groq_async("Hello", model="llama-3.1-8b-instant")
    
    assert isinstance(response, LLMResponse)
    assert response.content == "Hello, world!"
    assert response.model == "llama-3.1-8b-instant"
    assert response.input_tokens == 10
    assert response.output_tokens == 5


@pytest.mark.asyncio
async def test_llm_response_dataclass():
    """Test LLMResponse dataclass fields."""
    from scout.llm import LLMResponse
    
    response = LLMResponse(
        content="test content",
        cost_usd=0.001,
        model="llama-3.1-8b-instant",
        input_tokens=100,
        output_tokens=50,
        duration_ms=500,
    )
    
    assert response.content == "test content"
    assert response.cost_usd == 0.001
    assert response.model == "llama-3.1-8b-instant"
    assert response.input_tokens == 100
    assert response.output_tokens == 50
    assert response.duration_ms == 500


def test_supported_models():
    """Test that SUPPORTED_MODELS contains expected models."""
    from scout.llm import SUPPORTED_MODELS
    
    assert "llama-3.1-8b-instant" in SUPPORTED_MODELS
    assert "llama-3.1-70b-versatile" in SUPPORTED_MODELS
    assert "llama-3.3-70b-versatile" in SUPPORTED_MODELS
    assert "mixtral-8x7b-32768" in SUPPORTED_MODELS


def test_fallback_model():
    """Test fallback model constant."""
    from scout.llm import FALLBACK_8B_MODEL
    
    assert FALLBACK_8B_MODEL == "llama-3.1-8b-instant"
