"""Tests for LLM dispatch."""

import pytest
from unittest.mock import AsyncMock, patch
from scout.llm.dispatch import (
    get_provider_for_model,
    get_provider_info,
)


def test_get_provider_for_groq_models():
    """Test provider mapping for Groq models."""
    provider, func = get_provider_for_model("llama-3.1-8b-instant")
    assert provider == "groq"
    
    provider, func = get_provider_for_model("mixtral-8x7b-32768")
    assert provider == "groq"


def test_get_provider_for_google_models():
    """Test provider mapping for Google models."""
    provider, func = get_provider_for_model("gemini-2.0-flash")
    assert provider == "google"


def test_get_provider_for_minimax_models():
    """Test provider mapping for MiniMax models."""
    provider, func = get_provider_for_model("MiniMax-M2.5")
    assert provider == "minimax"
    
    provider, func = get_provider_for_model("abab6.5s")
    assert provider == "minimax"


def test_get_provider_for_anthropic_models():
    """Test provider mapping for Anthropic models."""
    provider, func = get_provider_for_model("claude-3-5-sonnet-20241022")
    assert provider == "anthropic"


def test_get_provider_for_unknown_model():
    """Test fallback for unknown models."""
    provider, func = get_provider_for_model("unknown-model-xyz")
    assert provider == "groq"  # Default fallback


def test_get_provider_info():
    """Test provider info retrieval."""
    info = get_provider_info("gemini-2.0-flash")
    assert info["provider"] == "google"
    assert info["supported"] is True
    assert info["model"] == "gemini-2.0-flash"


def test_get_provider_info_unknown():
    """Test provider info for unknown model."""
    info = get_provider_info("completely-unknown-model")
    assert info["provider"] == "groq"  # Default
    assert info["supported"] is True
