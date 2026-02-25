"""Tests for resilient_llm_client.py module."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from scout.resilient_llm_client import (
    LLMResponse,
    ResilientLLMClient,
    get_resilient_client,
)


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""

    def test_creation(self):
        """Test creating an LLMResponse."""
        response = LLMResponse(
            content="Hello, world!",
            cost_usd=0.001,
            tokens=10,
            provider="deepseek",
            model="deepseek-chat",
        )

        assert response.content == "Hello, world!"
        assert response.cost_usd == 0.001
        assert response.tokens == 10
        assert response.provider == "deepseek"
        assert response.model == "deepseek-chat"


class TestResilientLLMClient:
    """Tests for ResilientLLMClient class."""

    @patch("scout.resilient_llm_client.get_circuit_breaker_manager")
    @patch("scout.resilient_llm_client.get_rate_limiter")
    @patch("scout.resilient_llm_client.get_timeout_config")
    def test_creation(
        self, mock_timeout, mock_rate_limiter, mock_circuit
    ):
        """Test creating a ResilientLLMClient."""
        mock_circuit.return_value = MagicMock()
        mock_rate_limiter.return_value = MagicMock()
        mock_timeout.return_value = MagicMock()

        client = ResilientLLMClient()

        assert client.retry_config is not None
        assert client.timeout_config is not None

    @patch("scout.resilient_llm_client.get_circuit_breaker_manager")
    @patch("scout.resilient_llm_client.get_rate_limiter")
    @patch("scout.resilient_llm_client.get_timeout_config")
    def test_get_status(
        self, mock_timeout, mock_rate_limiter, mock_circuit
    ):
        """Test getting client status."""
        mock_circuit.return_value = MagicMock()
        mock_rate_limiter.return_value = MagicMock()
        mock_timeout.return_value = MagicMock()

        client = ResilientLLMClient()

        status = client.get_status()

        assert "circuit_breakers" in status
        assert "rate_limiters" in status


class TestGetResilientClient:
    """Tests for get_resilient_client function."""

    @patch("scout.resilient_llm_client.get_circuit_breaker_manager")
    @patch("scout.resilient_llm_client.get_rate_limiter")
    @patch("scout.resilient_llm_client.get_timeout_config")
    def test_singleton(
        self, mock_timeout, mock_rate_limiter, mock_circuit
    ):
        """Test that get_resilient_client returns singleton."""
        # Reset global
        import scout.resilient_llm_client as rlc_module
        rlc_module._client = None

        mock_circuit.return_value = MagicMock()
        mock_rate_limiter.return_value = MagicMock()
        mock_timeout.return_value = MagicMock()

        client1 = get_resilient_client()
        client2 = get_resilient_client()

        assert client1 is client2
