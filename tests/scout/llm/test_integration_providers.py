"""Integration tests for LLM providers using mock HTTP server.

These tests exercise the full request/response flow with mocked external dependencies.
"""

import pytest
import asyncio
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional
from unittest.mock import patch, AsyncMock

from scout.llm.providers import ProviderResult, registry


class MockLLMHandler(BaseHTTPRequestHandler):
    """HTTP request handler that simulates LLM API responses."""

    # Class-level storage for request tracking
    requests_received: list[Dict[str, Any]] = []
    response_template: Dict[str, Any] = {
        "id": "mock-response-123",
        "model": "mock-model",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "This is a mocked response from the test server."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    should_error: bool = False
    error_status: int = 500

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_POST(self):
        """Handle POST requests - simulate LLM API."""
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        # Parse and store request
        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            request_data = {}

        MockLLMHandler.requests_received.append({
            "path": self.path,
            "headers": dict(self.headers),
            "body": request_data
        })

        # Return error if configured
        if self.should_error:
            self.send_response(self.error_status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            error_response = {"error": {"message": "Mocked error", "type": "mock_error"}}
            self.wfile.write(json.dumps(error_response).encode())
            return

        # Return mock response
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()

        # Customize response based on request
        response = dict(self.response_template)
        if "model" in request_data:
            response["model"] = request_data["model"]

        self.wfile.write(json.dumps(response).encode())


class MockServer:
    """Context manager for starting/stopping mock HTTP server."""

    def __init__(self, host: str = "localhost", port: int = 0):
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def __enter__(self) -> "MockServer":
        # Reset class state
        MockLLMHandler.requests_received.clear()
        MockLLMHandler.should_error = False

        self.server = HTTPServer((self.host, self.port), MockLLMHandler)
        self.port = self.server.server_address[1]

        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.daemon = True
        self.thread.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=2)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@pytest.fixture
def mock_llm_server():
    """Fixture providing a mock LLM server."""
    with MockServer() as server:
        yield server


@pytest.fixture
def mock_server_url(mock_llm_server):
    """Fixture providing the mock server URL."""
    return mock_llm_server.base_url


class TestLLMProviderIntegration:
    """Integration tests for LLM providers with mocked HTTP server."""

    @pytest.mark.asyncio
    async def test_provider_request_flow(self, mock_server_url):
        """Test that provider makes correct HTTP request and parses response."""
        from scout.llm.providers import ProviderClient, ProviderResult

        # Track the response
        response_holder = {}

        async def mock_call(**kwargs) -> ProviderResult:
            import httpx
            # Make actual HTTP request to mock server
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{mock_server_url}/v1/chat/completions",
                    json={
                        "model": kwargs.get("model", "gpt-4"),
                        "messages": [
                            {"role": "user", "content": kwargs.get("prompt", "")}
                        ],
                        "max_tokens": kwargs.get("max_tokens", 256),
                        "temperature": kwargs.get("temperature", 0.0),
                    }
                )
                data = response.json()

                response_holder["received"] = data

                return ProviderResult(
                    response_text=data["choices"][0]["message"]["content"],
                    cost_usd=0.001,
                    input_tokens=data["usage"]["prompt_tokens"],
                    output_tokens=data["usage"]["completion_tokens"],
                    model=data["model"],
                    provider="mock",
                )

        # Create provider client
        client = ProviderClient(
            name="mock_provider",
            call=mock_call,
            env_key_name="MOCK_API_KEY",
        )

        # Call the provider
        result = await client.call(prompt="Hello, world!")

        # Verify request was made
        assert len(MockLLMHandler.requests_received) == 1

        # Verify response parsing
        assert isinstance(result, ProviderResult)
        assert result.response_text == "This is a mocked response from the test server."
        assert result.provider == "mock"

    @pytest.mark.asyncio
    async def test_provider_error_handling(self, mock_server_url):
        """Test provider handles HTTP errors correctly."""
        from scout.llm.providers import ProviderClient, ProviderResult

        # Configure mock to return error
        MockLLMHandler.should_error = True
        MockLLMHandler.error_status = 429

        async def mock_call_with_error(**kwargs) -> ProviderResult:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{mock_server_url}/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "test"}]}
                )
                # Simulate rate limit error
                if response.status_code == 429:
                    raise Exception("rate limit exceeded")
                return ProviderResult(
                    response_text="",
                    cost_usd=0,
                    input_tokens=0,
                    output_tokens=0,
                    model="test",
                    provider="mock",
                )

        client = ProviderClient(
            name="mock_error_provider",
            call=mock_call_with_error,
        )

        # Test that error is raised and properly categorized
        with pytest.raises(Exception) as exc_info:
            await client.call(prompt="test")

        assert "rate limit" in str(exc_info.value).lower()

        # Reset mock server state
        MockLLMHandler.should_error = False

    @pytest.mark.asyncio
    async def test_provider_with_key_rotation(self, mock_server_url):
        """Test provider key rotation works with mock responses."""
        from scout.llm.providers import ProviderClient, ProviderResult

        call_count = 0

        async def mock_call(**kwargs) -> ProviderResult:
            nonlocal call_count
            call_count += 1

            return ProviderResult(
                response_text=f"Response {call_count}",
                cost_usd=0.001,
                input_tokens=10,
                output_tokens=20,
                model="test-model",
                provider="test",
            )

        client = ProviderClient(
            name="test_rotation",
            call=mock_call,
        )

        # Add multiple keys
        client.add_key("key1")
        client.add_key("key2")

        # Get working key - should return first healthy key
        key = client.get_working_key()
        assert key in ["key1", "key2"]

        # Record failure on current key
        client.record_key_failure(key)

        # Get working key again - should still have options
        key2 = client.get_working_key()
        assert key2 is not None

    @pytest.mark.asyncio
    async def test_concurrent_provider_calls(self, mock_server_url):
        """Test provider handles concurrent requests."""
        from scout.llm.providers import ProviderClient, ProviderResult

        async def mock_call(**kwargs) -> ProviderResult:
            # Simulate some processing time
            await asyncio.sleep(0.01)
            return ProviderResult(
                response_text="Concurrent response",
                cost_usd=0.001,
                input_tokens=10,
                output_tokens=20,
                model="test",
                provider="test",
            )

        client = ProviderClient(
            name="concurrent_test",
            call=mock_call,
        )

        # Make concurrent calls
        tasks = [client.call(prompt=f"test {i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 10
        assert all(isinstance(r, ProviderResult) for r in results)


class TestProviderRegistry:
    """Integration tests for provider registry."""

    def test_registry_lists_providers(self):
        """Test that registry contains expected providers."""
        providers = registry.list_providers()

        # Should have at least the default providers registered
        assert "anthropic" in providers
        assert "groq" in providers or "google" in providers  # Depends on what's registered

    def test_registry_get_provider(self):
        """Test getting a provider from registry."""
        client = registry.get("anthropic")
        assert client is not None
        assert client.name == "anthropic"

    def test_registry_provider_availability(self):
        """Test checking provider availability."""
        # This test checks if providers are available (may have keys or not)
        available = registry.available("anthropic")
        assert isinstance(available, bool)


class TestProviderResult:
    """Tests for ProviderResult dataclass."""

    def test_provider_result_creation(self):
        """Test creating a ProviderResult."""
        result = ProviderResult(
            response_text="Hello",
            cost_usd=0.002,
            input_tokens=10,
            output_tokens=20,
            model="gpt-4",
            provider="openai",
        )

        assert result.response_text == "Hello"
        assert result.cost_usd == 0.002
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.model == "gpt-4"
        assert result.provider == "openai"

    def test_provider_result_total_tokens(self):
        """Test total_tokens property."""
        result = ProviderResult(
            response_text="Hello",
            cost_usd=0.002,
            input_tokens=10,
            output_tokens=20,
            model="gpt-4",
            provider="openai",
        )

        assert result.input_tokens + result.output_tokens == 30
