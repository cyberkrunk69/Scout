"""Tests for token_estimator module."""

import pytest
from scout.token_estimator import estimate_tokens, TokenEstimator


class TestTokenEstimator:
    """Test token estimation functions."""

    def test_estimate_tokens_empty_string(self):
        """Test token estimation with empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        """Test token estimation with short text."""
        text = "hello world"
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_long_text(self):
        """Test token estimation with long text."""
        # Approximately 1000 words
        text = " ".join(["word"] * 1000)
        tokens = estimate_tokens(text)
        # Rough estimate: ~1.3 tokens per word for English
        assert tokens > 1000

    def test_estimate_tokens_code_mode(self):
        """Test token estimation in code mode."""
        text = "def hello(): return 'world'"
        tokens = estimate_tokens(text, is_code=True)
        assert tokens > 0

    def test_estimate_tokens_handles_special_chars(self):
        """Test token estimation handles special characters."""
        text = "Hello 世界 🌍 123 !@#$%"
        tokens = estimate_tokens(text)
        assert tokens > 0

    def test_estimate_tokens_consistency(self):
        """Test that same text produces same token estimate."""
        text = "Consistent token estimation"
        tokens1 = estimate_tokens(text)
        tokens2 = estimate_tokens(text)
        assert tokens1 == tokens2

    def test_token_estimator_class(self):
        """Test TokenEstimator class."""
        estimator = TokenEstimator()
        assert estimator is not None
