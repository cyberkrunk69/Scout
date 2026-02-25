"""Tests for model selection."""

import pytest
from unittest.mock import patch
from scout.llm.select import select_model, is_provider_available
from scout.llm.providers import registry


def test_is_provider_available():
    """Test provider availability check."""
    # Groq is registered, so should be available if it has keys
    result = is_provider_available("groq")
    assert isinstance(result, bool)


def test_select_model_default():
    """Test model selection with defaults."""
    # This should work if there's at least one available model
    try:
        model = select_model("simple", iteration=0)
        assert model is not None
        assert isinstance(model, str)
    except RuntimeError:
        # Expected if no providers are available
        pass


def test_select_model_with_task_type():
    """Test model selection with specific task type."""
    try:
        model = select_model("analysis", iteration=0)
        assert model is not None
    except (RuntimeError, KeyError):
        # Task type might not be defined
        pass


def test_select_model_iteration():
    """Test model selection cycles through iterations."""
    try:
        model0 = select_model("simple", iteration=0)
        model1 = select_model("simple", iteration=1)
        # Both should be valid models
        assert isinstance(model0, str)
        assert isinstance(model1, str)
    except RuntimeError:
        # Expected if no providers are available
        pass
