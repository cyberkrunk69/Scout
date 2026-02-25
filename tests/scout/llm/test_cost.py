"""Tests for cost calculation."""

import pytest

from scout.llm.cost import (
    calculate_cost,
    get_provider_for_model,
    get_model_context_limit,
    is_free_model,
    MODEL_COSTS,
    TIER_MODELS,
    TASK_CONFIGS,
)


def test_calculate_cost_known_model():
    """Test cost calculation for known models."""
    # Test llama-3.1-8b-instant (Groq 8B)
    cost = calculate_cost("llama-3.1-8b-instant", 1000, 500)
    # 0.00005 * 1 + 0.00008 * 0.5 = 0.00005 + 0.00004 = 0.00009
    assert cost == pytest.approx(0.00009, rel=1e-6)


def test_calculate_cost_free_model():
    """Test cost calculation for free models."""
    cost = calculate_cost("deepseek/deepseek-r1:free", 1000, 500)
    assert cost == 0.0


def test_calculate_cost_unknown_model():
    """Test cost calculation raises for unknown model."""
    with pytest.raises(ValueError, match="not found"):
        calculate_cost("unknown-model", 100, 50)


def test_get_provider_for_model():
    """Test getting provider for model."""
    assert get_provider_for_model("llama-3.1-8b-instant") == "groq"
    assert get_provider_for_model("MiniMax-M2.5") == "minimax"
    assert get_provider_for_model("gemini-2.5-pro") == "google"


def test_get_provider_for_model_unknown():
    """Test provider lookup raises for unknown model."""
    with pytest.raises(ValueError, match="not in MODEL_COSTS"):
        get_provider_for_model("unknown-model")


def test_get_model_context_limit():
    """Test context limit retrieval."""
    assert get_model_context_limit("llama-3.1-8b-instant") == 8192
    assert get_model_context_limit("gemini-2.5-pro") == 2000000


def test_get_model_context_limit_default():
    """Test default context limit for unknown models."""
    limit = get_model_context_limit("unknown-model")
    assert limit == 128000  # Default


def test_is_free_model():
    """Test free model detection."""
    assert is_free_model("deepseek/deepseek-r1:free") is True
    assert is_free_model("openrouter/pony-alpha:free") is True
    assert is_free_model("llama-3.1-8b-instant") is False
    assert is_free_model("MiniMax-M2.5") is False


def test_model_costs_structure():
    """Test MODEL_COSTS has expected structure."""
    # Check a few models
    assert "llama-3.1-8b-instant" in MODEL_COSTS
    assert "deepseek/deepseek-r1:free" in MODEL_COSTS
    
    # Check required fields
    model_info = MODEL_COSTS["llama-3.1-8b-instant"]
    assert "provider" in model_info
    assert "cost_per_1k_input" in model_info
    assert "cost_per_1k_output" in model_info
    assert "context" in model_info


def test_tier_models_structure():
    """Test TIER_MODELS has expected structure."""
    assert "fast" in TIER_MODELS
    assert "medium" in TIER_MODELS
    assert "large" in TIER_MODELS
    
    # Check fast tier contains expected models
    assert "llama-3.1-8b-instant" in TIER_MODELS["fast"]


def test_task_configs_structure():
    """Test TASK_CONFIGS has expected structure."""
    assert "simple" in TASK_CONFIGS
    assert "plan" in TASK_CONFIGS
    assert "synthesis" in TASK_CONFIGS
    
    # Check required fields
    simple_config = TASK_CONFIGS["simple"]
    assert "tier" in simple_config
    assert "max_iterations" in simple_config
    assert "verify" in simple_config
    assert "escalation_allowed" in simple_config
