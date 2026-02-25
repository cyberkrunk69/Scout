"""Unified Cost Calculator for LLM Router.

Provides:
- MODEL_COSTS: All model pricing (free + paid)
- TIER_MODELS: Model assignments by tier
- TASK_CONFIGS: Task type configurations
- calculate_cost(): Compute cost from token counts
- get_provider_for_model(): Get provider name for a model
- is_free_model(): Check if model is free
"""

import os

LLM_MODE = os.environ.get("SCOUT_LLM_MODE", "auto").lower()

# All models - free (OpenRouter) + paid
MODEL_COSTS = {
    # === OPENROUTER FREE MODELS ===
    # Fast/Simple tier
    "deepseek/deepseek-r1:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 164000,
        "capabilities": ["reasoning", "coding"]
    },
    "stepfun/step-3.5-flash:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 256000,
        "capabilities": ["reasoning", "speed"]
    },
    "openrouter/aurora-alpha:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 128000,
        "capabilities": ["coding", "agentic"]
    },
    "google/gemini-2.5-flash-lite:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 1050000,
        "capabilities": ["speed", "low_latency"]
    },
    "meta-llama/llama-4-maverick:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 256000,
        "capabilities": ["multimodal", "reasoning"]
    },
    
    # Medium/Reasoning tier
    "openrouter/pony-alpha:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 200000,
        "capabilities": ["agentic", "tool_calling"],
        "note": "Feb 2026 - agent-optimized"
    },
    "gpt-oss/qwen3-235b-a22b-thinking:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 262000,
        "capabilities": ["reasoning", "math", "tool_use"]
    },
    "nvidia/nemotron-3-nano-30b-a3b:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 256000,
        "capabilities": ["agentic", "efficiency"]
    },
    "upstage/solar-pro-3:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 128000,
        "capabilities": ["reasoning"]
    },
    
    # Large/Complex tier
    "meta-llama/llama-4-scout:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 512000,
        "capabilities": ["extended_context", "reasoning"]
    },
    "moonshotai/kimi-vl-a3b-thinking:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 131000,
        "capabilities": ["visual_reasoning", "multimodal"]
    },
    "gpt-oss/gpt-oss-120b:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 131000,
        "capabilities": ["reasoning", "agentic"]
    },
    
    # Tool-Calling tier
    "arcee-ai/trinity-large-preview:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 131000,
        "capabilities": ["tool_calling", "agentic"]
    },
    "arcee-ai/trinity-mini:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 131000,
        "capabilities": ["tool_calling", "reasoning"]
    },
    "qwen/qwen3-vl-235b-a22b-thinking:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 131000,
        "capabilities": ["multimodal", "visual_coding"]
    },
    
    # Multimodal tier
    "nvidia/nemotron-nano-2-vl:free": {
        "provider": "openrouter",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 128000,
        "capabilities": ["multimodal", "video", "document_ai"]
    },
    
    # === PAID MODELS ===
    "MiniMax-M2.5": {
        "provider": "minimax",
        "cost_per_1k_input": 0.00030,
        "cost_per_1k_output": 0.00120,
        "context": 197000,
        "capabilities": ["reasoning", "coding"]
    },
    "llama-3.1-8b-instant": {
        "provider": "groq",
        "cost_per_1k_input": 0.00005,
        "cost_per_1k_output": 0.00008,
        "context": 8192,
        "capabilities": ["fast", "routing"]
    },
    # Google Gemini models
    "gemini-2.0-flash": {
        "provider": "google",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 1000000,
        "capabilities": ["speed", "free"]
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "cost_per_1k_input": 0.0,
        "cost_per_1k_output": 0.0,
        "context": 1000000,
        "capabilities": ["speed", "free"]
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "cost_per_1k_input": 0.00125,
        "cost_per_1k_output": 0.005,
        "context": 2000000,
        "capabilities": ["reasoning", "quality"]
    },
}

# Tier mappings - which models in which tier
TIER_MODELS = {
    "fast": [
        "llama-3.1-8b-instant",  # groq - primary (smallest/cheapest)
        "deepseek/deepseek-r1:free",        # openrouter fallback
        "stepfun/step-3.5-flash:free",      # openrouter fallback
        "openrouter/aurora-alpha:free",      # openrouter fallback
        "google/gemini-2.5-flash-lite:free", # openrouter fallback
        "meta-llama/llama-4-maverick:free",  # openrouter fallback
    ],
    "medium": [
        "openrouter/pony-alpha:free",
        "gpt-oss/qwen3-235b-a22b-thinking:free",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "upstage/solar-pro-3:free",
    ],
    "large": [
        "meta-llama/llama-4-scout:free",
        "moonshotai/kimi-vl-a3b-thinking:free",
        "gpt-oss/gpt-oss-120b:free",
        "MiniMax-M2.5",  # paid fallback
    ],
    "tool-calling": [
        "arcee-ai/trinity-large-preview:free",
        "arcee-ai/trinity-mini:free",
        "qwen/qwen3-vl-235b-a22b-thinking:free",
    ],
    "multimodal": [
        "meta-llama/llama-4-maverick:free",
        "moonshotai/kimi-vl-a3b-thinking:free",
        "nvidia/nemotron-nano-2-vl:free",
    ],
}

# Full TASK_CONFIGS with all fields preserved
TASK_CONFIGS = {
    "simple": {"tier": "fast", "max_iterations": 10, "verify": False, "escalation_allowed": True},
    "plan": {"tier": "medium", "max_iterations": 5, "verify": True, "escalation_allowed": True},
    "synthesis": {"tier": "large", "max_iterations": 1, "verify": "auto", "escalation_allowed": False},
    "verification": {"tier": "large", "max_iterations": 1, "verify": False, "escalation_allowed": False},
    "tool_calling": {"tier": "tool-calling", "max_iterations": 3, "verify": False, "escalation_allowed": True},
    "multimodal": {"tier": "multimodal", "max_iterations": 2, "verify": False, "escalation_allowed": True},
}


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Look up model cost from MODEL_COSTS and compute total."""
    config = MODEL_COSTS.get(model)
    if not config:
        raise ValueError(f"Model {model} not found in MODEL_COSTS")
    return (input_tokens / 1000) * config["cost_per_1k_input"] + \
           (output_tokens / 1000) * config["cost_per_1k_output"]


def get_provider_for_model(model: str) -> str:
    """Get provider name for a given model."""
    config = MODEL_COSTS.get(model)
    if not config:
        raise ValueError(f"Model {model} not in MODEL_COSTS")
    return config["provider"]


def get_model_context_limit(model: str) -> int:
    """Get context window limit for a model."""
    config = MODEL_COSTS.get(model, {})
    return config.get("context", 128000)  # default to 128k


def is_free_model(model: str) -> bool:
    """Check if model is free (cost = 0)."""
    config = MODEL_COSTS.get(model, {})
    return config.get("cost_per_1k_input", -1) == 0.0
