"""Unified Model Selector for LLM Router.

Provides:
- is_provider_available(): Check if provider has working keys
- select_model(): Select model based on task, tier, mode, and availability
"""

from typing import Optional
from scout.llm.cost import MODEL_COSTS, TIER_MODELS, TASK_CONFIGS, LLM_MODE
from scout.llm.providers import registry


def is_provider_available(provider_name: str) -> bool:
    """Check if provider is usable (registered + has working keys)."""
    try:
        return registry.available(provider_name)
    except KeyError:
        return False


def select_model(
    task_type: str,
    iteration: int = 0,
    current_tier: Optional[str] = None,
    mode: Optional[str] = None,
) -> str:
    """Select model based on task, iteration, tier, mode, and provider availability."""
    mode = mode or LLM_MODE
    task_config = TASK_CONFIGS.get(task_type, TASK_CONFIGS["simple"])
    target_tier = current_tier or task_config.get("tier", "fast")
    
    candidate_models = [m for m in TIER_MODELS.get(target_tier, []) if m in MODEL_COSTS]
    
    if mode == "free":
        candidate_models = [m for m in candidate_models if MODEL_COSTS[m]["cost_per_1k_input"] == 0.0]
    elif mode == "paid":
        candidate_models = [m for m in candidate_models if MODEL_COSTS[m]["provider"] == "minimax"]
    
    candidate_models = [
        m for m in candidate_models 
        if is_provider_available(MODEL_COSTS[m]["provider"])
    ]
    
    if not candidate_models:
        raise RuntimeError(f"No available models for mode={mode}, tier={target_tier}")
    
    return candidate_models[iteration % len(candidate_models)]
