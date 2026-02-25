"""LLM Router — Thin dispatcher using ProviderRegistry.

Routes LLM calls based on SCOUT_LLM_MODE:
- free = OpenRouter free models only
- paid = MiniMax only
- auto = OpenRouter free first, MiniMax fallback

Uses the new ProviderRegistry system for multi-key support.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    """Rich LLM response with full metadata."""
    content: str
    cost_usd: float
    model: str
    provider: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


from scout.llm.cost import LLM_MODE, MODEL_COSTS, TASK_CONFIGS
from scout.llm.select import select_model
from scout.llm.providers import (
    registry,
    ProviderResult,
    is_permanent_error,
    PERMANENT_ERROR_PATTERNS,
    get_circuit_breaker,
    is_provider_available,
)
from scout.llm.cost import get_provider_for_model, is_free_model
from scout.llm.ratelimit import rate_limiter

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY")
MINIMAX_KEY = os.environ.get("MINIMAX_API_KEY")

DEEPSEEK_RPM = 50

_deepseek_calls = 0
_deepseek_reset = 0.0

ESCALATION_QUALITY_PLATEAU_COUNT = int(os.environ.get("ESCALATION_QUALITY_PLATEAU_COUNT", "3"))
ESCALATION_MIN_COST_SAVINGS = float(os.environ.get("ESCALATION_MIN_COST_SAVINGS", "0.10"))
ESCALATION_EARLY_FOR_CRITICAL = os.environ.get("ESCALATION_EARLY_FOR_CRITICAL", "true").lower() == "true"

CRITICAL_KEYWORDS = {"security", "auth", "encryption", "password", "key", "token", "safe", "critical", "production"}


def get_tier_for_task(task_type: str) -> str:
    """Get the default tier for a task type."""
    return TASK_CONFIGS.get(task_type, TASK_CONFIGS.get("simple", {})).get("tier", "fast")


def get_tier_for_model(model: str) -> Optional[str]:
    """Find which tier a model belongs to."""
    from scout.llm.cost import TIER_MODELS
    for tier, models in TIER_MODELS.items():
        if model in models:
            return tier
    return None


def get_next_tier(current_tier: str) -> Optional[str]:
    """Get the next tier up in the hierarchy."""
    tier_order = ["fast", "medium", "large"]
    try:
        current_idx = tier_order.index(current_tier)
        if current_idx < len(tier_order) - 1:
            return tier_order[current_idx + 1]
    except ValueError:
        pass
    return None


def get_router_status() -> dict:
    """Get current router status including key health."""
    status = {
        "mode": LLM_MODE,
        "providers": {},
        "rate_limiter": {},
    }
    
    for name in registry.list_providers():
        try:
            provider = registry.get(name)
            keys_info = []
            for k in provider.keys:
                keys_info.append({
                    "healthy": k.is_healthy(),
                    "failures": k.failures,
                    "permanently_failed": k.permanently_failed,
                    "cooldown_until": k.cooldown_until,
                })
            status["providers"][name] = {
                "available": provider.has_keys(),
                "keys": keys_info,
            }
        except KeyError:
            pass
    
    status["rate_limiter"] = {
        "rpm_used": rate_limiter._rpm_count,
        "rpm_limit": rate_limiter.rpm_limit,
        "rpd_used": rate_limiter._rpd_count,
        "rpd_limit": rate_limiter.rpd_limit,
    }
    
    return status


async def call_llm(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    task_type: str = "simple",
    iteration: int = 0,
    model: Optional[str] = None,
) -> LLMResult:
    """
    Thin dispatcher using registry with multi-key rotation.
    
    Single entry point for all LLM calls.
    Returns LLMResult with full metadata.
    """
    if os.environ.get("SCOUT_USE_LEGACY_LLM", "").lower() == "true":
        from scout.llm import call_groq_async
        result = await call_groq_async(
            prompt=prompt,
            model=model or "llama-3.1-8b-instant",
            system=system,
            max_tokens=max_tokens,
            temperature=temperature if temperature else None,
        )
        return LLMResult(
            content=result.content,
            cost_usd=result.cost_usd,
            model=result.model,
            provider="groq",
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
    
    if model is None:
        model = select_model(task_type, iteration)
    
    provider_name = get_provider_for_model(model)
    
    if not is_provider_available(provider_name):
        breaker = get_circuit_breaker(provider_name)
        state = breaker.get_state()
        logger.warning(
            "provider_circuit_open",
            extra={
                "event": "provider_circuit_open",
                "provider": provider_name,
                "state": state["state"],
                "failure_count": state["failure_count"],
            }
        )
        fallback_providers = PROVIDER_FALLBACKS.get(provider_name, [])
        for fallback_provider in fallback_providers:
            if is_provider_available(fallback_provider):
                logger.info(f"Circuit breaker: skipping {provider_name}, trying fallback {fallback_provider}")
                return await _try_fallback_provider(
                    fallback_provider=fallback_provider,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    task_type=task_type,
                    iteration=iteration,
                    original_model=model,
                    original_error=RuntimeError(f"Circuit breaker open for {provider_name}"),
                )
        raise RuntimeError(f"Provider {provider_name} circuit is open and no fallbacks available")
    
    provider = registry.get(provider_name)
    
    available_keys = [k for k in provider.keys if k.is_healthy()]
    attempts = 0
    last_error = None
    
    for key_state in available_keys:
        if attempts >= provider.max_key_attempts:
            break
        attempts += 1
        
        try:
            result: ProviderResult = await provider.call(
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=key_state.key,
            )
            
            logger.info(
                "LLM routing: model=%s, provider=%s, key_health=%s, attempts=%d, cost=$%.6f",
                model,
                provider_name,
                key_state.is_healthy() if key_state else "N/A",
                attempts,
                result.cost_usd,
            )
            return LLMResult(
                content=result.response_text,
                cost_usd=result.cost_usd,
                model=model,
                provider=provider_name,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
            
        except Exception as e:
            last_error = e
            
            if is_permanent_error(e):
                logger.warning(
                    "LLM routing PERMANENT failure: model=%s, provider=%s, key=%s..., error=%s",
                    model,
                    provider_name,
                    key_state.key[:8] if key_state else "N/A",
                    str(e)[:100],
                )
                breaker = get_circuit_breaker(provider_name)
                is_quota = is_quota_error(e)
                breaker.record_failure(permanent=not is_quota)
                break
            
            provider.record_key_failure(key_state.key, permanent=False)
            logger.warning(
                "LLM routing key failure: model=%s, provider=%s, key=%s..., attempt=%d, error=%s",
                model,
                provider_name,
                key_state.key[:8] if key_state else "N/A",
                attempts,
                str(e)[:100],
            )
            continue
    
    if last_error and is_quota_error(last_error):
        logger.warning(
            "LLM routing quota exceeded for provider=%s, attempting fallback",
            provider_name,
        )
        _fallback_manager.record_failure(provider_name)
        
        fallback_providers = PROVIDER_FALLBACKS.get(provider_name, [])
        for fallback_provider in fallback_providers:
            if not _fallback_manager.is_available(fallback_provider):
                continue
            
            logger.warning(
                "provider_fallback",
                extra={
                    "event": "provider_fallback",
                    "original_provider": provider_name,
                    "fallback_provider": fallback_provider,
                    "reason": "quota_exceeded",
                    "free_fallback": False,
                    "task_type": task_type,
                    "model": model,
                }
            )
            
            try:
                return await _try_fallback_provider(
                    fallback_provider=fallback_provider,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    task_type=task_type,
                    iteration=iteration,
                    original_model=model,
                    original_error=last_error,
                )
            except Exception as fallback_error:
                logger.warning(
                    "LLM fallback failed for provider=%s: %s",
                    fallback_provider,
                    str(fallback_error)[:100],
                )
                logger.warning(
                    "provider_fallback_failed",
                    extra={
                        "event": "provider_fallback_failed",
                        "fallback_provider": fallback_provider,
                        "reason": str(fallback_error)[:100],
                    }
                )
                if is_quota_error(fallback_error):
                    _fallback_manager.record_failure(fallback_provider)
                continue
    
    current_tier = get_tier_for_model(model)
    next_tier = get_next_tier(current_tier) if current_tier else None
    
    if next_tier:
        logger.warning(
            "tier_escalation",
            extra={
                "event": "tier_escalation",
                "current_tier": current_tier,
                "next_tier": next_tier,
                "reason": "all_providers_exhausted",
                "original_provider": provider_name,
                "task_type": task_type,
            }
        )
        
        from scout.llm.cost import TIER_MODELS
        next_tier_models = TIER_MODELS.get(next_tier, [])
        
        for next_model in next_tier_models:
            try:
                next_provider = get_provider_for_model(next_model)
                logger.info(
                    "tier_escalation_attempt",
                    extra={
                        "event": "tier_escalation_attempt",
                        "tier": next_tier,
                        "model": next_model,
                        "provider": next_provider,
                    }
                )
                result = await call_llm(
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    task_type=task_type,
                    iteration=iteration,
                    model=next_model,
                )
                logger.info(
                    "tier_escalation_success",
                    extra={
                        "event": "tier_escalation_success",
                        "tier": next_tier,
                        "model": next_model,
                        "provider": next_provider,
                    }
                )
                return result
            except Exception as tier_error:
                logger.warning(
                    "tier_escalation_failed",
                    extra={
                        "event": "tier_escalation_failed",
                        "tier": next_tier,
                        "model": next_model,
                        "error": str(tier_error)[:100],
                    }
                )
                continue
        
        logger.warning(
            "tier_escalation_exhausted",
            extra={
                "event": "tier_escalation_exhausted",
                "tier": next_tier,
                "reason": "all_models_failed",
            }
        )
    
    _fallback_manager.record_failure(provider_name)
    raise last_error or RuntimeError(f"All keys exhausted for provider {provider_name}")


async def _try_fallback_provider(
    fallback_provider: str,
    prompt: str,
    system: Optional[str],
    max_tokens: int,
    temperature: float,
    task_type: str,
    iteration: int,
    original_model: str,
    original_error: Exception,
) -> LLMResult:
    """Try a fallback provider with an appropriate model."""
    from scout.llm.cost import TIER_MODELS, is_free_model
    
    fallback_model = None
    for tier_models in TIER_MODELS.values():
        for model in tier_models:
            try:
                model_provider = get_provider_for_model(model)
                if model_provider == fallback_provider:
                    fallback_model = model
                    break
            except ValueError:
                continue
        if fallback_model:
            break
    
    if not fallback_model:
        raise RuntimeError(f"No fallback model found for provider {fallback_provider}")
    
    free_fallback = is_free_model(fallback_model)
    
    logger.info(
        "LLM attempting fallback: provider=%s, model=%s (original error: %s)",
        fallback_provider,
        fallback_model,
        str(original_error)[:50],
    )
    
    logger.info(
        "provider_fallback_attempt",
        extra={
            "event": "provider_fallback_attempt",
            "fallback_provider": fallback_provider,
            "fallback_model": fallback_model,
            "original_model": original_model,
            "free_fallback": free_fallback,
            "task_type": task_type,
        }
    )
    
    provider = registry.get(fallback_provider)
    available_keys = [k for k in provider.keys if k.is_healthy()]
    
    for key_state in available_keys:
        try:
            result: ProviderResult = await provider.call(
                model=fallback_model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                api_key=key_state.key,
            )
            _fallback_manager.record_success(fallback_provider)
            logger.info(
                "LLM fallback succeeded: provider=%s, model=%s, cost=$%.6f",
                fallback_provider,
                fallback_model,
                result.cost_usd,
            )
            logger.info(
                "provider_fallback_success",
                extra={
                    "event": "provider_fallback_success",
                    "fallback_provider": fallback_provider,
                    "fallback_model": fallback_model,
                    "cost_usd": result.cost_usd,
                    "free_fallback": free_fallback,
                }
            )
            return LLMResult(
                content=result.response_text,
                cost_usd=result.cost_usd,
                model=fallback_model,
                provider=fallback_provider,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )
        except Exception as e:
            logger.warning(
                "LLM fallback key failed: provider=%s, key=%s..., error=%s",
                fallback_provider,
                key_state.key[:8] if key_state else "N/A",
                str(e)[:100],
            )
            continue
    
    raise RuntimeError(f"Fallback provider {fallback_provider} also failed")


async def call_minimax_async(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    model: Optional[str] = None,
) -> tuple[str, float]:
    """Deprecated: Use call_llm instead. Kept for backwards compatibility."""
    return await call_llm(
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        model=model,
    )


def escalation_judge(
    current_tier: str,
    iteration: int,
    cumulative_cost: float,
    quality_trend: list,
    task_criticality: str,
    task_config: dict,
) -> tuple[bool, str]:
    """
    Decide whether to escalate to a higher tier.
    """
    if not task_config.get("escalation_allowed", False):
        return False, "escalation_not_allowed"
    
    max_iters = task_config.get("max_iterations", 10)
    if iteration >= max_iters:
        next_tier = get_next_tier(current_tier)
        if next_tier:
            logger.info(f"[ESCALATION] tier={current_tier}->{next_tier} reason='max_iterations_reached_{iteration}' cumulative_cost=${cumulative_cost:.4f}")
            return True, f"max_iterations_reached_{iteration}"
        return False, "no_higher_tier_available"
    
    if len(quality_trend) >= ESCALATION_QUALITY_PLATEAU_COUNT:
        recent_scores = quality_trend[-ESCALATION_QUALITY_PLATEAU_COUNT:]
        score_range = max(recent_scores) - min(recent_scores)
        if score_range <= 5:
            next_tier = get_next_tier(current_tier)
            if next_tier:
                logger.info(f"[ESCALATION] tier={current_tier}->{next_tier} reason='quality_plateau_{recent_scores}' cumulative_cost=${cumulative_cost:.4f}")
                return True, f"quality_plateau_{recent_scores}"
    
    if ESCALATION_EARLY_FOR_CRITICAL and task_criticality == "critical":
        next_tier = get_next_tier(current_tier)
        if next_tier:
            logger.info(f"[ESCALATION] tier={current_tier}->{next_tier} reason='critical_task_early_escalation' cumulative_cost=${cumulative_cost:.4f}")
            return True, "critical_task_early_escalation"
    
    next_tier = get_next_tier(current_tier)
    if next_tier and cumulative_cost > ESCALATION_MIN_COST_SAVINGS:
        logger.info(f"[ESCALATION] tier={current_tier}->{next_tier} reason='cost_threshold_exceeded_{cumulative_cost:.2f}' cumulative_cost=${cumulative_cost:.4f}")
        return True, f"cost_threshold_exceeded_{cumulative_cost:.2f}"
    
    return False, "keep_current_tier"


from dataclasses import dataclass

MODEL_TIERS = {
    "fast": {
        "llama-3.1-8b-instant": {"provider": "groq", "cost_per_1k_input": 0.00005, "cost_per_1k_output": 0.00008},
    },
    "medium": {
        "deepseek-chat": {"provider": "deepseek", "cost_per_1k_input": 0.00014, "cost_per_1k_output": 0.00028},
    },
    "large": {
        "deepseek-chat": {"provider": "deepseek", "cost_per_1k_input": 0.00014, "cost_per_1k_output": 0.00028},
        "MiniMax-M2.5": {"provider": "minimax", "cost_per_1k_input": 0.00030, "cost_per_1k_output": 0.00120, "context": "200k"},
    },
}


async def call_deepseek_async(
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> tuple[str, float]:
    """Deprecated: Use call_llm instead. Kept for backward compatibility."""
    return await call_llm(
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        model="deepseek-chat",
    )


@dataclass
class LLMResponse:
    """Standardized LLM response with full metadata."""
    success: bool
    result: str
    error: Optional[str]
    metadata: dict


class TierFallbackManager:
    """Handles tier fallback strategy."""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.max_failures = 3
    
    def record_failure(self, provider: str):
        self.circuit_breakers[provider] = self.circuit_breakers.get(provider, 0) + 1
    
    def record_success(self, provider: str):
        self.circuit_breakers[provider] = 0
    
    def is_available(self, provider: str) -> bool:
        return self.circuit_breakers.get(provider, 0) < self.max_failures


_fallback_manager = TierFallbackManager()


def is_quota_error(error: Exception) -> bool:
    """Check if error is specifically a quota/credits exceeded error."""
    error_str = str(error).lower()
    quota_patterns = PERMANENT_ERROR_PATTERNS.get("quota_exceeded", [])
    for pattern in quota_patterns:
        if pattern.lower() in error_str:
            return True
    return False


PROVIDER_FALLBACKS = {
    "openrouter": ["minimax", "groq"],
    "groq": ["openrouter", "minimax"],
    "minimax": ["openrouter"],
}
