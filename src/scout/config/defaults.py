"""Default configuration values for Scout.

This module centralizes all hard-coded magic numbers (defaults, weights,
cost estimates, timeouts, etc.) into a single location. All modules
should import these constants instead of hard-coding values.

Usage:
    from scout.config import (
        BUDGET_COST_PER_MILLION_8B,
        RETRY_BASE_DELAY,
        HOTSPOT_WEIGHT_CHURN,
    )
"""

from __future__ import annotations

# =============================================================================
# Budget Defaults
# =============================================================================

# Token cost estimates (per million tokens)
BUDGET_COST_PER_MILLION_8B = 0.20
BUDGET_COST_PER_MILLION_70B = 0.90

# Token estimates for cost prediction
BUDGET_TOKENS_PER_SMALL_FILE = 500

# Operation cost estimates
BUDGET_BRIEF_COST_PER_FILE = 0.005
BUDGET_TASK_NAV_ESTIMATED_COST = 0.002
BUDGET_DRAFT_COST_PER_FILE = 0.0004

# Budget reservation settings
BUDGET_RESERVATION_TIMEOUT_SECONDS = 30
BUDGET_ALLOW_OVERAGE_PERCENT = 10

# Budget limits
BUDGET_DEFAULT_HOURLY_BUDGET = 1.0
BUDGET_HARD_SAFETY_CAP = 10.0

# Time windows
BUDGET_HOUR_SECONDS = 3600

# Cost calculation factors
BUDGET_CASCADE_BUFFER_FACTOR = 1.2  # 20% buffer for cascade cost estimation


# =============================================================================
# Retry Defaults
# =============================================================================

RETRY_BASE_DELAY = 1.0  # seconds
RETRY_MAX_DELAY = 30.0  # seconds
RETRY_MAX_RETRIES = 3
RETRY_JITTER_FACTOR = 0.1


# =============================================================================
# Circuit Breaker Defaults
# =============================================================================

CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_COOLDOWN_SECONDS = 300


# =============================================================================
# Rate Limiter Defaults
# =============================================================================

# OpenRouter free tier defaults
RATELIMIT_DEFAULT_RPM = 20
RATELIMIT_DEFAULT_RPD = 50
RATELIMIT_RPM_WINDOW_SECONDS = 60
RATELIMIT_RPD_WINDOW_SECONDS = 86400  # 24 hours

# Provider-specific limits
RATELIMIT_DEEPSEEK_RPM = 50
RATELIMIT_DEEPSEEK_TPM = 2_000_000
RATELIMIT_MINIMAX_RPM = 100
RATELIMIT_MINIMAX_TPM = 200_000


# =============================================================================
# Hotspot Detection Defaults
# =============================================================================

# Weight configuration for hotspot score calculation
HOTSPOT_WEIGHT_CHURN = 0.4
HOTSPOT_WEIGHT_ERROR = 0.4
HOTSPOT_WEIGHT_IMPACT = 0.2

# Analysis window
HOTSPOT_DEFAULT_DAYS = 30

# Normalization thresholds
HOTSPOT_MAX_CHURN = 10
HOTSPOT_MAX_ERRORS = 10
HOTSPOT_MAX_IMPACT = 50

# Output limits
HOTSPOT_DEFAULT_LIMIT = 10


# =============================================================================
# Trust/BM25 Defaults
# =============================================================================

# BM25 constants
BM25_BASE = 50
BM25_RANGE = 30
BM25_CLARITY_MAX = 30
BM25_EXACT_BOOST = 1.03
BM25_CLASS_BOOST = 1.01

# Trust scoring defaults
TRUST_DEFAULT_MIN_CONFIDENCE = 70

# Learner defaults
TRUST_DEFAULT_LEARNER_MIN_SAMPLES = 10
TRUST_DEFAULT_LEARNER_ADJUSTMENT_RATE = 0.05
TRUST_DEFAULT_LEARNER_CONFIDENCE_THRESHOLD = 0.8

# Auto-repair defaults
TRUST_DEFAULT_AUTO_REPAIR_ENABLED = True
TRUST_DEFAULT_AUTO_REPAIR_THRESHOLD = 5

# Audit defaults
AUDIT_DEFAULT_LEVEL = "INFO"  # "DEBUG", "INFO", "WARN", "ERROR"
AUDIT_DEFAULT_SAMPLE_RATE = 0.1  # 10% sample for DEBUG events


# =============================================================================
# File Names
# =============================================================================

AUDIT_LOG_FILENAME = "audit.jsonl"
TRUST_DB_FILENAME = "trust.db"


# =============================================================================
# Timeout Defaults
# =============================================================================

# Default timeouts (seconds)
TIMEOUT_CONNECT_DEFAULT = 10.0
TIMEOUT_READ_DEFAULT = 60.0

# Provider-specific overrides
TIMEOUT_DEEPSEEK_CONNECT = 10.0
TIMEOUT_DEEPSEEK_READ = 60.0
TIMEOUT_MINIMAX_CONNECT = 15.0
TIMEOUT_MINIMAX_READ = 90.0


# =============================================================================
# Token Estimation Defaults
# =============================================================================

# Character-to-token ratios (tokens ≈ chars / ratio)
TOKENS_PER_CHAR_ENGLISH = 4
TOKENS_PER_CHAR_CODE = 3

# Message overhead (tokens per message)
TOKEN_ESTIMATOR_MESSAGE_OVERHEAD = 4
TOKEN_ESTIMATOR_SYSTEM_MESSAGE_OVERHEAD = 50

# Default model for token estimation
TOKEN_ESTIMATOR_MODEL_DEFAULT = "gpt-3.5-turbo"


# =============================================================================
# Sliding Window Rate Limiter Defaults
# =============================================================================

SLIDING_WINDOW_REQUESTS_PER_MINUTE = 60
SLIDING_WINDOW_TOKENS_PER_MINUTE = 100_000  # 100k TPM default
