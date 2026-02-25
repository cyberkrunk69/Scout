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

# General Circuit Breaker (for operations like batch processing)
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2
CIRCUIT_BREAKER_TIMEOUT_SECONDS = 30.0  # Time in OPEN state before HALF_OPEN
CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS = 3

# Provider-level Circuit Breaker (for LLM providers - longer cooldown)
# This is used for provider-wide failures where we want longer recovery time
CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS = 300  # 5 minutes - for expensive API calls

# Backward compatibility alias (deprecated - use CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS)
CIRCUIT_BREAKER_COOLDOWN_SECONDS = CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS


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


# =============================================================================
# Navigation Defaults
# =============================================================================

# Default confidence levels for navigation suggestions
NAV_DEFAULT_CONFIDENCE = 85  # Default confidence when LLM doesn't provide one
NAV_FALLBACK_CONFIDENCE = 90  # Fallback confidence for heuristic results
NAV_INDEX_CONFIDENCE = 70  # Confidence for index-based suggestions (lower than LLM)

# Cost estimates for navigation
NAV_COST_8B_ESTIMATE = 0.0002  # Estimated cost for 8B model navigation
NAV_COST_70B_ESTIMATE = 0.0009  # Estimated cost for 70B model navigation

# Navigation fallback duration (milliseconds)
NAV_FALLBACK_DURATION_MS = 50  # Estimated duration for heuristic fallback

# Navigation context limits
NAV_CONTEXT_MAX_CHARS = 2000  # Maximum characters for context in nav prompts
NAV_SEARCH_RESULT_LIMIT = 20  # Maximum search results from index

# File listing limits
NAV_PYTHON_FILE_LIMIT = 50  # Maximum Python files to list for context

# Token estimation for files
NAV_TOKEN_MIN = 100  # Minimum estimated tokens for a file
NAV_TOKEN_MAX = 5000  # Maximum estimated tokens for a file
NAV_TOKEN_CHAR_RATIO = 4  # Characters per token estimate

# Task routing confidence thresholds (0.0-1.0)
TASK_HIGH_CONFIDENCE_THRESHOLD = 0.9
TASK_LOW_CONFIDENCE_THRESHOLD = 0.7


# =============================================================================
# Execution & Plans Defaults
# =============================================================================

# Executor defaults
EXECUTOR_TIMEOUT_SECONDS = 300  # Default timeout for step execution
EXECUTOR_ESTIMATED_COST = 0.001  # Rough estimate per step
EXECUTOR_MAX_BUDGET = 0.10  # Default max budget for a plan
EXECUTOR_DEFAULT_MAX_RETRIES = 2
EXECUTOR_DEFAULT_TIMEOUT_SECONDS = 300

# Batch processing defaults
BATCH_MAX_DEPTH = 3  # Maximum nesting depth for sub-batches
BATCH_DEFAULT_TIMEOUT = 600  # Default timeout for batch execution (seconds)

# Plan state defaults
PLAN_LOCK_TIMEOUT_SECONDS = 30  # Lock acquisition timeout
PLAN_STALE_LOCK_HOURS = 1  # Stale lock threshold
PLAN_CRITICAL_ACTION_TYPES = ["deploy", "delete", "modify_production"]  # High-risk actions

# Plan cleanup defaults
PLAN_ARCHIVE_DAYS = 7  # Archive completed plans older than this
PLAN_DELETE_DAYS = 30  # Delete archived plans older than this
PLAN_CACHE_DAYS = 30  # Default cache duration for plan store

# Step execution defaults
STEP_DEFAULT_TIMEOUT_SECONDS = 300
STEP_DEFAULT_MAX_RETRIES = 2

# Safety guard defaults
SAFETY_MAX_PATH_DEPTH = 10  # Maximum path depth for check_depth
SAFETY_MAX_LIST_DEPTH = 3  # Default max_depth for scout_list
SAFETY_MAX_FILE_SIZE_KB = 1024  # Maximum file size in KB for scout_read_file
SAFETY_DEFAULT_COMMAND_TIMEOUT = 30  # Default timeout for scout_command (seconds)
SAFETY_MAX_WAIT_SECONDS = 60  # Maximum wait time for scout_wait

# Web step defaults
WEBSTEP_DEFAULT_MAX_RETRIES = 1
WEBSTEP_DEFAULT_TIMEOUT_SECONDS = 30

# LLM prose parser defaults
LLM_PARSER_DEFAULT_MAX_RETRIES = 3


# =============================================================================
# VS Code Extension Parser Defaults
# =============================================================================

# VS Code storage paths
VSCODE_EXTENSIONS_PATH = ".vscode/extensions"
VSCODE_GLOBAL_STORAGE = "globalStorage"
VSCODE_WORKSPACE_STORAGE = "workspaceStorage"

# Parser retry defaults
PARSER_MAX_RETRIES = 3
PARSER_RETRY_DELAY = 0.5  # seconds

# Supported agents
SUPPORTED_AGENTS = ["copilot", "cline", "continue"]

# Agent identifiers
AGENT_COPILOT = "copilot"
AGENT_CLINE = "cline"
AGENT_CONTINUE = "continue"

# Copilot storage keys
COPILOT_CHAT_SESSION_INDEX = "chat.ChatSessionStore.index"
COPILOT_CHAT_SESSIONS_DIR = "chatSessions"

# Cline storage keys
CLINE_EXTENSION_ID = "saoudrizwan.claude-dev"
CLINE_API_CONVERSATION_DIR = "tasks"
CLINE_API_CONVERSATION_FILE = "api_conversation_history.json"
CLINE_FALLBACK_FILE = "state.json"

# Continue storage keys
CONTINUE_CONFIG_DIR = ".continue"
CONTINUE_SESSIONS_FILE = "sessions.json"
CONTINUE_HISTORY_DB = "history.db"
