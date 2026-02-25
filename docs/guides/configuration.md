# Configuration Guide

Scout Core provides extensive configuration options through `scout.config.defaults`. This guide covers all available settings and how to customize them.

## Configuration Overview

Scout uses a layered configuration system:

1. **Defaults** (`scout.config.defaults`) - Hard-coded safe defaults
2. **Environment Variables** - Override defaults via `SCOUT_*` vars
3. **Custom Config File** - YAML configuration for project-specific settings
4. **Programmatic** - Direct Python configuration

## Importing Configuration

```python
# Import specific values
from scout.config import (
    BUDGET_DEFAULT_HOURLY_BUDGET,
    RETRY_MAX_RETRIES,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
)

# Or import ScoutConfig for full control
from scout.app_config import ScoutConfig
```

## Configuration Categories

### Budget Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `BUDGET_COST_PER_MILLION_8B` | $0.20 | Cost per million tokens (8B models) |
| `BUDGET_COST_PER_MILLION_70B` | $0.90 | Cost per million tokens (70B models) |
| `BUDGET_TOKENS_PER_SMALL_FILE` | 500 | Estimated tokens for small files |
| `BUDGET_BRIEF_COST_PER_FILE` | $0.005 | Cost per file for brief operation |
| `BUDGET_TASK_NAV_ESTIMATED_COST` | $0.002 | Estimated cost for navigation |
| `BUDGET_DRAFT_COST_PER_FILE` | $0.0004 | Cost per file for draft operation |
| `BUDGET_RESERVATION_TIMEOUT_SECONDS` | 30 | Budget reservation timeout |
| `BUDGET_ALLOW_OVERAGE_PERCENT` | 10 | Allowed overage percentage |
| `BUDGET_DEFAULT_HOURLY_BUDGET` | $1.00 | Default hourly budget |
| `BUDGET_HARD_SAFETY_CAP` | $10.00 | Maximum allowed budget |
| `BUDGET_HOUR_SECONDS` | 3600 | Seconds per hour (for calculations) |
| `BUDGET_CASCADE_BUFFER_FACTOR` | 1.2 | 20% buffer for cascade estimation |

### Retry Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `RETRY_BASE_DELAY` | 1.0s | Base delay for exponential backoff |
| `RETRY_MAX_DELAY` | 30.0s | Maximum delay between retries |
| `RETRY_MAX_RETRIES` | 3 | Maximum number of retry attempts |
| `RETRY_JITTER_FACTOR` | 0.1 | Jitter factor (10% randomness) |

### Circuit Breaker Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | 5 | Failures before opening circuit |
| `CIRCUIT_BREAKER_SUCCESS_THRESHOLD` | 2 | Successes to close circuit |
| `CIRCUIT_BREAKER_TIMEOUT_SECONDS` | 30.0 | Time before trying HALF_OPEN |
| `CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS` | 3 | Max calls in HALF_OPEN state |
| `CIRCUIT_BREAKER_PROVIDER_COOLDOWN_SECONDS` | 300 | Provider-level cooldown (5 min) |

### Rate Limiter Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `RATELIMIT_DEFAULT_RPM` | 20 | Default requests per minute |
| `RATELIMIT_DEFAULT_RPD` | 50 | Default requests per day |
| `RATELIMIT_RPM_WINDOW_SECONDS` | 60 | RPM window size |
| `RATELIMIT_RPD_WINDOW_SECONDS` | 86400 | RPD window size (24 hours) |
| `RATELIMIT_DEEPSEEK_RPM` | 50 | DeepSeek RPM limit |
| `RATELIMIT_DEEPSEEK_TPM` | 2,000,000 | DeepSeek TPM limit |
| `RATELIMIT_MINIMAX_RPM` | 100 | MiniMax RPM limit |
| `RATELIMIT_MINIMAX_TPM` | 200,000 | MiniMax TPM limit |

### Hotspot Detection Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `HOTSPOT_WEIGHT_CHURN` | 0.4 | Weight for code churn |
| `HOTSPOT_WEIGHT_ERROR` | 0.4 | Weight for error frequency |
| `HOTSPOT_WEIGHT_IMPACT` | 0.2 | Weight for impact score |
| `HOTSPOT_DEFAULT_DAYS` | 30 | Analysis window (days) |
| `HOTSPOT_MAX_CHURN` | 10 | Maximum churn for normalization |
| `HOTSPOT_MAX_ERRORS` | 10 | Maximum errors for normalization |
| `HOTSPOT_MAX_IMPACT` | 50 | Maximum impact for normalization |
| `HOTSPOT_DEFAULT_LIMIT` | 10 | Default result limit |

### Trust/BM25 Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `BM25_BASE` | 50 | BM25 base score |
| `BM25_RANGE` | 30 | BM25 score range |
| `BM25_CLARITY_MAX` | 30 | Maximum clarity boost |
| `BM25_EXACT_BOOST` | 1.03 | Exact match boost factor |
| `BM25_CLASS_BOOST` | 1.01 | Class match boost factor |
| `SEARCH_CONFIDENCE_GAP_FACTOR` | 40 | Confidence gap multiplier |
| `SEARCH_CONFIDENCE_GAP_BASE` | 8 | Confidence gap base offset |
| `TRUST_DEFAULT_MIN_CONFIDENCE` | 70 | Default minimum confidence |
| `TRUST_DEFAULT_LEARNER_MIN_SAMPLES` | 10 | Min samples for trust learner |
| `TRUST_DEFAULT_LEARNER_ADJUSTMENT_RATE` | 0.05 | Learner adjustment rate |
| `TRUST_DEFAULT_LEARNER_CONFIDENCE_THRESHOLD` | 0.8 | Learner confidence threshold |
| `TRUST_DEFAULT_AUTO_REPAIR_ENABLED` | True | Auto-repair enabled by default |
| `TRUST_DEFAULT_AUTO_REPAIR_THRESHOLD` | 5 | Auto-repair threshold |

### Audit Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `AUDIT_DEFAULT_LEVEL` | "INFO" | Default audit log level |
| `AUDIT_DEFAULT_SAMPLE_RATE` | 0.1 | Sample rate for DEBUG events |
| `AUDIT_LOG_FILENAME` | "audit.jsonl" | Audit log filename |
| `TRUST_DB_FILENAME` | "trust.db" | Trust database filename |

### Timeout Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `TIMEOUT_CONNECT_DEFAULT` | 10.0s | Default connection timeout |
| `TIMEOUT_READ_DEFAULT` | 60.0s | Default read timeout |
| `TIMEOUT_DEEPSEEK_CONNECT` | 10.0s | DeepSeek connect timeout |
| `TIMEOUT_DEEPSEEK_READ` | 60.0s | DeepSeek read timeout |
| `TIMEOUT_MINIMAX_CONNECT` | 15.0s | MiniMax connect timeout |
| `TIMEOUT_MINIMAX_READ` | 90.0s | MiniMax read timeout |

### Token Estimation Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `TOKENS_PER_CHAR_ENGLISH` | 4 | English text ratio |
| `TOKENS_PER_CHAR_CODE` | 3 | Code text ratio |
| `TOKEN_ESTIMATOR_MESSAGE_OVERHEAD` | 4 | Overhead per message |
| `TOKEN_ESTIMATOR_SYSTEM_MESSAGE_OVERHEAD` | 50 | System message overhead |
| `TOKEN_ESTIMATOR_MODEL_DEFAULT` | "gpt-3.5-turbo" | Default model for estimation |

### Navigation Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `NAV_DEFAULT_CONFIDENCE` | 85 | Default confidence level |
| `NAV_FALLBACK_CONFIDENCE` | 90 | Fallback confidence |
| `NAV_INDEX_CONFIDENCE` | 70 | Index-based confidence |
| `NAV_COST_8B_ESTIMATE` | $0.0002 | 8B model navigation cost |
| `NAV_COST_70B_ESTIMATE` | $0.0009 | 70B model navigation cost |
| `NAV_FALLBACK_DURATION_MS` | 50 | Fallback duration |
| `NAV_CONTEXT_MAX_CHARS` | 2000 | Max context characters |
| `NAV_SEARCH_RESULT_LIMIT` | 20 | Max search results |
| `NAV_PYTHON_FILE_LIMIT` | 50 | Max Python files for context |
| `NAV_TOKEN_MIN` | 100 | Minimum estimated tokens |
| `NAV_TOKEN_MAX` | 5000 | Maximum estimated tokens |
| `NAV_TOKEN_CHAR_RATIO` | 4 | Characters per token |
| `TASK_HIGH_CONFIDENCE_THRESHOLD` | 0.9 | High confidence threshold |
| `TASK_LOW_CONFIDENCE_THRESHOLD` | 0.7 | Low confidence threshold |
| `NAV_ESTIMATED_COST` | $0.01 | Default navigation cost |
| `NAV_COST_BUFFER` | 1.2 | Cost estimation buffer |

### Execution & Plans Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `EXECUTOR_TIMEOUT_SECONDS` | 300 | Default step timeout |
| `EXECUTOR_ESTIMATED_COST` | $0.001 | Estimated cost per step |
| `EXECUTOR_MAX_BUDGET` | $0.10 | Default max budget for plan |
| `EXECUTOR_DEFAULT_MAX_RETRIES` | 2 | Default max retries |
| `EXECUTOR_DEFAULT_TIMEOUT_SECONDS` | 300 | Default execution timeout |
| `BATCH_MAX_DEPTH` | 3 | Maximum sub-batch depth |
| `BATCH_DEFAULT_TIMEOUT` | 600s | Default batch timeout |
| `PLAN_LOCK_TIMEOUT_SECONDS` | 30 | Lock acquisition timeout |
| `PLAN_STALE_LOCK_HOURS` | 1 | Stale lock threshold |
| `PLAN_CRITICAL_ACTION_TYPES` | ["deploy", "delete", "modify_production"] | Critical actions |
| `PLAN_ARCHIVE_DAYS` | 7 | Archive completed plans |
| `PLAN_DELETE_DAYS` | 30 | Delete archived plans |
| `PLAN_CACHE_DAYS` | 30 | Default cache duration |
| `STEP_DEFAULT_TIMEOUT_SECONDS` | 300 | Default step timeout |
| `STEP_DEFAULT_MAX_RETRIES` | 2 | Default step retries |

### Safety Guard Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `SAFETY_MAX_PATH_DEPTH` | 10 | Maximum path depth |
| `SAFETY_MAX_LIST_DEPTH` | 3 | Maximum list depth |
| `SAFETY_MAX_FILE_SIZE_KB` | 1024 | Maximum file size (KB) |
| `SAFETY_DEFAULT_COMMAND_TIMEOUT` | 30s | Default command timeout |
| `SAFETY_MAX_WAIT_SECONDS` | 60 | Maximum wait time |
| `SAFETY_DEFAULT_SLEEP` | 1 | Default sleep time |

### Web Step Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `WEBSTEP_DEFAULT_MAX_RETRIES` | 1 | Web step max retries |
| `WEBSTEP_DEFAULT_TIMEOUT_SECONDS` | 30 | Web step timeout |

### CLI Defaults

| Constant | Default | Description |
|----------|---------|-------------|
| `CLI_DISCOVERY_TIMEOUT` | 10s | CLI discovery timeout |

### File Operation Timeouts

| Constant | Default | Description |
|----------|---------|-------------|
| `FILE_READ_TIMEOUT` | 300s | File read timeout |
| `FILE_WRITE_TIMEOUT` | 120s | File write timeout |
| `FILE_DELETE_TIMEOUT` | 120s | File delete timeout |
| `FILE_EDIT_TIMEOUT` | 120s | File edit timeout |

## Overriding via Environment Variables

Set environment variables to override defaults:

```bash
# Budget settings
export SCOUT_BUDGET_HOURLY=5.00
export SCOUT_BUDGET_SAFETY_CAP=20.00

# Retry settings
export SCOUT_RETRY_MAX_RETRIES=5
export SCOUT_RETRY_BASE_DELAY=2.0

# Circuit breaker
export SCOUT_CIRCUIT_BREAKER_FAILURE_THRESHOLD=10

# Timeouts
export SCOUT_TIMEOUT_CONNECT=30
export SCOUT_TIMEOUT_READ=120
```

## Using Custom Configuration Files

### YAML Configuration

Create a `.scout/config.yaml` file:

```yaml
budget:
  hourly_budget: 5.00
  safety_cap: 20.00

retry:
  max_retries: 5
  base_delay: 2.0
  max_delay: 60.0

circuit_breaker:
  failure_threshold: 10
  success_threshold: 3
  timeout_seconds: 60.0

execution:
  max_budget: 0.50
  timeout_seconds: 600
```

### Config File Locations

Configuration files are loaded in order (later sources override earlier):

1. `~/.scout/config.yaml` - User-level config
2. `.scout/config.yaml` - Project-level config
3. Environment variables - Override everything

## Programmatic Configuration

### Using ScoutConfig

```python
from scout.app_config import ScoutConfig

# Create custom configuration
config = ScoutConfig(
    budget_hourly=5.00,
    budget_safety_cap=20.00,
    retry_max_retries=5,
    circuit_breaker_failure_threshold=10,
)

# Use in your application
from scout.llm.budget import BudgetService
budget_service = BudgetService(config=config)
```

### Accessing Config Values

```python
from scout.config import (
    BUDGET_DEFAULT_HOURLY_BUDGET,
    RETRY_MAX_RETRIES,
)

# Direct import
print(f"Default hourly budget: ${BUDGET_DEFAULT_HOURLY_BUDGET}")
print(f"Max retries: {RETRY_MAX_RETRIES}")
```

## Loading .env Files

Scout can automatically load environment variables from `.env` files:

```python
from scout.app_config import EnvLoader

# Load from default location (.env)
EnvLoader.load()

# Load from specific file
EnvLoader.load(path=Path("production.env"))
```

The `EnvLoader`:
- Uses `setdefault` - won't override existing env vars
- Supports quoted values
- Ignores comments

## Best Practices

### 1. Start with Defaults

```python
# Use defaults for most cases
from scout.config import BUDGET_DEFAULT_HOURLY_BUDGET
```

### 2. Override Only What's Needed

```python
# Only override specific values
from scout.app_config import ScoutConfig

config = ScoutConfig(
    budget_hourly=5.00  # Override just this one
)
```

### 3. Use Environment Variables for Deployment

```bash
# Production
export SCOUT_BUDGET_HOURLY=10.00

# Development
export SCOUT_BUDGET_HOURLY=1.00
```

### 4. Validate Configuration

```python
from scout.app_config import ScoutConfig

try:
    config = ScoutConfig(
        budget_hourly=100.00  # Exceeds safety cap!
    )
except ValueError as e:
    print(f"Invalid config: {e}")
```

## Related Documentation

- [scout.config module reference](../api/scout.config.md) - Full API documentation
- [ADR-003: Budget Service](../adr/ADR-003-budget-service-reservation-semantics.md) - Budget design
- [ADR-005: Circuit Breaker](../adr/ADR-005-circuit-breaker-pattern.md) - Circuit breaker design
- [ADR-010: Retry Mechanisms](../adr/ADR-010-retry-mechanisms.md) - Retry design
