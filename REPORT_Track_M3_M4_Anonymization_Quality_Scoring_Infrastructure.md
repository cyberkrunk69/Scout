# Research Report: Anonymization & Quality Scoring Infrastructure

## Executive Summary

This report details the investigation into existing `scout-core` and `dataclaw` codebases to identify reusable components for building the Anonymization (M3) and Quality Scoring (M4) tracks. The analysis covers existing infrastructure, reuse opportunities, identified gaps, and proposed implementation approaches.

---

## 1. Existing Anonymization Infrastructure

### 1.1 dataclaw/parser.py - AnonymizerWrapper

**File:** `/Users/vivariumenv1/GITHUBS/dataclaw/dataclaw/parser.py`

The `AnonymizerWrapper` class wraps Scout's `AnonymizerTool` and provides a unified interface for text and path anonymization:

```19:66:dataclaw/parser.py
class AnonymizerWrapper:
    """Wrapper around Scout AnonymizerTool that provides the same interface as the old AnonymizerWrapper."""

    def __init__(self, extra_usernames: list[str] | None = None):
        self._tool = AnonymizerTool()
        # Always include the current system username
        self._current_username = os.path.basename(os.path.expanduser("~"))
        self._extra_usernames = extra_usernames or []

    def _get_all_usernames(self) -> list[str]:
        """Combine current username with extra usernames (deduplicated)."""
        usernames = set(self._extra_usernames)
        if self._current_username:
            usernames.add(self._current_username)
        return list(usernames)

    def text(self, content: str) -> str:
        if not content:
            return content
        usernames = self._get_all_usernames()
        try:
            result = self._tool.run({
                "mode": "text",
                "data": content,
                "extra_usernames": usernames,
            })
            return result.get("result") or content
        except Exception as e:
            logging.getLogger(__name__).warning(f"Anonymizer text() failed: {e}, returning original")
            return content

    def path(self, file_path: str) -> str:
        if not file_path:
            return file_path
        usernames = self._get_all_usernames()
        try:
            result = self._tool.run({
                "mode": "path",
                "data": file_path,
                "extra_usernames": usernames,
            })
            return result.get("result") or file_path
        except Exception as e:
            logging.getLogger(__name__).warning(f"Anonymizer path() failed: {e}, returning original")
            return file_path
```

**Key observations:**
- Handles usernames via `extra_usernames` parameter
- Graceful fallback to original content on error
- Supports both text and path modes
- There's also a `PassthroughAnonymizer` for when raw data is needed (e.g., search indexing)

### 1.2 dataclaw/secrets.py - Pattern Detection

**File:** `/Users/vivariumenv1/GITHUBS/dataclaw/dataclaw/secrets.py`

Comprehensive secret detection with 20+ pattern types:

```9:108:dataclaw/secrets.py
SECRET_PATTERNS = [
    # JWT tokens — full 3-segment form
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}")),

    # JWT tokens — partial (header only or header+partial payload, e.g. truncated)
    ("jwt_partial", re.compile(r"eyJ[A-Za-z0-9_-]{15,}")),

    # PostgreSQL/database connection strings with passwords
    ("db_url", re.compile(r"postgres(?:ql)?://[^:]+:[^@\s]+@[^\s\"'`]+")),

    # Anthropic API keys
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),

    # OpenAI API keys
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{40,}")),

    # ... many more patterns
]
```

**Key functions:**
- `scan_text(text)` - Returns list of findings with type, position, and matched text
- `redact_text(text)` - Returns (redacted_text, count) tuple
- `redact_custom_strings(text, strings)` - Redacts specific custom strings
- `redact_session(session, custom_strings)` - Redacts entire session dict

**Entropy detection:**
```136:152:dataclaw/secrets.py
def _shannon_entropy(s: str) -> float:
    """Higher values indicate more random-looking strings."""
    # ...

def _has_mixed_char_types(s: str) -> bool:
    """Check if string has a mix of uppercase, lowercase, and digits."""
    # ...
```

### 1.3 scout/tools/anonymizer.py - Core Anonymizer

**File:** `/Users/vivariumenv1/GITHUBS/scout/src/scout/tools/anonymizer.py`

The core `Anonymizer` class provides deterministic username hashing:

```39:128:src/scout/tools/anonymizer.py
def _hash_username(username: str) -> str:
    """Hash username using SHA256, return first 8 chars with prefix."""
    return "user_" + hashlib.sha256(username.encode()).hexdigest()[:8]

def anonymize_path(
    path: str,
    username: str,
    username_hash: str,
    home: str | None = None,
) -> str:
    """Strip a path to project-relative and hash the username."""
    # Replaces /Users/<username> with ~ or hashed user
    # ...

def anonymize_text(text: str, username: str, username_hash: str) -> str:
    """Replace username references with hash in text."""
    # Replaces /Users/<username>, /home/<username>, hyphen-encoded paths
    # ...
```

**Tool Interface:**
```208:319:src/scout/tools/anonymizer.py
class AnonymizerTool:
    """Public tool interface for PII anonymization with audit logging."""

    name: str = "anonymizer"
    description: str = (
        "Anonymize PII (usernames, paths) in text or file paths. "
        "Replaces user references with deterministic hashes."
    )

    def __init__(self, config: Optional[Any] = None):
        # Loads extra_usernames from config
        # ...

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # mode: "text" or "path"
        # data: string to anonymize
        # extra_usernames: optional list
        # Returns: {"result": anonymized_string, "metadata": {...}}
```

**Key observations:**
- Uses SHA256 hashing for usernames (first 8 chars)
- Handles multiple path formats: `/Users/<user>`, `/home/<user>`, hyphen-encoded
- Has audit logging built in
- Can be configured via Scout config

---

## 2. Configuration Systems

### 2.1 dataclaw/config.py

**File:** `/Users/vivariumenv1/GITHUBS/dataclaw/dataclaw/config.py`

Simple JSON-based config stored at `~/.dataclaw/config.json`:

```12:29:dataclaw/config.py
class DataClawConfig(TypedDict, total=False):
    """Expected shape of the config dict."""
    repo: str | None
    excluded_projects: list[str]
    redact_strings: list[str]
    redact_usernames: list[str]
    last_export: dict
    stage: str | None
    projects_confirmed: bool
    search: dict | None
```

### 2.2 scout/app_config.py

**File:** `/Users/vivariumenv1/GITHUBS/scout/src/scout/app_config.py`

YAML-based layered config with env var overrides:

```115:232:src/scout/app_config.py
DEFAULT_CONFIG = {
    "triggers": {...},
    "limits": {...},
    "models": {...},
    "quality_gates": {
        "enabled": False,
        "stages": ["unit", "integration"],
        "auto_gate": False,
    },
    "budget": {...},
    # ...
}

class ScoutConfig:
    """Layered configuration: user YAML + env vars + hard caps."""
    # Load order: DEFAULT < ~/.scout/config.yaml < .scout/config.yaml < env vars
```

**Config key paths:**
- `anonymizer.extra_usernames` - Already supported
- `quality_gates.*` - Already exists but disabled by default
- Custom sections can be added easily

---

## 3. Model Name Patterns

### 3.1 Existing Patterns in Codebase

Found in `src/scout/llm/pricing.py` and provider modules:

```src/scout/llm/pricing.py
# Anthropic models
"claude-3-5-sonnet-20241022"
"claude-3-5-sonnet"
"claude-3-opus"
"claude-3-sonnet"
"claude-3-haiku"

# Google models
"gemini-2.5-pro"
"gemini-2.5-flash"
"gemini-2.5-pro-002"
"gemini-2.5-flash-002"
"gemini-2.0-flash"
"gemini-3-flash-preview"

# OpenAI models (from patterns in codebase)
"gpt-4.*"
"gpt-4o*"
"gpt-4-turbo*"

# Others seen in logs
"llama-3.1-8b-instant"
"llama-3.3-70b-versatile"
"MiniMax-M2.5"
"deepseek-chat"
```

### 3.2 Recommended Model Name Regex Patterns

**Proposed patterns for model name anonymization:**

```python
MODEL_NAME_PATTERNS = [
    # Anthropic Claude
    (r"claude-3-[a-z]+-\d{8}"),  # claude-3-5-sonnet-20241022
    (r"claude-3-[a-z]+"),         # claude-3-5-sonnet
    (r"claude-[0-9]-[a-z]+"),     # claude-3-opus
    
    # OpenAI GPT
    (r"gpt-4o(?:-[\w-]+)?"),     # gpt-4o, gpt-4o-mini
    (r"gpt-4-turbo(?:-[\w-]+)?"),
    (r"gpt-4[\w-]*"),
    (r"gpt-3\.5-turbo"),
    
    # Google Gemini
    (r"gemini-2\.5-[\w-]+"),      # gemini-2.5-pro, gemini-2.5-flash
    (r"gemini-2\.0-[\w-]+"),
    (r"gemini-1\.5-[\w-]+"),
    
    # Meta Llama
    (r"llama-[0-9\.]+-[\w-]+"),   # llama-3.1-8b-instant
    
    # DeepSeek
    (r"deepseek-chat"),
    
    # MiniMax
    (r"MiniMax-[A-Za-z0-9\.]+"),
    
    # Generic fallback - any provider-model pattern
    (r"[a-z]+-[0-9\.]+[\w-]*"),
]
```

---

## 4. Secret Detection Reuse

### 4.1 Current Reuse in dataclaw

The `secrets.py` module is already used in the parsing pipeline:

```312:316:dataclaw/parser.py
def _redact_and_truncate(text: str, anonymizer: AnonymizerWrapper) -> str:
    """Redact secrets BEFORE truncating to avoid partial secret leaks."""
    text, _ = redact_text(text)
    return anonymizer.text(text[:MAX_TOOL_INPUT_LENGTH])

def _summarize_tool_input(tool_name: str | None, input_data: Any, anonymizer: AnonymizerWrapper) -> str:
    """Summarize tool input for export."""
    # Uses redact_text for bash commands and grep patterns
```

### 4.2 Reuse Recommendations

**Can be reused directly:**
1. `redact_text()` - Works on plain strings
2. `scan_text()` - Returns structured findings
3. `redact_session()` - Works on session dicts with messages
4. `redact_custom_strings()` - For custom redactions

**Extension needed:**
- Add model name patterns to `SECRET_PATTERNS`
- Create `redact_model_names()` function to complement existing functions

---

## 5. Path Normalization

### 5.1 Current Capabilities

The `AnonymizerTool` already handles:
- Replacing `/Users/<username>` with `~` or hashed user
- Replacing `/home/<username>` with hashed user
- Handling hyphen-encoded paths (Claude Code format)
- Detecting home directory automatically

### 5.2 Gaps Identified

**Gap: Project-relative paths**
- Currently only handles home directory replacement
- Does NOT extract project root from session metadata
- No built-in utility for making paths relative to a known project root

**Recommendation:**
- Extend `AnonymizerTool` to accept optional `project_root` parameter
- Create helper function to extract project root from session `cwd` metadata

---

## 6. Quality Scoring - Heuristic Signals

### 6.1 Available Session Metadata

From `dataclaw/parser.py` parsing:

```162:177:dataclaw/parser.py
metadata = {
    "session_id": filepath.stem,
    "cwd": None,
    "git_branch": None,
    "claude_version": None,
    "model": None,
    "start_time": None,
    "end_time": None,
}
stats = {
    "user_messages": 0,
    "assistant_messages": 0,
    "tool_uses": 0,
    "input_tokens": 0,
    "output_tokens": 0,
}
```

### 6.2 Computable Heuristic Signals

| Signal | Source | Computation |
|--------|--------|-------------|
| Message count | `stats` | `user_messages + assistant_messages` |
| Turn depth | `messages` | Count alternating user/assistant pairs |
| Token density | `stats` | `output_tokens / assistant_messages` |
| Tool usage rate | `stats` | `tool_uses / assistant_messages` |
| Code acceptance | `tool_uses` | Compare Edit vs Read counts |
| Error rate | `content` | Scan for error keywords |
| Session duration | `start_time`, `end_time` | Time difference |
| Thinking usage | `messages` | Presence of `thinking` field |

### 6.3 Existing Quality Infrastructure

The `scout/trust` module exists but focuses on LLM output verification, not session quality scoring:

```src/scout/trust/__init__.py
"""
Scout Trust Subsystem - Production-Grade Implementation

Philosophy-aligned:
- Right-Size: Deterministic verification before LLM
- DRY: Single source of truth
- Auditability: Every decision logged
- Spaceship: Self-learning via TrustLearner
- Cost as Feature: Every operation logged, every penny visible
"""
```

**Gap: No existing quality scorer for sessions**
- Trust is about validating LLM outputs
- Need new `QualityScorer` class for session quality

---

## 7. LLM-Based Scoring Infrastructure

### 7.1 call_llm Signature

**File:** `/Users/vivariumenv1/GITHUBS/scout/src/scout/llm/router.py`

```123:155:src/scout/llm/router.py
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
```

**Key observations:**
- Supports `system` prompt parameter
- Returns `LLMResult` with `content`, `cost_usd`, `model`, `provider`, `input_tokens`, `output_tokens`
- No built-in JSON mode - would need to request JSON in prompt

### 7.2 batch_process

**File:** `/Users/vivariumenv1/GITHUBS/scout/src/scout/llm/batch.py`

```10:40:src/scout/llm/batch.py
async def batch_process(
    prompts: List[str],
    func: Callable[[str], Awaitable[Any]],
    max_concurrent: int = 5,
    rate_limiter: Optional[RateLimiter] = None,
    return_exceptions: bool = False,
) -> List[Union[Any, Exception]]:
    """Process a list of prompts with concurrency control."""
```

**Can be reused for:**
- Batch scoring multiple sessions in parallel
- Configurable concurrency (default 5)

### 7.3 BudgetService

**File:** `/Users/vivariumenv1/GITHUBS/scout/src/scout/llm/budget.py`

```79:172:src/scout/llm/budget.py
class BudgetService:
    """Centralized budget management with reservation semantics."""

    def check(self, estimated_cost: float, operation: str) -> bool:
        """Check if budget allows operation without reserving."""

    def reserve(
        self,
        estimated_cost: float,
        operation: str,
        timeout_seconds: Optional[int] = None,
    ) -> BudgetReservation:
        """Reserve budget for operation."""
```

**Can be reused for:**
- Checking budget before LLM-based scoring
- Reserving budget for batch scoring operations
- Logging scoring costs to audit

---

## 8. Configuration & CLI Patterns

### 8.1 CLI Pattern (dataclaw)

**File:** `/Users/vivariumenv1/GITHUBS/dataclaw/dataclaw/cli.py`

```658:711:dataclaw/cli.py
def main() -> None:
    parser = argparse.ArgumentParser(description="DataClaw — Claude Code -> Hugging Face")
    parser.add_argument(
        "--claude-dir", 
        type=Path, 
        default=None,
        help="Path to Claude Code directory"
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("prep", help="Data prep — discover projects")
    sub.add_parser("status", help="Show current stage and next steps")
    cf = sub.add_parser("confirm", help="Scan for PII, summarize export")
    sub.add_parser("list", help="List all projects")

    cfg = sub.add_parser("config", help="View or set config")
    cfg.add_argument("--repo", type=str)
    cfg.add_argument("--exclude", type=str)
    cfg.add_argument("--redact", type=str)
    # ...
```

**Pattern to follow:**
1. Use `argparse` with subparsers
2. Each subcommand is a function
3. Config handled separately
4. JSON output for agent compatibility

### 8.2 Audit Logging

**File:** `/Users/vivariumenv1/GITHUBS/scout/src/scout/audit.py`

```37:79:src/scout/audit.py
EVENT_TYPES = frozenset({
    "nav",
    "brief",
    "cascade",
    "validation_fail",
    "budget",
    # ... many more
    # DataClaw enrichment events
    "enrich",
})

class AuditLog:
    def log(
        self,
        event_type: str,
        *,
        cost: Optional[float] = None,
        model: Optional[str] = None,
        # ...
    ) -> None:
```

**New event types to add:**
- `quality_score` - For scoring events
- `anonymize_model_name` - For model name redaction

---

## 9. Testing Patterns

### 9.1 dataclaw Tests

**File:** `/Users/vivariumenv1/GITHUBS/dataclaw/tests/test_anonymizer.py`

Uses pytest with mocking:

```35:49:dataclaw/tests/test_anonymizer.py
class TestAnonymizerWrapper:
    @patch("dataclaw.parser.AnonymizerTool")
    def test_text_calls_tool(self, mock_tool_class):
        """Text should call the underlying tool."""
        mock_tool = MagicMock()
        mock_tool.run.return_value = {"result": "anonymized_text"}
        mock_tool_class.return_value = mock_tool
        wrapper = AnonymizerWrapper()
        result = wrapper.text("original text")
        # assertions...
```

**Testing patterns:**
- Use `@patch` for external dependencies
- MagicMock for tool responses
- Test both success and error paths
- Test empty/null inputs

### 9.2 secrets Tests

**File:** `/Users/vivariumenv1/GITHUBS/dataclaw/tests/test_secrets.py`

Comprehensive pattern testing:

```86:104:dataclaw/tests/test_secrets.py
def test_jwt_token(self):
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
    findings = scan_text(jwt)
    assert any(f["type"] == "jwt" for f in findings)
```

---

## 10. Implementation Recommendations

### 10.1 Anonymization Track (M3)

#### Component 1: ModelNameAnonymizer

```python
class ModelNameAnonymizer:
    """Anonymize model names in session data."""
    
    DEFAULT_PATTERNS = [
        # Anthropic
        re.compile(r"claude-3-[a-z]+-\d{8}"),
        re.compile(r"claude-3-[a-z]+"),
        # OpenAI
        re.compile(r"gpt-4o[\w-]*"),
        re.compile(r"gpt-4-turbo[\w-]*"),
        # Google
        re.compile(r"gemini-[\d\.]+-[\w-]+"),
        # ... more patterns
    ]
    
    def __init__(self, patterns: list[re.Pattern] | None = None):
        self.patterns = patterns or self.DEFAULT_PATTERNS
    
    def anonymize_text(self, text: str) -> str:
        """Replace model names with generic placeholder."""
        # Replace matches with "[MODEL]"
    
    def anonymize_session(self, session: dict) -> dict:
        """Anonymize all model names in session."""
        # Handle metadata["model"]
        # Handle any model references in content
```

#### Component 2: Extended Secrets Scanner

```python
def scan_for_model_names(text: str) -> list[dict]:
    """Scan text for model name patterns."""
    # Uses ModelNameAnonymizer patterns
    # Returns findings similar to scan_text()

def redact_model_names(text: str) -> tuple[str, int]:
    """Redact model names from text."""
    # Uses scan_for_model_names
    # Returns (redacted_text, count)
```

#### Integration Points:

1. **Config extension** (`dataclaw/config.py`):
   ```python
   class DataClawConfig(TypedDict, total=False):
       # ... existing fields ...
       model_name_patterns: list[str]  # Custom regex patterns
       redact_model_names: bool          # Enable/disable
   ```

2. **Parser integration** (`dataclaw/parser.py`):
   ```python
   def parse_project_sessions(
       project_dir_name: str,
       anonymizer: AnonymizerWrapper,
       redact_model_names: bool = True,  # NEW PARAM
       # ...
   ):
   ```

3. **CLI extension** (`dataclaw/cli.py`):
   ```python
   cfg.add_argument("--redact-model-names", action="store_true")
   ```

### 10.2 Quality Scoring Track (M4)

#### Component 1: HeuristicQualityScorer

```python
@dataclass
class QualitySignals:
    """Computed quality signals from session."""
    message_count: int
    turn_depth: int
    token_density: float
    tool_usage_rate: float
    code_acceptance_ratio: float
    error_rate: float
    has_thinking: bool
    session_duration_seconds: float

class HeuristicQualityScorer:
    """Compute quality signals from session metadata."""
    
    def score(self, session: dict) -> QualitySignals:
        """Compute all quality signals."""
        # Parse messages, stats, timestamps
        # Return QualitySignals
    
    def compute_composite_score(self, signals: QualitySignals) -> float:
        """Compute weighted composite score 0-100."""
        # Weights:
        # - tool_usage_rate: 20%
        # - code_acceptance: 25%
        # - error_rate: 25% (inverted)
        # - turn_depth: 15%
        # - token_density: 15%
```

#### Component 2: LLMQualityScorer

```python
class LLMQualityScorer:
    """LLM-based quality scoring with budget control."""
    
    def __init__(
        self,
        budget_service: BudgetService,
        model: str = "llama-3.1-8b-instant",
    ):
        self.budget_service = budget_service
        self.model = model
    
    SYSTEM_PROMPT = """You are a quality evaluator for coding assistant sessions.
Evaluate the session on:
- Problem clarity
- Solution effectiveness  
- Code quality
- Tool usage appropriateness
Return a JSON object with scores 0-100 for each dimension and an overall score."""
    
    async def score(
        self,
        session: dict,
        heuristic_signals: QualitySignals,
    ) -> dict:
        """Score session using LLM with heuristic context."""
        # Build prompt with truncated session content
        # Reserve budget
        # Call LLM
        # Parse and return scores
```

#### Integration Points:

1. **Config** (`dataclaw/config.py`):
   ```python
   class DataClawConfig(TypedDict, total=False):
       # ... existing ...
       quality_scoring: dict | None  # {"enabled": bool, "llm_threshold": float}
   ```

2. **CLI** (`dataclaw/cli.py`):
   ```python
   qscore = sub.add_parser("quality-score", help="Score session quality")
   qscore.add_argument("--project", type=str)
   qscore.add_argument("--use-llm", action="store_true")
   ```

3. **Audit** (`scout/audit.py`):
   ```python
   # Add to EVENT_TYPES:
   "quality_score"
   
   # Log scoring events:
   audit.log("quality_score", score=85, method="heuristic", cost=0.001)
   ```

---

## 11. Gaps Summary

| Gap | Location | Severity | Recommendation |
|-----|----------|----------|----------------|
| Model name redaction | `secrets.py` | HIGH | Add patterns, create dedicated function |
| Project-relative paths | `anonymizer.py` | MEDIUM | Accept optional project_root param |
| Quality scorer | N/A | HIGH | Build from scratch |
| LLM scoring budget | N/A | MEDIUM | Reuse BudgetService |
| Batch scoring | N/A | MEDIUM | Extend batch_process for scoring |
| Test coverage | N/A | MEDIUM | Add tests for new components |

---

## 12. Timeline Estimate

| Phase | Tasks | Effort |
|-------|-------|--------|
| **M3a** | Model name patterns + scanner | 1-2 days |
| **M3b** | Config + CLI integration | 1 day |
| **M3c** | Testing | 0.5 day |
| **M4a** | Heuristic signals + scorer | 2 days |
| **M4b** | LLM scorer + budget | 2 days |
| **M4c** | CLI + audit integration | 1 day |
| **M4d** | Testing | 1 day |
| **Total** | | **8.5 days** |

---

## Appendix: Key File Reference

| Component | File Path |
|-----------|-----------|
| AnonymizerWrapper | `/Users/vivariumenv1/GITHUBS/dataclaw/dataclaw/parser.py` |
| Secrets Detection | `/Users/vivariumenv1/GITHUBS/dataclaw/dataclaw/secrets.py` |
| Core Anonymizer | `/Users/vivariumenv1/GITHUBS/scout/src/scout/tools/anonymizer.py` |
| dataclaw Config | `/Users/vivariumenv1/GITHUBS/dataclaw/dataclaw/config.py` |
| Scout Config | `/Users/vivariumenv1/GITHUBS/scout/src/scout/app_config.py` |
| LLM Router | `/Users/vivariumenv1/GITHUBS/scout/src/scout/llm/router.py` |
| Batch Process | `/Users/vivariumenv1/GITHUBS/scout/src/scout/llm/batch.py` |
| Budget Service | `/Users/vivariumenv1/GITHUBS/scout/src/scout/llm/budget.py` |
| Audit Log | `/Users/vivariumenv1/GITHUBS/scout/src/scout/audit.py` |
| CLI | `/Users/vivariumenv1/GITHUBS/dataclaw/dataclaw/cli.py` |
| Tests - Anonymizer | `/Users/vivariumenv1/GITHUBS/dataclaw/tests/test_anonymizer.py` |
| Tests - Secrets | `/Users/vivariumenv1/GITHUBS/dataclaw/tests/test_secrets.py` |
