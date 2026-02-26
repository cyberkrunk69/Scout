# Research Report: `/think-cheap` Mode Implementation

## Executive Summary

This report documents the technical feasibility study for implementing the `/think-cheap` mode - a semantic enrichment layer using Groq 8b to add intent, emotional tags, and security markers to sessions on demand. 

**Key Finding**: The Scout and DataClaw codebases provide strong foundations for all required components, but **the Groq LLM provider does not yet exist** - it's marked for Phase 2 extraction. This is the primary gap that must be addressed first.

---

## 1. Groq 8b Provider Capabilities

### Current State: NOT YET IMPLEMENTED

The Scout codebase has extensive infrastructure for LLM routing, but the actual LLM provider modules are **not yet extracted** (marked as "Phase 2" in comments). Here's what exists:

#### Existing Code References
- **Groq model reference**: `middle_manager.py` line 59 defines `GROQ_70B_MODEL = "llama-3.3-70b-versatile"`
- **LLM router stub**: `router.py` line 30-31 has imports for `# from scout.llm.router import call_llm`
- **Cost constants**: `router.py` lines 55-62 define token cost estimates

#### What Needs to Be Built

**File to create**: `src/scout/llm/providers/groq.py`

```python
# Proposed Groq Provider Structure
class GroqProvider:
    """Groq LLM provider with JSON schema support and cost tracking."""
    
    DEFAULT_MODEL = "llama-3.1-8b-instant"  # or the 8b model when available
    
    def __init__(self, api_key: str, audit: Optional[AuditLog] = None):
        self.client = Groq(api_key=api_key)
        self._audit = audit or AuditLog()
    
    async def complete(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        response_format: Optional[dict] = None,  # JSON schema support
        max_tokens: int = 1024,
    ) -> LLMResponse:
        # Implementation needed
        pass
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, model: str) -> float:
        # Based on Groq pricing: ~$0.20/M input, $0.40/M output for 8b
        rates = {"llama-3.1-8b-instant": (0.0002, 0.0004)}
        input_rate, output_rate = rates.get(model, (0.0002, 0.0004))
        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
```

#### JSON Mode / Function Calling

Groq supports structured output via the `response_format` parameter. Based on Groq API docs:

```python
# Example: JSON schema enforcement
response = self.client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": prompt}],
    response_format={"type": "json_object"},
    # Or for strict schema:
    # response_format={"type": "json_object", "schema": enrichment_schema}
)
```

#### Batching: NOT NATIVELY SUPPORTED

Groq does **not** have a batch API. The recommended approach is parallel requests with concurrency control:

```python
async def batch_complete(
    self,
    prompts: list[str],
    model: str = "llama-3.1-8b-instant",
    max_concurrent: int = 5,
) -> list[LLMResponse]:
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def _complete_with_sem(prompt: str) -> LLMResponse:
        async with semaphore:
            return await self.complete(prompt, model)
    
    return await asyncio.gather(*[_complete(p) for p in prompts])
```

#### Cost Tracking

Cost calculation exists as constants in `router.py` lines 55-62:

```55:62:src/scout/router.py
# Token cost estimates (8B: $0.20/million, 70B: ~$0.90/million)
TOKENS_PER_SMALL_FILE = 500
COST_PER_MILLION_8B = 0.20
COST_PER_MILLION_70B = 0.90
BRIEFCOST_PER_FILE = 0.005
TASK_NAV_ESTIMATED_COST = 0.002  # 8B + retry + possible 70B escalation
DRAFT_COST_PER_FILE = 0.0004
```

The audit log (`audit.py`) tracks costs per call:

```173:252:src/scout/audit.py
def log(
    self,
    event_type: str,
    *,
    cost: Optional[float] = None,
    model: Optional[str] = None,
    input_t: Optional[int] = None,
    output_t: Optional[int] = None,
    # ...
) -> None:
```

#### Error Handling & Retries

Currently **no built-in retry mechanism** exists in the codebase. The `middle_manager.py` shows retry logic in the `MiddleManagerGate` class (lines 473-621), but this is application-level, not provider-level.

**Recommendation**: Implement in the Groq provider:
- Exponential backoff (base: 1s, max: 60s)
- Circuit breaker pattern (open after 5 consecutive failures)
- Audit logging for all retry attempts

---

## 2. Router & EIN Pattern

### Current State: STUBS EXIST

The router (`router.py`) has the foundational patterns but lacks full implementation:

#### Escalation Judge Logic

The escalation logic is partially implemented in `TriggerRouter` (lines 156-292) and `MiddleManagerGate`:

```272:298:src/scout/middle_manager.py
class MiddleManagerGate:
    """
    Gates context compression for big_brain integration.
    Tier 1: deterministic freshness
    Tier 2: compress via 70B + parse via BriefParser
    Tier 3: confidence threshold (0.75 conservative)
    """
```

The `TaskRouter` class (lines 1310-1470) provides intent-based routing:

```1320:1345:src/scout/router.py
class TaskRouter:
    """Decides execution path for a given intent."""
    
    TOOL_MAPPING: dict[IntentType, str] = {
        IntentType.QUERY_CODE: "scout_function_info",
        IntentType.TEST: "scout_run",
        IntentType.DOCUMENT: "scout_doc_sync",
    }
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.9
    LOW_CONFIDENCE_THRESHOLD = 0.7
```

#### Intent Classification

Intent classification is **stubbed** in `router.py` lines 35-51:

```35:51:src/scout/router.py
# Stub for IntentType (will be replaced with real import in Phase 2)
class IntentType:
    QUERY_CODE = "query_code"
    TEST = "test"
    DOCUMENT = "document"
    IMPLEMENT_FEATURE = "implement_feature"
    FIX_BUG = "fix_bug"
    REFACTOR = "refactor"
    OPTIMIZE = "optimize"
```

**Reuse Recommendation**: For `/think-cheap`, we can extend `IntentType` with enrichment-specific intents:
- `ENRICH_EMOTIONAL`: "analyze emotional patterns"
- `ENRICH_INTENT`: "classify session intent"
- `ENRICH_SECURITY`: "detect security issues"

---

## 3. Subtext Storage & Indexing

### Current State: FULLY IMPLEMENTED

The `search.py` module provides complete FTS5 + BM25 search:

#### FTS5 Index Structure

```29:80:src/scout/search.py
class SearchIndex:
    """FTS5 + BM25 search index with configurable field weights."""
    
    DEFAULT_CONFIG: SearchConfig = {
        "index_path": "~/.scout/search.db",
        "field_weights": {
            "title": 5.0,
            "content": 3.0,
        },
        "tokenizer": "porter unicode61",
    }
```

#### Schema for Enrichment Storage

The search index stores documents with flexible field weights. For enrichment, we should:

1. **Add new fields to the document structure**:
```python
{
    "id": "session_id",
    "title": "project - date",
    "content": "...",  # existing content
    "intent": "debug|feature|question|vent|other",
    "emotional_tags": "frustration,excitement",
    "security_flags": "exposed_api_key",
    "enrichment_confidence": 0.95,
    "enriched_at": "2026-02-24T12:00:00Z",
}
```

2. **Configure field weights for enrichment queries**:
```python
config = {
    "field_weights": {
        "title": 5.0,
        "content": 3.0,
        "emotional_tags": 4.0,  # Boost for emotional queries
        "intent": 3.5,
        "security_flags": 5.0,   # High priority for security
    }
}
```

#### Field-Weighted Queries

The current implementation supports field weights (lines 383-409):

```383:409:src/scout/search.py
# Use BM25 with field weights
weights = [self._field_weights.get(f, 1.0) for f in fields]
weight_str = ", ".join(str(w) for w in weights)
```

#### Incremental Updates

**`add_documents()` method exists** (lines 182-254):

```182:254:src/scout/search.py
def add_documents(self, documents: List[Dict[str, Any]]) -> int:
    """Insert or update documents."""
    # Check if exists, delete old, insert new
```

This is perfect for `/think-cheap` - we can update only enriched sessions without rebuilding the entire index.

---

## 4. DataClaw's Session Parser

### Current State: FULLY IMPLEMENTED

The session parser (`dataclaw/parser.py`) provides comprehensive session parsing:

#### Session Parsing Flow

```104:129:src/dataclaw/dataclaw/parser.py
def parse_project_sessions(
    project_dir_name: str,
    anonymizer: AnonymizerWrapper,
    include_thinking: bool = True,
    claude_dir: Path | None = None,
) -> list[dict]:
```

#### Data Structure

```176:184:src/dataclaw/dataclaw/parser.py
return {
    "session_id": metadata["session_id"],
    "model": metadata["model"],
    "git_branch": metadata["git_branch"],
    "start_time": metadata["start_time"],
    "end_time": metadata["end_time"],
    "messages": messages,
    "stats": stats,
}
```

#### Session ID Correlation

The `session_id` is available in both:
- Parsed session: `session["session_id"]`
- Search index document: `doc["id"]` = session ID

This provides the join key for correlating search results with session data.

#### Anonymizer Integration

```19:55:src/dataclaw/dataclaw/parser.py
class AnonymizerWrapper:
    """Wrapper around Scout AnonymizerTool."""
    
    def __init__(self, extra_usernames: list[str] | None = None):
        self._tool = AnonymizerTool()
```

The anonymizer is already integrated into the parsing pipeline - enrichment should happen **after** anonymization.

---

## 5. Enrichment Schema

### Proposed JSON Schema

```json
{
  "session_id": "uuid-string",
  "enrichment": {
    "intent": "debug|feature|question|vent|exploration|other",
    "intent_confidence": 0.92,
    "emotional_tags": ["frustration", "excitement", "confusion"],
    "emotional_confidence": 0.88,
    "security_issues": ["exposed_api_key", "jwt_token"],
    "security_confidence": 0.95,
    "enriched_at": "2026-02-24T12:00:00Z",
    "model_used": "llama-3.1-8b-instant",
    "cost_usd": 0.0003
  }
}
```

### Storage Strategy

1. **Add to session JSON** (subtext):
   - Add `enrichment` object to each session
   - Fields are optional - check `if "enrichment" in session`

2. **Add to FTS5 index**:
   - Flatten enrichment fields for searching
   - Use separate columns: `intent`, `emotional_tags`, `security_flags`

---

## 6. Batching Logic

### Proposed Implementation

Since Groq doesn't support native batching, we implement application-level batching:

```python
# dataclaw/enrichment.py

class EnrichmentBatcher:
    """Batch enrichment requests with concurrency control."""
    
    def __init__(
        self,
        groq_provider: GroqProvider,
        index: SearchIndex,
        max_concurrent: int = 5,
        batch_size: int = 10,
    ):
        self.groq = groq_provider
        self.index = index
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
    
    async def enrich_sessions(
        self,
        sessions: list[dict],
        dimensions: list[str] = ["intent", "emotional", "security"],
    ) -> list[dict]:
        """Enrich sessions with requested dimensions."""
        
        # Track which sessions need which enrichments
        # (Query index for sessions missing specific fields)
        
        # Build prompts for each session
        prompts = []
        for session in sessions:
            prompt = self._build_enrichment_prompt(session, dimensions)
            prompts.append((session["session_id"], prompt))
        
        # Batch execute with concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def _enrich_with_sem(sess_id: str, prompt: str):
            async with semaphore:
                return await self._enrich_single(sess_id, prompt, dimensions)
        
        results = await asyncio.gather(*[
            _enrich_with_sem(sid, prompt) 
            for sid, prompt in prompts
        ])
        
        # Update index with enriched documents
        for session, enrichment in zip(sessions, results):
            self._apply_enrichment(session, enrichment)
        
        # Batch update index
        self.index.add_documents(sessions)
        
        return results
```

### Sync/Async Decision

**Recommendation**: Use async for LLM calls, with sync wrapper for CLI:

```python
# For CLI compatibility
def enrich_command_sync(query: str, dimensions: list[str]) -> dict:
    """Sync wrapper for CLI."""
    return asyncio.run(enrich_command_async(query, dimensions))
```

---

## 7. CLI Integration

### Current Pattern

DataClaw CLI (`cli.py`) uses argparse with subcommands:

```656:710:src/dataclaw/dataclaw/cli.py
def main() -> None:
    parser = argparse.ArgumentParser(description="DataClaw — Claude Code -> Hugging Face")
    sub = parser.add_subparsers(dest="command")
    
    # Subcommands: prep, status, confirm, list, config, export, index, search
```

#### Handler Pattern

```771:812:src/dataclaw/dataclaw/cli.py
def _handle_index(args) -> None:
    """Handle the index subcommand - build search index."""
    # Implementation
```

### `/think-cheap` Command Implementation

Add to `cli.py`:

```python
# New subcommand
think = sub.add_parser("think-cheap", help="Enrich sessions with intent, emotions, security")
think.add_argument("query", type=str, nargs="?", help="Search query to match sessions")
think.add_argument("--dimensions", "-d", type=str, 
                   default="intent,emotional,security",
                   help="Comma-separated dimensions to enrich")
think.add_argument("--limit", "-n", type=int, default=20,
                   help="Max sessions to enrich")
think.add_argument("--budget", "-b", type=float, default=0.50,
                   help="Max cost budget (USD)")

# Handler
if command == "think-cheap":
    _handle_think_cheap(args)
    return

def _handle_think_cheap(args) -> None:
    """Handle the think-cheap subcommand."""
    # 1. Search for candidate sessions
    # 2. Filter those missing enrichment
    # 3. Batch enrich with Groq
    # 4. Update index
    # 5. Log to audit
```

---

## 8. Audit Logging

### Current Audit Implementation

Full audit system in `audit.py`:

```93:252:src/scout/audit.py
class AuditLog:
    """Append-only JSONL event log with line buffering, fsync cadence."""
    
    def log(
        self,
        event_type: str,
        *,
        cost: Optional[float] = None,
        model: Optional[str] = None,
        input_t: Optional[int] = None,
        output_t: Optional[int] = None,
        # ...
    ) -> None:
```

#### Event Types

```37:77:src/scout/audit.py
EVENT_TYPES = frozenset({
    "nav", "brief", "cascade", "validation_fail", "budget", "skip",
    "trigger", "tldr", "deep", "doc_sync", "commit_draft", "pr_snippet",
    # ... 
})
```

**Recommendation**: Add new event types:
- `think_cheap_start`
- `think_cheap_batch`
- `think_cheap_session`
- `think_cheap_complete`

### Usage for `/think-cheap`

```python
# Log enrichment batch
audit.log(
    "think_cheap_batch",
    cost=total_cost,
    model="llama-3.1-8b-instant",
    sessions_enriched=len(sessions),
    dimensions=dimensions,
    budget_limit=args.budget,
)

# Log per-session enrichment
for session_id, enrichment in results:
    audit.log(
        "think_cheap_session",
        session_id=session_id,
        cost=enrichment.get("cost_usd"),
        dimensions_enriched=list(enrichment.keys()),
    )
```

---

## 9. Cost Controls

### Current Budget System

Full budget controls in `config.py`:

```91:114:src/scout/config.py
# Hard safety caps (NON-OVERRIDABLE)
HARD_MAX_COST_PER_EVENT = 1.00
HARD_MAX_HOURLY_BUDGET = 10.00
HARD_MAX_AUTO_ESCALATIONS = 3
```

```439:458:src/scout/config.py
def should_process(
    self,
    estimated_cost: float,
    file_path: Optional[Path] = None,
    hourly_spend: float = 0.0,
) -> bool:
    """Check all limits before any LLM call."""
```

#### Hourly Spend Tracking

```301:308:src/scout/audit.py
def hourly_spend(self, hours: int = 1) -> float:
    """Sum costs in last N hours."""
    cutoff = datetime.now(timezone.utc).replace(microsecond=0, second=0, minute=0)
    cutoff = cutoff - timedelta(hours=hours)
    events = self.query(since=cutoff)
    return sum(e.get("cost", 0) or 0 for e in events)
```

### `/think-cheap` Budget Integration

```python
def estimate_enrichment_cost(sessions: list[dict], dimensions: list[str]) -> float:
    """Estimate cost before running enrichment."""
    # ~500 tokens input per session, ~200 tokens output
    tokens_per_session = 700
    sessions_count = len(sessions)
    dimensions_count = len(dimensions)
    
    # Rough estimate: $0.20/1M input, $0.40/1M output for 8b
    input_cost = (sessions_count * tokens_per_session * 0.20) / 1_000_000
    output_cost = (sessions_count * 200 * 0.40 * dimensions_count) / 1_000_000
    
    return input_cost + output_cost
```

---

## 10. Testing Strategy

### Current Test Coverage

**Scout tests**: Empty (`tests/` directory is empty)

**DataClaw tests**: Full test suite in `dataclaw/tests/`:
- `test_parser.py` - Session parsing tests
- `test_cli.py` - CLI command tests
- `test_secrets.py` - Secret redaction tests
- `test_config.py` - Configuration tests
- `conftest.py` - Fixtures

### New Tests Needed

```python
# tests/test_enrichment.py

def test_enrichment_schema_validation():
    """Test enrichment JSON schema."""
    pass

def test_groq_provider_json_mode():
    """Test Groq provider with JSON schema."""
    pass

def test_enrichment_batcher_concurrency():
    """Test batcher respects max_concurrent."""
    pass

def test_enrichment_cost_estimation():
    """Test cost estimation accuracy."""
    pass

def test_cli_think_cheap_command():
    """Test think-cheap CLI integration."""
    pass

def test_enrichment_search_integration():
    """Test enrichment fields are searchable."""
    pass
```

---

## 11. Implementation Roadmap

### Phase 1: Foundation (Week 1)

| Task | Hours | Files to Create/Modify |
|------|-------|----------------------|
| Create Groq provider with JSON mode | 4 | `src/scout/llm/__init__.py`, `src/scout/llm/providers/groq.py` |
| Add cost tracking to provider | 2 | `src/scout/llm/providers/groq.py` |
| Add retry/circuit breaker | 2 | `src/scout/llm/providers/groq.py` |
| Create enrichment schema | 1 | New file or extend parser |

### Phase 2: Core Enrichment (Week 2)

| Task | Hours | Files to Create/Modify |
|------|-------|----------------------|
| Implement `EnrichmentBatcher` | 4 | `dataclaw/enrichment.py` |
| Create Groq prompts for enrichment | 2 | `dataclaw/enrichment.py` |
| Add enrichment fields to search index | 2 | `dataclaw/search.py` |
| Integrate with session parsing | 2 | `dataclaw/parser.py` |

### Phase 3: CLI Integration (Week 2-3)

| Task | Hours | Files to Create/Modify |
|------|-------|----------------------|
| Add `think-cheap` subcommand | 3 | `dataclaw/cli.py` |
| Implement budget checking | 2 | `dataclaw/cli.py` |
| Add audit logging | 2 | `dataclaw/cli.py`, `scout/audit.py` |

### Phase 4: Testing & Polish (Week 3)

| Task | Hours | Files to Create/Modify |
|------|-------|----------------------|
| Write unit tests | 4 | `tests/test_enrichment.py` |
| Integration tests | 3 | New test files |
| Documentation | 2 | Update README |

### Total Estimated Hours: 33

---

## Risk Assessment

| Component | Risk Level | Mitigation |
|-----------|------------|------------|
| Groq JSON mode | Medium | Test with sample schemas; Groq supports `response_format` |
| Batching performance | Low | Use asyncio with semaphore; proven pattern |
| Index schema changes | Medium | Version the schema; add migration path |
| Cost overruns | Medium | Implement budget guard; estimate before running |
| Session correlation | Low | Use stable session_id from parser |

---

## Recommendations Summary

1. **Create Groq provider first** - This is a prerequisite for all LLM operations
2. **Reuse Scout's search index** - Already has FTS5, BM25, field weights, incremental updates
3. **Reuse DataClaw's parser** - Already provides session parsing with session_id correlation
4. **Reuse Scout's audit** - Comprehensive audit logging with hourly spend tracking
5. **Add enrichment tracking table** - SQLite table to track which sessions have which enrichments
6. **Use async with sync wrapper** - Scout uses asyncio; DataClaw CLI is sync - use `asyncio.run()`

---

## Appendix: File Reference Map

### Scout Core Files
| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `src/scout/search.py` | FTS5 + BM25 search | `SearchIndex`, `build()`, `add_documents()`, `search()` |
| `src/scout/audit.py` | Audit logging | `AuditLog`, `log()`, `hourly_spend()` |
| `src/scout/config.py` | Configuration | `ScoutConfig`, `should_process()`, `hourly_spend()` |
| `src/scout/router.py` | LLM routing | `TriggerRouter`, `TaskRouter`, `IntentType` |
| `src/scout/middle_manager.py` | Context gating | `MiddleManagerGate`, `BriefParser` |

### DataClaw Files
| File | Purpose | Key Functions/Classes |
|------|---------|----------------------|
| `dataclaw/parser.py` | Session parsing | `parse_project_sessions()`, `AnonymizerWrapper` |
| `dataclaw/cli.py` | CLI commands | `main()`, `_handle_index()`, `_handle_search()` |
| `dataclaw/search.py` | Search integration | `build_index()`, `search()`, `_session_to_document()` |
| `dataclaw/config.py` | Configuration | `load_config()`, `save_config()` |

### Files to Create
| File | Purpose |
|------|---------|
| `src/scout/llm/__init__.py` | LLM package init |
| `src/scout/llm/providers/groq.py` | Groq provider |
| `dataclaw/enrichment.py` | Enrichment batching logic |
| `dataclaw/enrichment_config.py` | Enrichment-specific config |
| `tests/test_enrichment.py` | Enrichment tests |

---

*Report generated: 2026-02-24*
*Research conducted by: AI Assistant*
