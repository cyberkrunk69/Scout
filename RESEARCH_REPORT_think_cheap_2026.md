# Research Spike: Deep Dive for `/think-cheap` Implementation

**Date:** February 25, 2026  
**Objective:** Gather technical details for implementing `/think-cheap` – a semantic enrichment layer using Groq 8b to add intent, emotional tags, and security markers to sessions on demand.

---

## Executive Summary

The scout-core codebase provides **all necessary infrastructure** for `/think-cheap` implementation. However, the **DataClaw module does not exist** in this repository – it was planned but never implemented. This creates a clear path: implement the DataClaw layer on top of existing scout-core components.

**Key Findings:**
- ✅ Groq provider exists (`src/scout/llm/providers/groq.py`)
- ✅ LLM routing with fallback (`src/scout/llm/router.py`)
- ✅ Batching utility (`src/scout/llm/batch.py`)
- ✅ Budget service (`src/scout/llm/budget.py`)
- ✅ Audit logging with "enrich" event type (`src/scout/ ✅ FTS5audit.py`)
- + BM25 search index (`src/scout/search.py`)
- ✅ Anonymizer tool (`src/scout/tools/anonymizer.py`)
- ❌ DataClaw module (does not exist)
- ❌ Session enrichment tracking (needs to be built)

---

## 1. Session Data Flow in DataClaw

### Finding: DataClaw Module Does Not Exist

The research questions reference `dataclaw/parser.py` and `dataclaw/search.py`, but **these files do not exist** in the current codebase. The reports reference a planned DataClaw implementation that was never created in this repository.

### Existing Search Index (Scout-core)

**File:** `src/scout/search.py`

The `SearchIndex` class provides FTS5 + BM25 search:

```29:45:src/scout/search.py
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

**Session ID Correlation:**
- Documents use `id` field as the unique identifier
- `add_documents()` method supports upserts (lines 182-254):

```182:254:src/scout/search.py
def add_documents(self, documents: List[Dict[str, Any]]) -> int:
    """Insert or update documents."""
    # Check if exists, delete old, insert new
```

**Storage Strategy for Enrichment Fields:**
- Current schema: FTS5 table with `title` and `content` columns
- Extra metadata stored in `doc_metadata` table as JSON
- **Recommendation:** Flatten enrichment fields into `content` as key-value strings (e.g., "intent:debug emotional_tags:frustration") – this keeps schema simple and enables FTS5 searching

---

## 2. Enrichment Orchestrator (from Track B)

### Finding: Design Exists, Implementation Does Not

**File referenced in reports:** `dataclaw/enrichment.py`  
**Status:** Does not exist

The Track B reports describe the intended design:

```python
# From REPORT_Track_B_Intent_Classifier.md
class EnrichmentOrchestrator:
    def __init__(
        self,
        llm_call: Callable,
        batch_size: int = 10,
        max_concurrent: int = 5,
    ):
        self.llm_call = llm_call
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
```

**Methods described:**
- `enrich_sessions()`: Batch processing (STUB - depends on Tracks A & C)
- `enrich_single()`: Single text enrichment

**Pydantic Models described:**

```python
class IntentType(str, Enum):
    DEBUG = "debug"
    FEATURE = "feature"
    QUESTION = "question"
    VENT = "vent"
    EXPLORATION = "exploration"
    OTHER = "other"

class EmotionalEnrichment(BaseModel):
    emotional_tags: List[str]
    confidence: float

class SecurityEnrichment(BaseModel):
    security_issues: List[str]
    confidence: float
    excerpts: Optional[List[str]] = None

class IntentEnrichment(BaseModel):
    intent: IntentType
    confidence: float
```

---

## 3. LLM Provider & Batching

### 3.1 Groq Provider

**File:** `src/scout/llm/providers/groq.py`

The Groq provider is **fully implemented**:

```14:71:src/scout/llm/providers/groq.py
async def _call_groq(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    api_key: str = None,
    **kwargs,
) -> ProviderResult:
    # Implementation exists
```

**Supported Models** (from `src/scout/llm/__init__.py`):

```21:27:src/scout/llm/__init__.py
SUPPORTED_MODELS = {
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
}
```

**JSON Mode:** Groq supports `response_format={"type": "json_object"}` but not strict schema validation. For enrichment, we'll need prompt engineering to ensure JSON output.

### 3.2 Router - call_llm Function

**File:** `src/scout/llm/router.py`

The `call_llm` function is the **primary entry point**:

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
```

Returns `LLMResult` (dataclass with content, cost_usd, model, provider, input_tokens, output_tokens).

### 3.3 Batching Utility

**File:** `src/scout/llm/batch.py`

```10:40:src/scout/llm/batch.py
async def batch_process(
    prompts: List[str],
    func: Callable[[str], Awaitable[Any]],
    max_concurrent: int = 5,
    rate_limiter: Optional[RateLimiter] = None,
    return_exceptions: bool = False,
) -> List[Union[Any, Exception]]:
    """Process a list of prompts with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    # ... implementation
```

**Perfect for `/think-cheap`**: Can process multiple session enrichments in parallel.

### 3.4 Budget Service

**File:** `src/scout/llm/budget.py`

```79:172:src/scout/llm/budget.py
class BudgetService:
    """Centralized budget management with reservation semantics."""
    
    def check(self, estimated_cost: float, operation: str) -> bool
    def reserve(self, estimated_cost: float, operation: str, ...) -> BudgetReservation
    def commit(self, reservation: Reservation, actual_cost: float) -> None
    def rollback(self, reservation: Reservation) -> None
```

**Usage Pattern:**
```python
reservation = budget_service.reserve(estimated_cost=0.10, operation="enrichment_batch")
try:
    # Do enrichment
    budget_service.commit(reservation, actual_cost)
except Exception:
    budget_service.rollback(reservation)
    raise
```

---

## 4. Audit Logging

**File:** `src/scout/audit.py`

The audit system is **fully implemented** and already has the "enrich" event type:

```37:79:src/scout/audit.py
EVENT_TYPES = frozenset({
    # ... other events ...
    # DataClaw enrichment events
    "enrich",
})
```

**Logging Enrichment Events:**

```175:254:src/scout/audit.py
def log(
    self,
    event_type: str,
    *,
    cost: Optional[float] = None,
    model: Optional[str] = None,
    input_t: Optional[int] = None,
    output_t: Optional[int] = None,
    **kwargs: Any,
) -> None:
```

**Get Audit Instance:**

```568:574:src/scout/audit.py
def get_audit() -> AuditLog:
    """Get or create global AuditLog instance."""
    global _audit_instance
    if _audit_instance is None:
        _audit_instance = AuditLog()
    return _audit_instance
```

**Example Usage for /think-cheap:**
```python
audit = get_audit()
audit.log(
    "enrich",
    cost=0.001,
    model="llama-3.1-8b-instant",
    input_t=100,
    output_t=50,
    dimension="emotional",
    session_id="abc123",
)
```

---

## 5. CLI Integration

### Finding: No DataClaw CLI Exists

The research questions reference `dataclaw/cli.py`, but this file does not exist. The Scout-core doesn't have a CLI for enrichment either.

**Recommendation:** Create a new CLI module or extend scout-core CLI with enrichment commands.

---

## 6. Search Index Enrichment Field Storage

### Current Schema

```89:124:src/scout/search.py
def _create_schema(self, conn: sqlite3.Connection) -> None:
    """Create FTS5 virtual table for documents."""
    # Current: documents table with title, content
    # Extra data stored in doc_metadata as JSON
```

### Recommended Approach: Flatten into Content

For enrichment fields to be searchable:

1. **Add to content string:**
```python
document = {
    "id": session_id,
    "title": f"{project} - {date}",
    "content": f"{messages}\n\nintent:debug emotional_tags:frustration,confusion security_flags:",
}
```

2. **Or store in metadata (not searchable):**
```python
extra = {
    "enrichment": {
        "intent": "debug",
        "emotional_tags": ["frustration", "confusion"],
        "security_flags": [],
    }
}
```

**Recommendation:** Flatten into `content` for searchability. Use metadata for structured retrieval.

---

## 7. Tracking Enriched Sessions

### Gap: No Enrichment Tracking Exists

**Options to implement:**

1. **Separate SQLite table** (`session_enrichments`):
```sql
CREATE TABLE session_enrichments (
    session_id TEXT PRIMARY KEY,
    dimensions TEXT,  -- JSON array: ["intent", "emotional"]
    enriched_at TIMESTAMP,
    model_used TEXT,
    cost_usd REAL
);
```

2. **Store in search metadata** - Query doc_metadata for sessions missing enrichment

3. **Re-enrich everything** - Simple but wasteful

**Recommendation:** Separate SQLite table for precise tracking

---

## 8. Existing Components That Can Be Reused

| Component | File | Status | Reuse Potential |
|-----------|------|--------|------------------|
| Groq Provider | `src/scout/llm/providers/groq.py` | ✅ Exists | Direct use |
| LLM Router | `src/scout/llm/router.py` | ✅ Exists | `call_llm()` |
| Batching | `src/scout/llm/batch.py` | ✅ Exists | `batch_process()` |
| Budget Service | `src/scout/llm/budget.py` | ✅ Exists | `BudgetService` |
| Audit | `src/scout/audit.py` | ✅ Exists | `get_audit()` |
| Search Index | `src/scout/search.py` | ✅ Exists | `SearchIndex` |
| Intent Classifier | `src/scout/llm/intent.py` | ✅ Exists | Can extend |
| Anonymizer | `src/scout/tools/anonymizer.py` | ✅ Exists | `AnonymizerTool` |
| Trust System | `src/scout/trust/` | ✅ Exists | Future quality scoring |

---

## 9. Potential Gaps

| Gap | Severity | Mitigation |
|-----|----------|-------------|
| DataClaw module doesn't exist | HIGH | Create from scratch |
| No enrichment tracking | HIGH | Build SQLite table |
| No CLI commands | HIGH | Create CLI module |
| JSON mode not strict | MEDIUM | Prompt engineering |
| No negative FTS5 queries | LOW | Track separately |

---

## 10. Implementation Plan

### Phase 1: Foundation (DataClaw Module Creation)

| Task | Hours | Files to Create |
|------|-------|-----------------|
| Create DataClaw package structure | 1 | `dataclaw/__init__.py`, `dataclaw/pyproject.toml` |
| Implement session parser | 3 | `dataclaw/parser.py` |
| Implement enrichment orchestrator | 4 | `dataclaw/enrichment.py` |
| Subtotal | **8** | |

### Phase 2: CLI & Integration

| Task | Hours | Files to Create |
|------|-------|-----------------|
| Create CLI with think-cheap command | 3 | `dataclaw/cli.py` |
| Integrate with search index | 2 | `dataclaw/search.py` |
| Add enrichment tracking table | 2 | `dataclaw/tracking.py` |
| Subtotal | **7** | |

### Phase 3: Testing & Polish

| Task | Hours |
|------|-------|
| Write unit tests | 4 |
| Integration tests | 3 |
| Documentation | 2 |
| Subtotal | **9** |

### Total Estimated Hours: **24**

---

## 11. Risk Assessment

| Component | Risk Level | Mitigation |
|-----------|------------|------------|
| DataClaw module creation | Medium | Use existing scout patterns |
| JSON parsing from LLM | Medium | Prompt engineering, validation |
| Cost overruns | Medium | Budget service pre-check |
| Session correlation | Low | Use stable session_id |

---

## 12. Recommendations Summary

1. **Create DataClaw module** - The core gap; all other components exist
2. **Reuse scout.llm.router.call_llm** - Already handles Groq, fallback, cost tracking
3. **Reuse scout.llm.batch.batch_process** - Perfect for parallel enrichment
4. **Reuse scout.audit.get_audit()** - Already has "enrich" event type
5. **Reuse scout.search.SearchIndex** - Use `add_documents()` for upserts
6. **Build enrichment tracking** - SQLite table for precise dimension tracking
7. **Flatten enrichment fields into content** - For FTS5 searchability

---

## Appendix: Key File Reference Map

### Scout-core Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/scout/llm/router.py` | LLM routing | `call_llm()` |
| `src/scout/llm/providers/groq.py` | Groq provider | `_call_groq()` |
| `src/scout/llm/batch.py` | Batching | `batch_process()` |
| `src/scout/llm/budget.py` | Budget service | `BudgetService` |
| `src/scout/audit.py` | Audit logging | `get_audit()`, `AuditLog.log()` |
| `src/scout/search.py` | Search index | `SearchIndex`, `add_documents()` |
| `src/scout/llm/intent.py` | Intent classifier | `IntentClassifier` |
| `src/scout/tools/anonymizer.py` | Anonymization | `AnonymizerTool` |

### Files to Create

| File | Purpose |
|------|---------|
| `dataclaw/__init__.py` | Package init |
| `dataclaw/pyproject.toml` | Dependencies |
| `dataclaw/parser.py` | Session parsing |
| `dataclaw/enrichment.py` | Enrichment orchestration |
| `dataclaw/search.py` | Search integration |
| `dataclaw/tracking.py` | Enrichment tracking |
| `dataclaw/cli.py` | CLI commands |

---

*Report generated: 2026-02-25*
*Research conducted by: AI Assistant*
