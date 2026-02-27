# Plan & Execution System — Deep Dive

The Plan & Execution system is Scout's crown jewel — a production-grade autonomous agent platform that transforms natural language requests into executed, validated, and safely guarded actions.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BIG BRAIN (big_brain.py)                         │
│    Intent Routing • Dual-Model Context Compression • Autonomous Planning   │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MIDDLE MANAGER (middle_manager.py)                    │
│         Tier 1: Freshness Check → Tier 2: 70B Compress → Tier 3: Gate    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PLAN STATE (plan_state.py)                           │
│            Atomic Locking • Lifecycle Management • Disk Persistence          │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   BATCH PIPELINE (batch_pipeline.py)                       │
│    Circuit Breaker • Retry Logic • Conditionals • Variable Interpolation    │
└───────────────────────────┬─────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│   SAFETY      │  │ QUALITY GATES │  │    CIRCUIT   │
│  (safety.py) │  │ (runtime.py)  │  │  BREAKER     │
│ Path Guards   │  │ Unit→Int→E2E  │  │  Resilience   │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## 1. Big Brain — Orchestration Layer

The top-level coordinator handling all AI-driven operations.

### Intent Interpretation

- **Pattern-based fast routing**: Regex classifiers for `nav` (symbol lookup) vs `query` (documentation) — instant, free, no LLM call.
- **LLM fallback**: When patterns don't match, routes to MiniMax for natural language interpretation.
- **Tool selection**: Picks cheapest tool satisfying the user's request.

### Dual-Model Routing (TICKET-19, TICKET-43)

| Model | When Used | Cost |
|-------|-----------|------|
| **M2.1 (Flash)** | Confidence high, context compression passed | ~$0.001/1K tokens |
| **M2.5 (Pro)** | Escalation needed, raw facts required | ~$0.015/1K tokens |

Cost optimization: Gate-approved briefs route to cheap models; only escalate to expensive ones when necessary.

### Autonomous Planning

Takes natural language requests → generates structured execution plans:

```json
{
  "plan": [
    {
      "command": "nav",
      "args": {"task": "auth_handler"},
      "depends_on": [],
      "reasoning": "First, locate the authentication module"
    },
    {
      "command": "query",
      "args": {"scope": "auth.py", "include_deep": true},
      "depends_on": [0],
      "reasoning": "Then understand how auth flow works"
    }
  ],
  "reasoning": "Explore auth before making changes"
}
```

---

## 2. Middle Manager — The Gate System

The brain of context compression. Implements a **3-tier freshness gate**:

### Tier 1: Deterministic Freshness
- Checks `deps_graph` trust metadata for `invalidation_cascade_triggered` and `stale_ratio`
- If >50% symbols stale → **immediate escalation** (no API call)

### Tier 2: 70B Compression
- Calls Groq `llama-3.3-70b-versatile` to compress context
- Extracts structured output via `BriefParser`:
  - `confidence_score`: float 0.00–1.00
  - `[GAP]` markers: missing critical information
  - `None identified`: explicit complete coverage

### Tier 3: Confidence Threshold
- Default threshold: **0.75**
- Confidence ≥ threshold + no suspicious → **pass** (use compressed)
- Confidence < threshold OR suspicious → retry up to **3 times**
- Max retries → **escalate** to raw facts

### Gap Expansion

When low confidence + gaps detected:
1. Extract symbols from gap text via regex: `(\S+\.py)::(\w+)`
2. Call `hydrate_facts()` to fetch more structured facts
3. Single expansion depth — prevents infinite loops
4. Retry with expanded context

---

## 3. Plan State — Lifecycle Management

Disk-based, atomic locking for plan execution.

### Lock Mechanism

- **Atomic mkdir** (`O_CREAT | O_EXCL`) — POSIX-safe, no race conditions
- Stores: PID, timestamp, config
- **Stale lock recovery**: scans locks >1 hour old
- **Timeout**: 30 seconds default

### Lifecycle States

```
active/ ──────► completed/ ──────► archived/
   │                  │                   │
 executing        success            failed/manual
```

### Persistence

- **Atomic writes**: temp file + rename
- Full context serialization:
  - request, depth, max_depth
  - summary, parent_goals, constraints
  - discoveries, pivots_needed, sub_plan_outcomes
- JSON-based, human-readable

---

## 4. Batch Pipeline — Execution Runtime

The runtime executor for plans and batch operations.

### Execution Modes

- **Sequential** (default): tasks run one-by-one
- **Parallel**: independent tasks run concurrently

### Self-Healing Features

#### Circuit Breaker
- Pre-configured for 17+ commands needing JSON output
- Tracks failures, opens circuit after threshold
- Stops execution when OPEN

#### Retry Logic
- Exponential backoff
- `is_retryable()` checks error type
- Per-task attempt tracking

#### Conditionals
```yaml
- command: run_tests
  if: ${tests_exist} == true

- command: deploy
  skip_if: ${environment} == "staging"

- command: rollback
  stop_if: ${deployment_failed} == true
```

#### Variable Interpolation
- `${variable}` syntax
- `store_as`: save output to context
- Results available to subsequent tasks

#### Auto-JSON Injection
Automatically injects `--json` for commands needing structured output:
- git_status, git_diff, git_log, git_branch, git_show
- plan, lint, audit, run, nav, doc_sync, query
- roast, status, index, validate, ci_guard, brief

---

## 5. Safety Guard — Security Layer

Protects the workspace from malicious operations.

### Path Validation
- `pathlib.resolve()` + `samefile()` — symlink-aware
- Prevents traversal: `../../etc/passwd`
- Falls back to string prefix if workspace doesn't exist

### Command Whitelist
```python
DEFAULT_COMMAND_WHITELIST = [
    "git", "npm", "python", "pip", "ruff", "pytest",
    "node", "docker", "yarn", "pnpm", "uv"
]
```

### Resource Limits
- `SAFETY_MAX_PATH_DEPTH`: 15
- `SAFETY_MAX_LIST_DEPTH`: 10
- `SAFETY_MAX_FILE_SIZE_KB`: 1024

### Primitive Tools
- `scout_mkdir`, `scout_remove`, `scout_copy`, `scout_move`
- `scout_list`, `scout_read_file`, `scout_write_file`
- `scout_command` (with timeout + whitelist)
- `scout_wait`, `scout_condition` (polling)

---

## 6. Quality Gates — Validation Pipeline

Full CI/CD-like pipeline for code changes.

### Gate Stages
```
Unit Test → Integration Test → E2E Test → Ready for Merge
```

### Flow
1. Change submitted → blind vote ballot created
2. Approved → moves to `needs_qa`
3. QA writes unit test → test ballot created
4. Approved → unit tests run → integration task created
5. Integration batched → runs → e2e task created
6. All three pass → `ready_for_merge`

### Change Lifecycle
```
pending_vote → needs_qa → qa_in_progress → needs_integration 
           → needs_e2e → ready_for_merge
           
(rejected / blocked at any stage)
```

---

## 7. Circuit Breaker — Failure Isolation

### States
- **CLOSED**: Normal operation
- **OPEN**: Failing — reject calls
- **HALF_OPEN**: Testing recovery

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| failure_threshold | 5 | Failures before opening |
| success_threshold | 3 | Successes to close |
| timeout | 60s | Wait before HALF_OPEN |
| permanent_failure | 10 | Mark permanently failed |

### Provider vs Operations

- **Providers**: Longer cooldown (60s), permanent failure detection
- **Operations**: Shorter timeout (30s), no permanent failure

---

## 8. Key Design Principles

### Cost Is a Product Feature
- Index-first routing (free) → LLM only when needed
- Dual-model: M2.1 for pass, M2.5 for escalate
- Hard caps: per-event and hourly limits
- Every call logged with model, tokens, cost

### Progressive Autonomy
- Assistive tooling → CI guards → Git hooks → expand autonomy where confidence high
- Quality gates prevent bad changes from merging

### Truth Over Style
- Confidence scores reflect actual uncertainty
- Gap markers explicitly identify missing context
- Freshness validation via FACT_CHECKSUM

### Git-Native by Design
- Plans live in `.scout/plans/`
- Subtext travels with code
- Hooks integrate with Git workflow

---

## Related Documentation

- [ADR-006: Execution Framework](../adr/ADR-006-execution-framework.md)
- [ADR-007: Plan Executor State Machine](../adr/ADR-007-plan-executor-state-machine.md)
- [ADR-008: Batch Pipeline](../adr/ADR-008-batch-pipeline.md)
- [ADR-010: Retry Mechanisms](../adr/ADR-010-retry-mechanisms.md)
- [Plan Execution Guide](../guides/plan_execution.md)
- [Batch Pipeline Guide](../guides/batch_pipeline.md)
