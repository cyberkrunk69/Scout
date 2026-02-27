# Scout-Core Extraction Plan: Comprehensive Migration Roadmap

## Executive Summary

The Vivarium codebase contains approximately **60,000+ lines** of Scout-related code across 100+ modules. The initial extraction to `scout-core` has already accomplished:

- ✅ LLM provider infrastructure (Groq)
- ✅ Cost tracking and budgeting
- ✅ Intent classification
- ✅ Anonymization

**Remaining extraction scope:** ~45,000 lines across 90+ modules

This document provides a detailed inventory and phased migration plan to extract the remaining high-value, reusable components.

---

## Categorized Inventory

### 1. LLM Infrastructure (Remaining)

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `llm/router.py` | 694 | LLM routing with ProviderRegistry, multi-key support | `llm/cost`, `llm/providers`, `llm/ratelimit` | **HIGH** - Generic provider routing | **P1** |
| `llm/dispatch.py` | 227 | Unified LLM dispatcher for multiple providers | `llm/__init__`, `llm/google`, `llm/anthropic`, `llm/minimax`, `audit` | **HIGH** - Multi-provider dispatch | **P1** |
| `llm/select.py` | 52 | Model selection based on task, tier, mode | `llm/cost`, `llm/providers` | **HIGH** - Generic model selection | **P1** |
| `llm/providers/__init__.py` | 350+ | ProviderRegistry with key health tracking | None (self-contained) | **HIGH** - Core infrastructure | **P1** |
| `llm/providers/groq.py` | ~100 | Groq provider implementation | `providers/__init__` | **HIGH** - Already partially done | **P1** |
| `llm/providers/google.py` | ~100 | Google Gemini provider | `providers/__init__` | **HIGH** - Generic provider | **P2** |
| `llm/providers/minimax.py` | ~100 | MiniMax provider | `providers/__init__` | **HIGH** - Generic provider | **P2** |
| `llm/anthropic.py` | 217 | Anthropic Claude provider | `llm/pricing` | **HIGH** - Generic provider | **P2** |
| `llm/google.py` | 192 | Google Gemini (legacy) | `llm/pricing` | **MEDIUM** - Legacy, use providers/ | **P3** |
| `llm/minimax.py` | 181 | MiniMax (legacy) | None | **MEDIUM** - Legacy, use providers/ | **P3** |
| `llm/cost.py` | 252 | Cost tracking, model configs | None | **HIGH** - Generic cost models | **P1** (if not done) |
| `llm/pricing.py` | 71 | Price estimation utilities | None | **HIGH** - Generic pricing | **P1** (if not done) |
| `llm/circuit_breaker.py` | 194 | Circuit breaker for providers | None | **HIGH** - Generic resilience | **P1** |
| `llm/ratelimit.py` | 75 | Rate limiting | None | **HIGH** - Generic rate limiting | **P1** |
| `llm/retry.py` | 226 | Retry logic with backoff | None | **HIGH** - Generic retry | **P2** |

**Extraction Strategy:** The provider pattern is already established. Extract `providers/` directory with the registry pattern, then add remaining providers (Google, MiniMax, Anthropic).

---

### 2. Tools Module

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `tools/__init__.py` | 639 | Tool registry and metadata | `tools/*`, `analysis/hotspots` | **HIGH** - Generic tool system | **P1** |
| `tools/llm.py` | 631 | LLM-powered tools (plan, nav, query, roast) | `cache`, `tool_output` | **MEDIUM** - Uses Scout-specific prompts | **P2** |
| `tools/git.py` | 464 | Git operations (status, diff, log, commit) | `cache`, `tool_output` | **HIGH** - Generic git ops | **P1** |
| `tools/github.py` | 511 | GitHub API tools (PR, issues) | `cache`, `tool_output` | **HIGH** - Generic GitHub API | **P1** |
| `tools/validation.py` | 282 | Linting and validation tools | `cache`, `tool_output` | **HIGH** - Generic validation | **P1** |
| `tools/file_ops.py` | 390 | File read/write/delete operations | `tool_output` | **HIGH** - Generic file ops | **P1** |
| `tools/batch.py` | 214 | Batch processing tools | `tool_output` | **HIGH** - Generic batch | **P2** |
| `tools/admin.py` | 824 | Admin and debug tools | `tool_output` | **MEDIUM** - Some Vivarium-specific | **P3** |
| `tools/doc_gen.py` | 180 | Documentation generation tool | `cache`, `doc_generation`, `tool_output` | **MEDIUM** - Depends on doc_gen | **P2** |
| `tools/browser_agent.py` | 1507 | Browser automation tool | **HEAVILY COUPLED** to `vivarium.browser` | **LOW** - Vivarium-specific | **Skip** |

**Extraction Strategy:**
- **Direct copy** for: `git.py`, `github.py`, `validation.py`, `file_ops.py`, `batch.py`
- **Refactor** for: `llm.py` - extract generic parts, leave Scout prompts
- **Skip** for: `browser_agent.py` - too coupled to Vivarium browser

---

### 3. Trust & Verification System

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `trust/orchestrator.py` | 349 | Trust orchestration | `trust/*` (self-contained) | **HIGH** - Generic trust system | **P2** |
| `trust/store.py` | 231 | Trust score persistence | `trust/models` | **HIGH** - Generic trust store | **P2** |
| `trust/verifier.py` | 218 | Trust verification | `trust/constants`, `trust/models` | **HIGH** - Generic verification | **P2** |
| `trust/auditor.py` | 216 | Trust auditing | `trust/constants` | **HIGH** - Generic auditing | **P2** |
| `trust/penalizer.py` | 148 | Trust penalty application | `trust/*` | **HIGH** - Generic penalty | **P2** |
| `trust/learner.py` | 160 | Trust learning from feedback | `trust/store`, `trust/auditor` | **HIGH** - Generic learning | **P2** |
| `trust/models.py` | 61 | Trust data models | None | **HIGH** - Generic models | **P2** |
| `trust/constants.py` | 60 | Trust constants | None | **HIGH** - Generic constants | **P2** |

**Extraction Strategy:** This module is remarkably self-contained. All dependencies are internal to `trust/`. **Direct copy** with import path changes.

---

### 4. Adaptive Engine

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `adaptive_engine/gates.py` | 332 | Threshold gates for decision making | `audit`, `llm/retry` | **HIGH** - Generic gating | **P2** |
| `adaptive_engine/blind_voting.py` | 376 | Multi-model blind voting | `adaptive_engine/gates`, `audit`, `llm/retry` | **HIGH** - Generic voting | **P2** |
| `adaptive_engine/triggers.py` | 359 | Adaptive triggers | None | **HIGH** - Generic triggers | **P2** |
| `adaptive_engine/state.py` | 168 | Adaptive state management | None | **HIGH** - Generic state | **P3** |
| `adaptive_engine/cost.py` | 164 | Adaptive cost management | None | **MEDIUM** - Cost-specific | **P2** |
| `adaptive_engine/protocols.py` | 91 | Type protocols | None | **HIGH** - Generic protocols | **P2** |

**Extraction Strategy:** Low external dependencies (mainly `audit` and `llm/retry`). **Direct copy** after extracting base modules.

---

### 5. Execution Engine

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `execution/executor.py` | 290 | Plan execution engine | `execution/actions`, `execution/registry` | **MEDIUM** - Some Scout-specific | **P2** |
| `execution/actions.py` | 171 | Execution actions | None | **HIGH** - Generic actions | **P2** |
| `execution/registry.py` | 118 | Tool registry | None | **HIGH** - Generic registry | **P2** |
| `execution/safety.py` | 304 | Execution safety | `execution/*` | **MEDIUM** - Some Scout-specific | **P3** |
| `execution/llm_prose_parser.py` | 481 | LLM prose parser | None | **MEDIUM** - Specialized | **P3** |
| `execution/mapper.py` | 109 | Action mapping | None | **MEDIUM** - Execution-specific | **P3** |

**Extraction Strategy:** Core execution is generic, but some actions may be Scout-specific. **Refactor** to extract generic core, leave Vivarium-specific actions.

---

### 6. Documentation Sync & Generation

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `doc_sync/ast_facts.py` | 888 | AST-based fact extraction | None (uses stdlib) | **VERY HIGH** - Pure AST parsing | **P1** |
| `doc_sync/synthesizer.py` | 420 | Documentation synthesis | `doc_sync/*`, `graph` | **HIGH** - Generic synthesis | **P2** |
| `doc_sync/relationship_formatter.py` | 368 | Relationship formatting | `config`, `graph` | **MEDIUM** - Some config deps | **P2** |
| `doc_generation/generator.py` | 650+ | Doc generation | `adapters/*`, `config`, `audit`, `llm` | **MEDIUM** - Depends on many | **P3** |
| `doc_generation.py` | 2195 | Main doc generation ( monolithic) | Many | **MEDIUM** - Too coupled | **Skip** |

**Extraction Strategy:**
- **Direct copy** for `ast_facts.py` - extremely generic, pure Python AST
- **Refactor** for `synthesizer.py` - extract generic parts
- **Leave monolithic `doc_generation.py`** - too coupled to Vivarium

---

### 7. Self-Improvement System

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `self_improvement/suggestion_engine.py` | 497 | Improvement suggestion engine | `self_improvement/audit_analyzer` | **MEDIUM** - Scout-specific logic | **P3** |
| `self_improvement/audit_analyzer.py` | 467 | Audit analysis for improvements | `tool_output` | **MEDIUM** - Some generic parts | **P3** |
| `self_improvement/pr_creator.py` | 489 | PR creation for improvements | `tools/base` | **MEDIUM** - GitHub-specific | **P3** |
| `self_improvement/recommender.py` | 194 | Recommendation generation | `self_improvement/analyzer` | **MEDIUM** - Scout-specific | **P3** |
| `self_improvement/improvement_tracker.py` | 254 | Track improvements | `audit`, `tool_output` | **MEDIUM** - Some generic | **P3** |
| `self_improvement/engine.py` | 122 | Improvement pipeline | `tool_output`, `validation_pipeline` | **LOW** - Too coupled | **P4** |
| `self_improvement/analyzer.py` | 114 | Analysis tools | `tool_output` | **MEDIUM** - Some generic | **P3** |
| `self_improvement/applier.py` | 87 | Apply improvements | `audit` | **MEDIUM** - Some generic | **P3** |

**Extraction Strategy:** This system is moderately coupled to Scout's audit and validation. **Refactor** to extract generic analysis parts, leave pipeline-specific logic.

---

### 8. Research Agent Swarm

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `research/synthesizer.py` | 615 | Research synthesis | `audit`, `research/agent` | **LOW** - Research-specific | **P4** |
| `research/agent.py` | 482 | Research agent | `tool_output` | **LOW** - Vivarium-specific | **P4** |
| `research/swarm_executor.py` | 376 | Swarm execution | `audit`, `research/agent`, `stealth_rotator` | **LOW** - Vivarium-specific | **Skip** |
| `research/intent_detection.py` | 286 | Research intent detection | None | **MEDIUM** - Could be generic | **P3** |
| `research/stealth_rotator.py` | 229 | Stealth rotation (agent identity) | None | **LOW** - Vivarium-specific | **Skip** |
| `research/planner.py` | 92 | Research planning | `orchestration/core` | **LOW** - Vivarium-specific | **Skip** |
| `research/command.py` | 265 | Research commands | `research/*` | **LOW** - Vivarium-specific | **Skip** |

**Extraction Strategy:** Most research components are tightly coupled to Vivarium's agent swarm. Only `intent_detection.py` has potential reuse. **Skip** most, extract `intent_detection` if useful.

---

### 9. Plan & State Management

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `plan_state.py` | 388 | Plan state management | None | **HIGH** - Generic state | **P2** |
| `plan_store.py` | 358 | Plan persistence | `trust/store` | **MEDIUM** - Depends on trust | **P2** |
| `plan_executor.py` | 310 | Plan execution | `execution/actions` | **MEDIUM** - Some generic | **P2** |
| `plan_validation.py` | 281 | Plan validation | None | **HIGH** - Generic validation | **P2** |
| `plan_capture.py` | 153 | Plan capture | None | **MEDIUM** - Some generic | **P3** |
| `plan_codegen.py` | 239 | Plan code generation | None | **MEDIUM** - Some generic | **P3** |
| `plan_pruner.py` | 212 | Plan pruning | None | **MEDIUM** - Some generic | **P3** |
| `plan_io.py` | 179 | Plan I/O | None | **MEDIUM** - Some generic | **P3** |
| `batch_plan_parser.py` | 433 | Batch plan parsing | None | **MEDIUM** - Batch-specific | **P3** |
| `batch_pipeline.py` | 362 | Batch pipeline | `batch_*` | **MEDIUM** - Batch-specific | **P3** |
| `batch_subbatch.py` | 207 | Sub-batch handling | None | **MEDIUM** - Batch-specific | **P3** |
| `batch_expression.py` | 246 | Batch expressions | None | **MEDIUM** - Batch-specific | **P3** |
| `batch_context.py` | 79 | Batch context | None | **MEDIUM** - Batch-specific | **P3** |
| `batch_path_validator.py` | 126 | Batch path validation | None | **MEDIUM** - Batch-specific | **P3** |
| `batch_cli_discovery.py` | 208 | CLI discovery for batch | None | **LOW** - CLI-specific | **P4** |

**Extraction Strategy:** Plan management is moderately generic. **Direct copy** or **refactor** depending on trust dependencies.

---

### 10. Configuration & Infrastructure

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `config.py` | 571 | Configuration management | None | **HIGH** - Generic config | **P1** |
| `audit.py` | 555 | Audit logging | None | **HIGH** - Generic audit | **P1** |
| `graph.py` | 494 | Code graph / dependency graph | None | **HIGH** - Generic graph | **P1** |
| `deps.py` | 501 | Dependency analysis | None | **HIGH** - Generic deps | **P1** |
| `cache.py` | 449 | Caching infrastructure | None | **HIGH** - Generic cache | **P1** |
| `ignore.py` | 149 | Ignore patterns (.gitignore) | None | **HIGH** - Generic ignore | **P1** |
| `tool_output.py` | 253 | Tool output formatting | None | **HIGH** - Generic output | **P1** |
| `validator.py` | 342 | Validation framework | None | **HIGH** - Generic validation | **P1** |
| `validation_pipeline.py` | 556 | Validation pipeline | `audit`, `config`, `tool_output` | **HIGH** - Generic pipeline | **P1** |
| `retry.py` | 297 | Retry utilities | None | **HIGH** - Generic retry | **P1** |
| `circuit_breaker.py` | 216 | Circuit breaker | None | **HIGH** - Generic CB | **P1** |
| `scheduler.py` | 441 | Task scheduling | None | **MEDIUM** - Scheduling-specific | **P2** |
| `context.py` | 279 | Context management | None | **MEDIUM** - Context-specific | **P2** |
| `progress.py` | 438 | Progress tracking | None | **MEDIUM** - UI-specific | **P3** |
| `session.py` | 256 | Session management | None | **MEDIUM** - Session-specific | **P3** |
| `persistence.py` | 125 | Basic persistence | None | **MEDIUM** - Some generic | **P2** |

**Extraction Strategy:** These are core infrastructure modules. Most are highly generic and self-contained. **Direct copy** for all.

---

### 11. Adapters (Language Support)

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `adapters/python.py` | 481 | Python language adapter | `adapters/base` | **VERY HIGH** - Pure AST | **P1** |
| `adapters/javascript.py` | 342 | JavaScript adapter | `adapters/base` | **HIGH** - Generic JS | **P2** |
| `adapters/base.py` | 111 | Base adapter interface | None | **VERY HIGH** - Interface | **P1** |
| `adapters/registry.py` | 99 | Adapter registry | None | **HIGH** - Generic registry | **P1** |
| `adapters/plain_text.py` | 116 | Plain text adapter | None | **HIGH** - Generic text | **P2** |

**Extraction Strategy:** Highly generic. **Direct copy** for all.

---

### 12. Analysis & Quality Gates

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `analysis/hotspots.py` | 318 | Code hotspot analysis | `audit`, `graph` | **HIGH** - Generic analysis | **P2** |
| `quality_gates.py` | 286 | Quality gates | `audit`, `config`, `tool_output`, `validation_pipeline`, `runtime.quality_gates` | **MEDIUM** - Some Vivarium deps | **P3** |
| `lint/` | ~200 | Linting modules | None | **HIGH** - Generic linting | **P2** |

**Extraction Strategy:** `hotspots.py` is highly generic. `quality_gates.py` has Vivarium runtime dependency - may need encapsulation.

---

### 13. Orchestration

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `orchestration/core.py` | 628 | Task orchestration | None | **MEDIUM** - Specialized orchestration | **P3** |

**Extraction Strategy:** Complex orchestration logic. **Refactor** if useful, otherwise skip.

---

### 14. CLI (Mostly Vivarium-Specific)

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `cli/root.py` | 862 | CLI root | Many Scout modules | **LOW** - Vivarium CLI | **Skip** |
| `cli/plan.py` | 3449 | Plan CLI (largest) | Many modules | **LOW** - Vivarium CLI | **Skip** |
| `cli/bootstrap.py` | 1273 | Bootstrap CLI | Many | **LOW** - Vivarium-specific | **Skip** |
| `cli/index.py` | 966 | Navigation/index CLI | `audit` | **LOW** - Vivarium CLI | **Skip** |
| `cli/brief.py` | 828 | Brief CLI | Many | **LOW** - Vivarium CLI | **Skip** |
| `cli/scout.py` | 746 | Scout CLI | `config` | **LOW** - Vivarium CLI | **Skip** |
| `cli/doc_sync.py` | 944 | Doc sync CLI | Many | **LOW** - Vivarium CLI | **Skip** |
| `cli/run.py` | 280 | Run CLI | `utils/summarize` | **LOW** - Vivarium CLI | **Skip** |
| Other CLI files | ~2000 | Various commands | Various | **LOW** - Vivarium-specific | **Skip** |

**Extraction Strategy:** Most CLI is Vivarium-specific. Extract only if generic command infrastructure is needed.

---

### 15. Research & Special Modules

| File | Lines | Purpose | Dependencies | Reusability | Priority |
|------|-------|---------|--------------|--------------|----------|
| `improve.py` | 894 | Improvement pipeline | Many modules | **LOW** - Vivarium-specific | **Skip** |
| `big_brain.py` | 1051 | Big brain (complex reasoning) | `audit`, `middle_manager` | **LOW** - Vivarium-specific | **Skip** |
| `middle_manager.py` | 683 | Middle management gate | `audit`, `deps`, `adaptive_engine/gates` | **LOW** - Vivarium-specific | **Skip** |
| `router.py` | 1436 | Main router (complex) | Many modules | **MEDIUM** - Some generic parts | **Refactor** |
| `git_analyzer.py` | 305 | Git analysis | None | **HIGH** - Generic git | **P2** |
| `git_drafts.py` | 328 | Git drafts | None | **MEDIUM** - Git-specific | **P3** |
| `token_estimator.py` | 142 | Token estimation | None | **HIGH** - Generic | **P2** |
| `similarity.py` | 169 | Similarity scoring | None | **HIGH** - Generic | **P2** |
| `syntax_repair.py` | 267 | Syntax repair | None | **MEDIUM** - Specialized | **P3** |
| `state_recovery.py` | 292 | State recovery | None | **MEDIUM** - Recovery-specific | **P3** |
| `resilient_llm_client.py` | 221 | Resilient LLM client | None | **HIGH** - Generic LLM | **P2** |
| `sliding_window.py` | 216 | Sliding window | None | **HIGH** - Generic | **P2** |
| `timeout_config.py` | 70 | Timeout config | None | **HIGH** - Generic | **P2** |
| `tool_circuit_breaker.py` | 77 | Tool circuit breaker | None | **HIGH** - Generic | **P2** |
| `env_validator.py` | 147 | Environment validation | None | **MEDIUM** - Env-specific | **P3** |
| `raw_briefs.py` | 77 | Raw briefs storage | None | **LOW** - Vivarium-specific | **Skip** |
| `refresh_queue.py` | 319 | Refresh queue | None | **MEDIUM** - Queue-specific | **P3** |
| `registry.py` | 186 | Generic registry | None | **HIGH** - Generic | **P1** |
| `parameter_registry.py` | 207 | Parameter registry | None | **MEDIUM** - Param-specific | **P2** |
| `base_planner.py` | 224 | Base planner | None | **MEDIUM** - Planning-specific | **P3** |
| `budget_service.py` | 352 | Budget service | None | **HIGH** - Generic budget | **P2** |

---

## Dependency Graph (Textual)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CORE INFRASTRUCTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  config.py ─────┐                                                           │
│  audit.py ──────┼──► graph.py ──► deps.py                                  │
│  cache.py ──────┤         │                                                 │
│  tool_output.py ┘         ▼                                                 │
│  validator.py ◄───────────┘                                                 │
│                                                                       ▲     │
└───────────────────────────────────────────────────────────────┬───────┘     │
                                                                │             │
                              ┌─────────────────────────────────┘             │
                              ▼                                                     │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM INFRASTRUCTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  llm/router.py ◄──── llm/cost.py ──► llm/select.py                        │
│         │                  │                                                 │
│         ▼                  ▼                                                 │
│  llm/providers/__init__.py (ProviderRegistry)                              │
│         │                                                                  │
│    ┌────┼────┬──────────┐                                                  │
│    ▼    ▼    ▼          ▼                                                  │
│  Groq Google MiniMax Anthropic                                             │
│                                                                       ▲     │
└───────────────────────────────────────────────────────────────┬───────────┘
                                                                │
                              ┌─────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TOOLS LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  tools/__init__.py (Tool Registry)                                        │
│         │                                                                   │
│    ┌────┼────┬────┬────┬─────┬──────┐                                    │
│    ▼    ▼    ▼    ▼    ▼     ▼      ▼                                    │
│ git.py github.py file_ops.py validation.py batch.py doc_gen.py          │
│                                                                       ▲     │
└───────────────────────────────────────────────────────────────┬───────────┘
                                                                │
                              ┌─────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRUST & ADAPTIVE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  trust/orchestrator.py ◄── trust/store.py ◄── trust/verifier.py          │
│         │                         │                                        │
│         ▼                         ▼                                        │
│  trust/auditor.py ────────── trust/penalizer.py                           │
│                                                                       ▲     │
│  adaptive_engine/gates.py ◄── adaptive_engine/blind_voting.py          │
│         │                                                                   │
│         ▼                                                                   │
│  adaptive_engine/triggers.py                                               │
│                                                                       ▲     │
└───────────────────────────────────────────────────────────────┬───────────┘
                                                                │
                              ┌─────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXECUTION & PLANS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  execution/executor.py ──► execution/actions.py ──► execution/registry  │
│         │                                                                   │
│         ▼                                                                   │
│  plan_executor.py ──► plan_state.py ──► plan_store.py                     │
│         │                           │                                      │
│         ▼                           ▼                                      │
│  plan_validation.py           trust/store.py                              │
│                                                                       ▲     │
└───────────────────────────────────────────────────────────────┬───────────┘
                                                                │
                              ┌─────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DOC SYNC & ADAPTERS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  doc_sync/ast_facts.py ──► doc_sync/synthesizer.py                        │
│         │                                                                   │
│         ▼                                                                   │
│  adapters/python.py ──► adapters/javascript.py                            │
│         │                                                                   │
│         ▼                                                                   │
│  adapters/registry.py                                                      │
└─────────────────────────────────────────────────────────────────────────────┘

KEY:
  ───► = imports/depends on
  ▲   = higher-level modules depend on lower
```

---

## Phased Migration Plan

### Phase 1: Core Infrastructure (Week 1-2)

**Objective:** Extract foundational modules that everything else depends on.

| Module | Files | Lines | Effort |
|--------|-------|-------|--------|
| Config & Audit | `config.py`, `audit.py` | ~1,100 | 4 hrs |
| Graph & Deps | `graph.py`, `deps.py` | ~1,000 | 4 hrs |
| Cache & Tool Output | `cache.py`, `tool_output.py` | ~700 | 3 hrs |
| Validator & Validation Pipeline | `validator.py`, `validation_pipeline.py` | ~900 | 4 hrs |
| Ignore Patterns | `ignore.py` | 150 | 1 hr |
| Registry | `registry.py` | 186 | 1 hr |

**Total Phase 1:** ~3,000 lines | **~17 hours**

---

### Phase 2: LLM Provider Infrastructure (Week 2-3)

**Objective:** Complete the LLM provider abstraction.

| Module | Files | Lines | Effort |
|--------|-------|-------|--------|
| Provider Registry | `llm/providers/__init__.py` | 350+ | 6 hrs |
| Additional Providers | `google.py`, `minimax.py`, `anthropic.py` | ~600 | 8 hrs |
| Router & Dispatch | `llm/router.py`, `llm/dispatch.py`, `llm/select.py` | ~1,000 | 8 hrs |
| Cost & Pricing | `llm/cost.py`, `llm/pricing.py` | ~320 | 3 hrs |
| Resilience | `circuit_breaker.py`, `retry.py`, `ratelimit.py` | ~500 | 4 hrs |

**Total Phase 2:** ~2,800 lines | **~29 hours**

---

### Phase 3: Tools System (Week 3-4)

**Objective:** Extract reusable tool implementations.

| Module | Files | Lines | Effort |
|--------|-------|-------|--------|
| Tool Registry | `tools/__init__.py` | 639 | 4 hrs |
| Git Tools | `tools/git.py` | 464 | 4 hrs |
| GitHub Tools | `tools/github.py` | 511 | 4 hrs |
| Validation Tools | `tools/validation.py` | 282 | 3 hrs |
| File Operations | `tools/file_ops.py` | 390 | 3 hrs |
| Batch Tools | `tools/batch.py` | 214 | 2 hrs |
| LLM Tools (refactor) | `tools/llm.py` | 631 | 6 hrs |

**Total Phase 3:** ~3,100 lines | **~26 hours**

---

### Phase 4: Trust & Adaptive Engine (Week 4-5)

**Objective:** Extract decision-making infrastructure.

| Module | Files | Lines | Effort |
|--------|-------|-------|--------|
| Trust System | `trust/*.py` (8 files) | ~1,400 | 12 hrs |
| Adaptive Gates | `adaptive_engine/gates.py` | 332 | 4 hrs |
| Blind Voting | `adaptive_engine/blind_voting.py` | 376 | 4 hrs |
| Triggers & State | `adaptive_engine/triggers.py`, `state.py` | ~530 | 4 hrs |

**Total Phase 4:** ~2,600 lines | **~24 hours**

---

### Phase 5: Execution & Plans (Week 5-6)

**Objective:** Extract execution and plan management.

| Module | Files | Lines | Effort |
|--------|-------|-------|--------|
| Execution Core | `execution/executor.py`, `actions.py`, `registry.py` | ~580 | 6 hrs |
| Execution Safety | `execution/safety.py`, `llm_prose_parser.py` | ~780 | 8 hrs |
| Plan State & Store | `plan_state.py`, `plan_store.py`, `plan_validation.py` | ~1,000 | 8 hrs |
| Plan Execution | `plan_executor.py`, `plan_capture.py`, `plan_codegen.py` | ~700 | 6 hrs |
| Batch Pipeline | `batch_*.py` (7 files) | ~1,600 | 12 hrs |

**Total Phase 5:** ~4,600 lines | **~40 hours**

---

### Phase 6: Documentation & Adapters (Week 6-7)

**Objective:** Extract doc generation and language adapters.

| Module | Files | Lines | Effort |
|--------|-------|-------|--------|
| AST Fact Extraction | `doc_sync/ast_facts.py` | 888 | 6 hrs |
| Doc Synthesizer | `doc_sync/synthesizer.py`, `relationship_formatter.py` | ~800 | 8 hrs |
| Python Adapter | `adapters/python.py` | 481 | 4 hrs |
| JS Adapter | `adapters/javascript.py` | 342 | 3 hrs |
| Adapter Base & Registry | `adapters/base.py`, `registry.py` | 210 | 2 hrs |

**Total Phase 6:** ~2,700 lines | **~23 hours**

---

### Phase 7: Analysis & Utilities (Week 7-8)

**Objective:** Extract analysis tools and utilities.

| Module | Files | Lines | Effort |
|--------|-------|--------|--------|
| Code Hotspots | `analysis/hotspots.py` | 318 | 3 hrs |
| Budget Service | `budget_service.py` | 352 | 3 hrs |
| Token Estimation | `token_estimator.py` | 142 | 1 hr |
| Similarity | `similarity.py` | 169 | 2 hrs |
| Resilient LLM | `resilient_llm_client.py` | 221 | 2 hrs |
| Sliding Window | `sliding_window.py` | 216 | 2 hrs |
| Various Utils | `timeout_config.py`, `tool_circuit_breaker.py`, `parameter_registry.py`, etc. | ~600 | 5 hrs |

**Total Phase 7:** ~2,000 lines | **~18 hours**

---

### Phase 8: Self-Improvement (Week 8-9)

**Objective:** Extract self-improvement components if valuable.

| Module | Files | Lines | Effort |
|--------|-------|-------|--------|
| Analyzer | `self_improvement/analyzer.py`, `audit_analyzer.py` | ~580 | 8 hrs |
| Recommender | `self_improvement/recommender.py`, `suggestion_engine.py` | ~700 | 10 hrs |
| Tracker & Applier | `self_improvement/improvement_tracker.py`, `applier.py` | ~340 | 4 hrs |

**Total Phase 8:** ~1,600 lines | **~22 hours**

---

## Effort Summary by Phase

| Phase | Description | Lines | Hours |
|-------|-------------|-------|-------|
| 1 | Core Infrastructure | ~3,000 | 17 |
| 2 | LLM Providers | ~2,800 | 29 |
| 3 | Tools System | ~3,100 | 26 |
| 4 | Trust & Adaptive | ~2,600 | 24 |
| 5 | Execution & Plans | ~4,600 | 40 |
| 6 | Docs & Adapters | ~2,700 | 23 |
| 7 | Analysis & Utils | ~2,000 | 18 |
| 8 | Self-Improvement | ~1,600 | 22 |
| | **TOTAL** | **~22,400** | **~199** |

---

## Recommendations for DataClaw

### Top Priority Extractions (Immediate Value)

1. **Provider Registry + Multi-Provider Support**
   - Already partially done for Groq
   - Add Google, MiniMax, Anthropic
   - Enables DataClaw to use multiple LLM backends

2. **Trust System**
   - Self-contained, high value
   - Enables quality scoring for LLM outputs
   - Can be used for any agent system

3. **AST Fact Extraction (`doc_sync/ast_facts.py`)**
   - Pure Python stdlib, no dependencies
   - Enables grounded documentation generation
   - Very high reusability

4. **Tool System (git, github, validation)**
   - Already well-structured
   - Enables DataClaw to interact with repositories

5. **Graph & Dependency Analysis**
   - Already extracted in scout-core
   - Enable code understanding

### Modules to Leave in Vivarium

- **Browser Agent** - Too coupled to `vivarium.browser`
- **CLI Commands** - Vivarium-specific workflows
- **Research Swarm** - Vivarium-specific agent behavior
- **Self-Improvement Pipeline** - Too coupled to Vivarium's audit/validation
- **Monolithic `doc_generation.py`** - Too coupled, should be refactored separately
- **`big_brain.py`, `middle_manager.py`** - Vivarium-specific orchestration

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **Vivarium runtime deps** in some modules | Use adapter pattern; create thin Vivarium-specific wrappers |
| **Circular dependencies** (trust ↔ plan_store) | Extract in order; resolve cycles by introducing abstraction layers |
| **Configuration coupling** (`config.py`) | Extract generic config, leave Vivarium-specific in wrapper |
| **Audit dependency everywhere** | Already extracted; ensure it's stable first |
| **Large monolithic files** (`doc_generation.py` 2195 lines) | Refactor before extraction; split into smaller modules |

---

## Conclusion

The Vivarium codebase contains approximately **22,400 lines** of extractable, reusable Scout infrastructure across **8 phases** spanning **~9 weeks** of effort. The highest-value immediate extractions for DataClaw are:

1. **Complete LLM provider infrastructure** (Phase 2)
2. **Trust system** (Phase 4)
3. **AST fact extraction** (Phase 6)
4. **Tool system** (Phase 3)

By following this phased approach, DataClaw can incrementally adopt Scout's infrastructure while maintaining a clean separation from Vivarium-specific components.
