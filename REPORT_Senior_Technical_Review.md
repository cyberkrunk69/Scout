# Senior Technical Review Report: Scout-Core Infrastructure Update

**Date:** February 25, 2026  
**Commit:** `dab1ea5` (pushed to origin/main)  
**Reviewer:** Technical Senior Review  

---

## Executive Summary

This report documents the comprehensive infrastructure updates made to the Scout repository over the past 6 hours. The implementation adds substantial new functionality including an execution engine, plan executor, batch pipeline, config system, dataclaw modules, and extensive documentation. The codebase is **functional and importable** with browser functionality intentionally stubbed.

**Key Finding:** The implementation is approximately **85% complete** with several areas of dead code, missing tests, and incomplete documentation. There are also some hard-coded values that should be centralized.

---

## 1. What Was Done

### 1.1 Core Infrastructure Added

| Module | Files | Purpose | Status |
|--------|-------|---------|--------|
| `execution/` | 5 files | Execution engine with actions, safety, llm_prose_parser, mapper, registry | ✅ Functional |
| `plan_executor.py` | 312 lines | WebPlanExecutor for multi-step web automation | ⚠️ Dead code (never imported) |
| `batch_*` | 7 files | Batch pipeline, context, expression, path validation, subbatch, plan parser | ✅ Functional |
| `config/` | 2 files | Config system with defaults | ✅ Functional |
| `dataclaw/` | 5 files | VSCode data ingestion and parsing | ⚠️ Not wired up |
| `vscode_parsers.py` | 680+ lines | VSCode session parsing | ⚠️ Not wired up |
| `vscode_storage.py` | 200+ lines | VSCode storage utilities | ⚠️ Not wired up |
| `llm/` updates | Multiple | Circuit breaker, dispatch, pricing, providers, exampleprovider | ✅ Functional |

### 1.2 Documentation Added

- **10 ADRs** (Architecture Decision Records) covering:
  - Provider registry
  - Router/dispatch/select separation
  - Budget service reservation semantics
  - Trust system design
  - Circuit breaker pattern
  - Execution framework
  - Plan executor state machine
  - Batch pipeline
  - Caching strategy
  - Retry mechanisms

- **API Documentation** (`docs/api/`):
  - `scout.md` - Core functionality
  - `scout.llm.md` - LLM integration
  - `scout.execution.md` - Plan execution
  - `scout.trust.md` - Trust system

- **Guides**:
  - `adding_a_provider.md` - 936 lines
  - `CONTRIBUTING.md`
  - `testing.md`

### 1.3 Testing Added

- 54 test files added covering:
  - LLM providers, budget, circuit breaker, dispatch, retry
  - Execution executor
  - Batch pipeline
  - Trust system (store, verifier, orchestrator)
  - VSCode parsers
  - Hotspots analysis

---

## 2. Deviations from Expected Implementation

### 2.1 Dead Code / Unused Modules

| Module | Lines | Issue |
|--------|-------|-------|
| `plan_executor.py` | 312 | `WebPlanExecutor` class is never imported anywhere in the codebase |
| `dataclaw/` | ~400 | Parser modules are not imported outside their own package |
| `vscode_parsers.py` | 680+ | Not imported anywhere in the codebase |
| `vscode_storage.py` | 200+ | Not imported anywhere in the codebase |

**Impact:** Approximately **1,600+ lines of code** that are shipped but not connected to anything. This increases the package size without providing value.

### 2.2 Stubs Present

1. **`src/scout/browser/__init__.py`** - ✅ Intentional stub (raises ImportError)
   ```python
   raise ImportError(
       "Browser support is not available in this version. "
       "If you need browser tools, install scout-core with the 'browser' extra"
   )
   ```

2. **`src/scout/progress.py`** - ⚠️ Stub that pretends to be functional
   - Contains `Status` enum, `ProgressEvent` dataclass, `ProgressReporter` class
   - But only stores events in memory (no actual reporting)
   - Marked as stub but ships as functional-looking code

3. **`src/scout/router.py:1148`** - `create_human_escalation_ticket()` is a stub method

### 2.3 Dependency Issue Fixed

- **`watchdog`** was in `dev` dependencies but imported unconditionally in `cache.py`
- Fixed by moving to main dependencies and removing from dev

---

## 3. Code Quality Issues

### 3.1 Hard-Coded Values / Magic Numbers

The following should be centralized in `config/defaults.py`:

| Location | Value | Should Be |
|----------|-------|-----------|
| `router.py:70` | `estimated_cost: float = 0.01` | Config |
| `router.py:108-109` | `print("1. Wait for next hour...")` hardcoded messages | i18n or config |
| `router.py:206` | `base_cost * 1.2` hardcoded 20% buffer | Config |
| `router.py:499` | `NAV_DEFAULT_CONFIDENCE + 10` hardcoded boost | Config |
| `search.py:330,586` | `gap_top2 ** 0.5 * 40 - 8` magic formula | Documented constant |
| `execution/safety.py:344` | `await asyncio.sleep(1)` hardcoded | Config |
| `llm_prose_parser.py:191` | `sleep_time = 2 ** attempt` hardcoded backoff | Config |
| `batch_cli_discovery.py:65` | `timeout=10` hardcoded | Config |
| `tools/file_ops.py:154,239,350` | `timeout=300,120,120` hardcoded | Config |

### 3.2 TODO Comments Found (15 instances)

```
src/scout/trust/orchestrator.py:314       - TODO: Import doc sync when available
src/scout/graph.py:405                    - TODO: Phase 2 - extract doc_sync
src/scout/tools/__init__.py:477,503,509  - TODO: Extract LLM/Doc gen/Admin tools
src/scout/validator.py:20                 - TODO: Phase 2 - modules not extracted
src/scout/middle_manager.py:20            - TODO: Phase 2 - modules not extracted
src/scout/deps.py:106                     - TODO: Phase 2 - extract adapters
src/scout/router.py:788,1148              - TODO: Phase 2 - trust.auditor, stub ticket
src/scout/progress.py:1                   - Stub implementation (by design)
```

---

## 4. Test Coverage Analysis

### 4.1 Areas WITH Tests

| Area | Test Files | Coverage |
|------|-----------|----------|
| LLM (providers, budget, circuit breaker, dispatch, retry) | 12+ | ✅ Good |
| Execution executor | 1 | ✅ Basic |
| Batch pipeline | 2 | ✅ Basic |
| Trust system | 3 | ✅ Basic |
| Cache | 1 | ✅ Basic |
| VSCode parsers | 1 | ✅ Basic |

### 4.2 Areas WITHOUT Tests

| Module | Lines | Priority |
|--------|-------|----------|
| `plan_executor.py` (dead code anyway) | 312 | N/A |
| `dataclaw/` | ~400 | Low (not wired) |
| `vscode_parsers.py` | 680+ | Low (not wired) |
| `vscode_storage.py` | 200+ | Low (not wired) |
| `batch_cli_discovery.py` | 200 | Medium |
| `batch_context.py` | 40 | Medium |
| `batch_expression.py` | 100+ | Medium |
| `batch_path_validator.py` | 126 | Medium |
| `plan_capture.py` | 153 | Medium |
| `plan_codegen.py` | 239 | Medium |
| `plan_io.py` | 179 | Medium |
| `plan_pruner.py` | 212 | Medium |
| `plan_state.py` | 424 | Medium |
| `plan_store.py` | 375 | Medium |
| `plan_validation.py` | 281 | Medium |
| `progress.py` (stub) | 73 | N/A |
| `ast_facts.py` | 900+ | Low (utility) |

### 4.3 Test Quality Observations

- Most tests are **basic smoke tests** (import + simple call)
- Few tests verify actual behavior with assertions
- No property-based testing
- No performance/load tests (beyond `test_batch_performance.py`)
- Integration tests are minimal

---

## 5. Documentation Coverage

### 5.1 Documentation Present

| Type | Status | Notes |
|------|--------|-------|
| ADRs | ✅ Complete | 10 comprehensive ADRs |
| API docs | ✅ Basic | Docstrings present, mkdocs configured |
| Guides | ✅ Complete | Adding provider guide is 936 lines |
| README | ✅ Updated | Current |
| CONTRIBUTING | ✅ Added | New |

### 5.2 Documentation Gaps

| Area | Gap |
|------|-----|
| `dataclaw/` | No usage documentation (module not wired) |
| `vscode_parsers.py` | No usage documentation (module not wired) |
| `execution/` | Basic docstrings, no tutorial |
| `batch_pipeline.md` | ADR exists but no user guide |
| `plan_executor.md` | ADR exists but no user guide |
| Configuration | No guide on how to configure scout-core |

---

## 6. Maintainability Assessment

### 6.1 Positive Aspects

1. **Centralized configuration** - `config/defaults.py` contains ~280 lines of named constants
2. **Consistent patterns** - All providers follow similar structure
3. **Type hints** - Extensive use of typing throughout
4. **Docstrings** - Most public APIs have docstrings
5. **Error handling** - Consistent use of custom exceptions
6. **Logging** - Structured logging throughout

### 6.2 Concerns

1. **Dead code** - 1,600+ lines shipped but unused
2. **Inconsistent config** - Some values still hardcoded
3. **Large modules** - `router.py` is 1,700+ lines
4. **Complex dependencies** - Some circular or unclear imports
5. **Missing cleanup** - Deleted files (`tool_circuit_breaker.py`, `config.py`) not fully replaced

---

## 7. Is This "Full Assed"?

**Short Answer: No, approximately 85% complete.**

### What's Done Well (✅)
- Core LLM infrastructure is solid
- Browser stub works correctly
- Package installs and imports correctly
- Documentation framework is in place
- Basic test coverage exists

### What's Missing/Incomplete (⚠️)
1. **Dead code shipped** - 1,600+ lines of unused code
2. **Not wired up** - dataclaw, vscode_parsers, vscode_storage not connected
3. **Magic numbers** - ~15+ hardcoded values scattered throughout
4. **Test gaps** - Many new modules lack tests
5. **Documentation gaps** - User guides for new features missing
6. **WebPlanExecutor** - 312 lines of dead code for browser automation

---

## 8. Recommendations for Production Readiness

### Priority 1: Remove Dead Code
```bash
# Delete unused modules
rm src/scout/plan_executor.py  # Or wire it up
# OR wire it up if browser support is planned
```

### Priority 2: Centralize Configuration
Move remaining hardcoded values to `config/defaults.py`

### Priority 3: Wire Up or Remove Unused Modules
- Either integrate dataclaw, vscode_parsers, vscode_storage
- Or remove them with clear comments about future plans

### Priority 4: Add Tests
Focus on: batch_pipeline, execution, plan_state, config

### Priority 5: Documentation
Add user-facing guides for:
- Batch pipeline usage
- Plan execution
- Configuration options

---

## 9. Conclusion

The Scout-Core implementation is **functional and shippable** with the current state being a solid foundation. However, it's not "full assed" due to:

1. Significant dead code (~1,600 lines)
2. Unwired new modules
3. Magic numbers scattered throughout
4. Incomplete test coverage
5. Missing user documentation

**Verdict:** Ship as MVP, with a roadmap to clean up dead code and complete wiring within 2-3 sprints.

---

* 25, Report generated: February2026*
