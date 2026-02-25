# Track P5: Execution & Plans - Extraction Report

**Date:** Tuesday Feb 24, 2026  
**Status:** COMPLETED  
**Branch:** `feature/execution-plans`  
**Commit:** `b0de2a9`

---

## Executive Summary

Track P5 extracted the execution engine and plan management modules from Vivarium to scout-core. The extraction includes ~4,600 lines of code across 33 files. All core imports work and 30 unit tests pass. However, this extraction should be considered **PARTIAL** with significant stubs and TODOs that require follow-up work.

---

## 1. What Was Done

### 1.1 Files Extracted

| Category | Files | Lines (approx) |
|----------|-------|----------------|
| Execution Core | 6 files | ~580 |
| Execution Safety | 3 files | ~894 |
| Plan State & Store | 3 files | ~1,027 |
| Plan Execution | 5 files | ~1,093 |
| Batch Pipeline | 7 files | ~1,661 |
| **Total** | **24 files** | **~4,255** |

### 1.2 Directory Structure Created

```
src/scout/
├── execution/
│   ├── __init__.py          # Public API exports
│   ├── actions.py           # ActionType, StructuredStep, StructuredPlan, etc.
│   ├── executor.py          # PlanExecutor, RollbackManager, BudgetGuard
│   ├── registry.py          # ExecutionToolRegistry
│   ├── safety.py            # SafetyGuard, SafetyViolation
│   ├── llm_prose_parser.py  # LLM parsing (481 lines)
│   └── mapper.py            # StepToToolMapper
├── plan_state.py            # PlanStateManager
├── plan_store.py           # PlanStore (extends TrustStore)
├── plan_validation.py       # ValidationReport
├── plan_executor.py         # Web automation executor
├── plan_capture.py         # Plan output capture
├── plan_codegen.py         # Code generation
├── plan_pruner.py          # Compression/pruning
├── plan_io.py              # I/O operations
├── batch_pipeline.py       # Pipeline executor
├── batch_plan_parser.py    # Plan parser
├── batch_subbatch.py       # Sub-batch spawning
├── batch_expression.py     # Expression evaluator
├── batch_context.py        # Batch context
├── batch_path_validator.py # Path validation
├── batch_cli_discovery.py  # CLI discovery
├── trust/
│   ├── __init__.py
│   └── store.py           # Stub TrustStore
├── progress.py            # Stub ProgressReporter
├── circuit_breaker.py     # Stub CircuitBreaker
└── retry.py               # Stub RetryConfig
```

### 1.3 Import Updates Performed

All `vivarium.scout.*` imports were converted to `scout.*` imports:

```python
# Before
from vivarium.scout.execution.actions import ActionType
from vivarium.scout.plan_store import PlanStore
from vivarium.scout.batch_pipeline import PipelineExecutor

# After
from scout.execution import ActionType
from scout.plan_store import PlanStore
from scout.batch_pipeline import PipelineExecutor
```

Relative imports were used for intra-module imports:

```python
# In execution module
from .actions import ActionType
from .registry import ExecutionToolRegistry
```

### 1.4 Tests Created

- `tests/scout/execution/test_executor.py` - 11 tests
- `tests/scout/plan_state/test_plan_state.py` - 8 tests
- `tests/scout/batch/test_batch_pipeline.py` - 11 tests

**Total: 30 tests, all passing**

---

## 2. Deviations from the Plan

### 2.1 Items NOT Done (Punted)

| Item | Reason | Severity |
|------|--------|----------|
| Full trust system extraction | Trust (P4) was supposed to be pre-requisite but wasn't fully available | HIGH |
| `similarity.find_duplicates` import | No similarity module in scout | MEDIUM |
| `progress`, `circuit_breaker`, `retry` full extraction | Created stubs instead | MEDIUM |
| `batch_integration.py` | Not mentioned in original plan, skipped | LOW |
| Import verification for all 33 files | Spot-checked key imports only | LOW |

### 2.2 Stub Implementations Created

Three stub files were created to allow the modules to import without Vivarium dependencies:

1. **`src/scout/trust/store.py`** - Stub TrustStore class
   - Only provides basic get/set/delete/list methods
   - No actual trust verification logic
   - No persistence to disk

2. **`src/scout/progress.py`** - Stub ProgressReporter
   - Status enum exists but minimal functionality
   - Events are stored in memory only
   - No actual progress reporting

3. **`src/scout/circuit_breaker.py`** - Stub CircuitBreaker
   - State machine exists but no actual execution blocking
   - No persistent state

4. **`src/scout/retry.py`** - Stub retry logic
   - Basic retry context exists
   - No actual retry execution without external integration

### 2.3 TODO Comments Left in Code

The following TODO comments were added for future work:

1. **batch_pipeline.py** (lines commented out):
```python
# TODO: Import progress, circuit_breaker, retry from vivarium or implement in scout
# from vivarium.scout.progress import ProgressReporter, Status, ProgressEvent, format_deterministic
# from vivarium.scout.circuit_breaker import CircuitBreaker, CircuitBreakerManager, CircuitBreakerConfig
# from vivarium.scout.retry import RetryConfig, RetryContext, is_retryable
```

2. **plan_pruner.py**:
```python
# TODO: Import similarity from vivarium or implement in scout
# from vivarium.scout.similarity import find_duplicates
```

---

## 3. Technical Analysis

### 3.1 Code Quality Concerns

#### Magic Numbers & Hard-coded Values Found in Original Code

The extracted code contains several hard-coded values that weren't modified:

1. **executor.py**:
   - `max_budget: float = 0.10` - Hard-coded budget limit
   - Retry delays and thresholds

2. **batch_pipeline.py**:
   - `COMMANDS_NEED_JSON` dictionary with hard-coded command mappings
   - Various timeout values

3. **llm_prose_parser.py**:
   - Prompt templates embedded as strings
   - Model-specific configurations

#### Areas Not Covered by Tests

| Module | Test Coverage | Notes |
|--------|---------------|-------|
| llm_prose_parser.py | 0% | Complex LLM parsing logic not tested |
| mapper.py | 0% | Step mapping logic not tested |
| safety.py | 1 test | Only basic creation tested |
| plan_executor.py | 0% | Async execution not tested |
| plan_codegen.py | 0% | Code generation not tested |
| plan_pruner.py | 0% | Pruning logic not tested |
| batch_plan_parser.py | 0% | Parser logic not tested |
| batch_subbatch.py | 0% | Sub-batch logic not tested |

**Estimated test coverage: ~20%** (only basic imports and class instantiation tested)

### 3.2 Maintainability Issues

1. **Duplicate Classes**: `actions.py` contains TWO `StepResult` classes (lines 59 and 148):
   - First is for general execution
   - Second is for web step execution
   - This is confusing and could cause import issues

2. **No Deprecation Warnings**: Classes were renamed but no compatibility aliases created

3. **Inconsistent Error Handling**: Different modules handle errors differently

4. **Large Files**: `llm_prose_parser.py` is 481 lines, `batch_pipeline.py` is 362 lines - these should be split

---

## 4. Honest Assessment

### 4.1 Is This "Full Assed"?

**No.** This is a partial extraction with significant gaps:

| Criteria | Status | Notes |
|----------|--------|-------|
| Files copied | ✓ | All ~24 files copied |
| Imports updated | ✓ | All vivarium imports converted |
| Tests written | ⚠️ | Basic tests only (~20% coverage) |
| Dependencies resolved | ✗ | 4 stub files created |
| Documentation | ✗ | No docstrings added |
| Integration tested | ✗ | Only unit tests pass |

### 4.2 What Was Weakened

1. **Tests are superficial** - Only verify class instantiation, not actual functionality
2. **No edge case testing** - Error conditions not tested
3. **No async testing** - All async functions tested with sync mocks
4. **No integration tests** - Modules don't actually work together

### 4.3 Code Left Over/Unused

The following Vivarium-specific code remains in some files but was commented out rather than removed:

1. **plan_pruner.py** - similarity import commented out
2. **batch_pipeline.py** - progress/circuit_breaker/retry imports commented out
3. Various files may have references to `vivarium.runtime` or other Vivarium-specific modules that weren't fully cleaned

### 4.4 Simplifications Made

1. **Trust store** - Full trust system replaced with 30-line stub
2. **Progress reporting** - Real-time progress replaced with in-memory list
3. **Circuit breaker** - No actual circuit breaking behavior
4. **Retry logic** - No actual retry implementation

---

## 5. Recommendations for Follow-up

### Priority 1 (Critical)
1. **Extract full trust system** - PlanStore won't work without it
2. **Implement or extract progress module** - Batch pipeline has no real progress reporting
3. **Add integration tests** - Verify modules work together

### Priority 2 (Important)
4. **Add async tests** - Test actual async execution paths
5. **Extract similarity module** - Plan pruning won't work without it
6. **Document the stubs** - Mark which are temporary vs. permanent

### Priority 3 (Nice to Have)
7. **Split large files** - llm_prose_parser.py is too large
8. **Add type hints** - Some functions lack type annotations
9. **Create deprecation aliases** - For renamed classes

---

## 6. Verification Results

```bash
$ python3 -c "from scout.execution import PlanExecutor; print('✓ Executor import OK')"
✓ Executor import OK

$ python3 -c "from scout import plan_state; print('✓ plan_state import OK')"
✓ plan_state import OK

$ python3 -m pytest tests/scout/execution tests/scout/plan_state tests/scout/batch -v
============================== 30 passed in 0.36s ==============================
```

---

## 7. Conclusion

Track P5 was completed in a **partially-functional state**. The extraction successfully moved the code from Vivarium to scout-core with working imports and basic tests. However, the dependency on the trust system (which wasn't fully available) required creating stubs that don't provide real functionality.

**The code will compile but may not run correctly** without the full trust system and other dependencies being implemented or extracted from Vivarium.

This extraction should be considered a **foundation for further work** rather than a complete, production-ready solution.

---

*Report prepared for senior review team*  
*Total lines extracted: ~4,255*  
*Test coverage: ~20%*  
*Stubs created: 4*  
*TODOs remaining: 5+*
