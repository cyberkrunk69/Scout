# Technical Due Diligence Report
## feature/execution-plans Branch Merge (Commit b0de2a9)

**Review Date:** February 24, 2026  
**Reviewer:** Senior Technical Review Team  
**Status:** MERGED INTO MAIN - Requires Follow-up Work

---

## 1. Executive Summary

The `feature/execution-plans` branch has been successfully merged into `main`. This report provides a comprehensive technical review of the 25 new files introduced by the merge, encompassing the execution engine, batch processing, and plan management modules.

### Verdict: **NOT PRODUCTION READY** - Significant technical debt identified

The codebase demonstrates rapid prototyping with numerous stubs, hardcoded values, incomplete implementations, and minimal test coverage. While the architecture shows promise, **significant rework is required** before this can be considered production-grade software.

---

## 2. Files Added

### Execution Module (7 files)
| File | Lines | Purpose |
|------|-------|---------|
| `execution/__init__.py` | 72 | Module exports |
| `execution/actions.py` | 181 | Action types, step/plan definitions |
| `execution/executor.py` | 318 | Plan execution engine |
| `execution/llm_prose_parser.py` | 565 | LLM prose parsing |
| `execution/mapper.py` | 116 | Step-to-tool mapping |
| `execution/registry.py` | 161 | Tool registry |
| `execution/safety.py` | 342 | Execution safety guards |

### Batch Module (7 files)
| File | Lines | Purpose |
|------|-------|---------|
| `batch_cli_discovery.py` | 245 | CLI command discovery |
| `batch_context.py` | 81 | Execution context |
| `batch_expression.py` | 285 | Expression evaluation |
| `batch_path_validator.py` | 136 | Path validation |
| `batch_pipeline.py` | 457 | Batch pipeline orchestration |
| `batch_plan_parser.py` | 524 | Plan parsing |
| `batch_subbatch.py` | 231 | Subbatch handling |

### Plan Module (8 files)
| File | Lines | Purpose |
|------|-------|---------|
| `plan_capture.py` | 137 | Plan capture |
| `plan_codegen.py` | 277 | Code generation |
| `plan_executor.py` | 345 | Plan execution (DUPLICATE) |
| `plan_io.py` | 165 | Plan I/O |
| `plan_pruner.py` | 198 | Plan pruning |
| `plan_state.py` | 485 | State management |
| `plan_store.py` | 397 | Plan storage |
| `plan_validation.py` | 312 | Plan validation |

### Test Files (3 files)
| File | Tests | Coverage |
|------|-------|----------|
| `test_executor.py` | 11 | ~20% |
| `test_batch_pipeline.py` | 19 | ~5% |
| `test_plan_state.py` | 9 | ~10% |

---

## 3. Critical Issues Found

### 3.1 DUPLICATE CLASS DEFINITIONS (BUG)

**Location:** `execution/actions.py` lines 58-66 and 148-159

```python
# First definition (lines 58-66)
@dataclass
class StepResult:
    step_id: str
    status: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Second definition (lines 148-159) - OVERWRITES FIRST
@dataclass  
class StepResult:
    step_id: str
    action_type: str
    status: str
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    web_status_code: Optional[int] = None
```

**Impact:** The second `StepResult` class completely overwrites the first. Any code expecting the first definition will receive runtime errors.

### 3.2 DUPLICATE PLANEXECUTOR CLASSES (ARCHITECTURAL)

**Locations:**
- `execution/executor.py` - `PlanExecutor` 
- `plan_executor.py` - Another `PlanExecutor`

Two separate `PlanExecutor` classes exist with different interfaces. This violates DRY and will cause confusion.

### 3.3 RACE CONDITIONS (CONCURRENCY BUG)

**Location:** `plan_state.py` lines 61-85

```python
async def acquire_lock(self, plan_id: str) -> bool:
    lock_path = self.locks_dir / f"{plan_id}.lock"
    # RACE CONDITION: Between exists() and mkdir()
    if not lock_path.exists():
        try:
            lock_path.mkdir(parents=True, exist_ok=True)
            return True
        except FileExistsError:
            return False
```

**Impact:** Two concurrent requests can both pass the `exists()` check before either creates the directory.

### 3.4 SQL INJECTION VULNERABILITY (SECURITY)

**Location:** `plan_store.py` line 260

```python
await conn.execute(
    f"UPDATE plans SET {column} = {column} + 1 WHERE plan_id = ?",
    (plan_id,)
)
```

**Impact:** String formatting in SQL allows column name injection. While `column` is controlled internally, this is a dangerous pattern.

### 3.5 PATH TRAVERSAL (SECURITY)

**Location:** `safety.py` line 25

```python
def validate_path(self, path: Path, base: Path) -> bool:
    resolved = path.resolve()
    return str(resolved).startswith(str(base.resolve()))
```

**Impact:** Can be bypassed with symlinks. Should use `os.path.realpath()` or check actual inode.

### 3.6 STUB IMPLEMENTATIONS

| Location | Stub | Status |
|----------|------|--------|
| `executor.py:257` | `execute_prose()` | Returns hardcoded JSON |
| `executor.py:63` | `generate_undo_plan()` | Just `pass` |
| `llm_prose_parser.py:281` | `scout_wait` condition | TODO comment |
| `plan_pruner.py:20` | `find_duplicates()` | Imports non-existent module |

---

## 4. Hardcoded Values & Magic Numbers

### Configuration Hardcoded (Should Be Externalized)

| Location | Value | Should Be |
|----------|-------|-----------|
| `executor.py:235` | `estimated_cost = 0.001` | Calculated |
| `executor.py:37` | `timeout_seconds = 300` | Configurable |
| `batch_subbatch.py:75` | `MAX_DEPTH = 3` | Config |
| `batch_subbatch.py:88` | `".venv/bin/python"` | Path discovery |
| `batch_cli_discovery.py:11-15` | Static command list | Discovered |
| `batch_pipeline.py:35-54` | Commands need JSON | Config |
| `plan_io.py:14-17` | Hardcoded paths | Config |
| `safety.py:49-52` | Command whitelist | Config |
| `plan_validation.py:95` | Critical action types | Config |

---

## 5. Test Coverage Analysis

### Current Coverage: ~10% (Estimated)

### Tests Are Sanity Checks Only

**Example from `test_batch_pipeline.py`:**
```python
def test_pipeline_executor_exists(self):
    from scout.batch_pipeline import PipelineExecutor
    assert PipelineExecutor is not None
```

These tests verify imports work, not actual functionality.

### Missing Test Coverage

1. **Execution flow**: No end-to-end execution tests
2. **Error handling**: No tests for failure modes
3. **Concurrency**: No locking/race condition tests
4. **Expressions**: Complex interpolation logic untested
5. **State transitions**: Plan state machine untested
6. **Integration**: No integration tests between modules

---

## 6. Code Quality Issues

### TODO/FIXME Comments Found
- `llm_prose_parser.py:281` - Condition waiting unimplemented
- `plan_pruner.py:20-21` - Missing similarity module import

### Print Statements (Should Use Logging)
- `plan_capture.py:111, 117, 138`

### Inconsistent Error Handling
- `mapper.py`: Returns `None` instead of raising exceptions
- `plan_capture.py`: Validation errors print but don't raise

### Complex Conditional Logic
- `batch_expression.py:92-124`: 10+ nested conditionals in `_eval_interpolation`
- `batch_plan_parser.py:288-344`: 20+ sequential if statements for command inference

---

## 7. Documentation Gaps

### Missing Module Docstrings
- `batch_pipeline.py`
- `batch_context.py`
- `plan_executor.py`

### Unclear Purposes
- `plan_codegen.py`: Unclear if for code generation FROM plans or FOR plans
- `plan_executor.py`: Duplicate of `executor.py` - relationship unclear

### Incomplete Type Hints
- Extensive use of `Any` throughout
- Missing return types on some methods

---

## 8. Maintainability Assessment

### DRY Violations
1. Two `PlanExecutor` classes
2. Two `StepResult` classes  
3. Command mappings duplicated in 4+ files
4. Hardcoded paths duplicated

### Code Organization Issues
- Global mutable state (`_CLI_CACHE` in `batch_cli_discovery.py`)
- Mixed async/sync patterns (`plan_state.py` uses both `asyncio.Lock` and `threading.Lock`)
- Circular import risks (`batch_pipeline.py:209`)

---

## 9. Recommendations

### P0 - Must Fix Before Production

1. **Fix duplicate class definitions** - Consolidate `StepResult` and `PlanExecutor`
2. **Fix race conditions** - Use atomic file operations for locking
3. **Fix SQL injection** - Use parameterized queries
4. **Fix path traversal** - Use proper path resolution

### P1 - Should Fix Before Production

1. **Externalize configuration** - Move hardcoded values to config
2. **Add comprehensive tests** - Target 80% coverage
3. **Implement stub functions** - Complete or remove TODO items
4. **Add proper logging** - Replace print statements

### P2 - Nice to Have

1. **Document modules** - Add docstrings
2. **Simplify conditionals** - Extract to helper methods
3. **Type annotations** - Replace `Any` with specific types

---

## 10. Conclusion

The merge was successful from a git operations perspective, but **this code is NOT production ready**. The codebase shows signs of rapid prototyping with:

- **~3 critical bugs** (duplicates, race conditions, security)
- **~5 stub implementations** 
- **~25+ hardcoded values**
- **<10% test coverage**
- **Significant technical debt**

**Verdict: "Half-Assed"** - The architecture is sound but implementation is incomplete.

### Follow-up Tracks Required
- Track I3: Fix missing dependencies (noted in original plan)
- **NEW**: Fix critical bugs (duplicates, race conditions, security)
- **NEW**: Externalize configuration
- **NEW**: Comprehensive test suite
- **NEW**: Complete stub implementations

---

*Report generated from automated analysis and code review. All findings are based on static analysis of the merged codebase.*
