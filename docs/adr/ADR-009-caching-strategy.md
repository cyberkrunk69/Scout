# ADR-009: Caching Strategy

**Date:** 2025-02-25
**Status:** Accepted
**Deciders:** Scout Architecture Team
**Related:** ADR-003

## Test Status

| Test Suite | Tests | Status |
|------------|-------|--------|
| `tests/scout/test_cache.py` | 16 | ✅ PASS |

**Total:** 16 tests, all passing.

---

## Context

Scout makes expensive operations that benefit from caching:
- LLM calls for similar prompts
- File system operations (stat, read)
- Repository analysis (git operations)
- Tool outputs that don't change frequently

Without caching:
- Repeated API calls cost money
- Redundant file operations slow execution
- Poor user experience for repeated queries

Early approaches:
- No caching at all
- Simple in-memory maps
- No TTL or eviction

This led to:
- Memory bloat over time
- Stale data served
- No cache invalidation

## Decision

Implemented a **decorator-based caching system** with TTL, LRU eviction, and file watching:

### Core Design

```python
@simple_cache(ttl_seconds=30, max_size=1000)
async def expensive_operation(param: str) -> Result:
    # Actual work happens here
    return await do_work(param)
```

### Key Features

#### 1. TTL-based Expiration

```python
@simple_cache(ttl_seconds=60)  # Cache expires after 60 seconds
def get_file_metadata(path: str) -> Metadata:
    return os.stat(path)
```

#### 2. LRU Eviction

When max_size reached, oldest entry evicted:

```python
@simple_cache(max_size=100)  # Evict LRU when full
def recent_searches(query: str) -> List[Result]:
    return db.search(query)
```

#### 3. Thread/Async Safety

Appropriate locking per function type:

```python
if is_async:
    lock = asyncio.Lock()  # For async functions
else:
    lock = threading.Lock()  # For sync functions
```

#### 4. Request Coalescing

Multiple concurrent calls with same key share execution:

```python
# If 3 calls with key "foo" arrive simultaneously:
# - First call executes
# - Others wait for result
# - All get same cached result
```

### File-Based Dependency Tracking

Cache entries can depend on files; when files change, cache invalidates:

```python
@simple_cache(ttl_seconds=300, dependencies=["src/**/*.py"])
def analyze_file(path: str) -> AnalysisResult:
    return parse_and_analyze(path)
```

How it works (`src/scout/cache_deps.py`):
1. Resolve glob patterns to actual file paths
2. Register dependency (cache_entry, files)
3. File watcher monitors files
4. On file change, invalidate dependent cache entries

### Cache Key Generation

Deterministic keys excluding transient data:

```python
# Excluded from key:
TRANSIENT_FIELDS = {
    "timestamp", "session_id", "request_id",
    "token", "nonce", "cache", "stats"
}

# SHA256 hash of filtered args/kwargs
key = hashlib.sha256(filtered_key_string.encode()).hexdigest()[:32]
```

### Statistics and Monitoring

Cache exposes stats for observability:

```python
@cached(ttl_seconds=60)
def expensive_call(x: int) -> int:
    return x * 2

# Access stats
print(expensive_call.cache_info())
# {'size': 10, 'hits': 50, 'misses': 20, 'hit_rate_percent': 71.43, ...}

# Clear cache
expensive_call.cache_clear()
```

## Consequences

### Positive

- **Cost savings**: Fewer LLM API calls
- **Performance**: Faster repeated operations
- **Simplicity**: Decorator-based, minimal code changes
- **Observability**: Cache stats exposed
- **Safety**: Request coalescing prevents duplicate work

### Negative

- **Memory**: In-memory only, grows with use
- **Staleness**: TTL may serve stale data
- **Complexity**: Dependency tracking adds overhead

### Mitigations

- TTL prevents indefinite staleness
- LRU eviction bounds memory
- File watching invalidates on source change
- Cache stats help tune settings

## Implementation Notes

### Code References

- Cache decorator: `src/scout/cache.py`
- Dependency tracking: `src/scout/cache_deps.py`
- Tests: `tests/scout/test_cache.py`

### Usage Patterns

```python
# Simple TTL cache
@simple_cache(ttl_seconds=30)
async def get_user_profile(user_id: str) -> dict:
    return await api.fetch_profile(user_id)

# With file dependencies
@simple_cache(ttl_seconds=300, dependencies=["src/**/*.py"])
def get_imports(file_path: str) -> List[str]:
    return parse_imports(file_path)

# Sync function
@simple_cache(ttl_seconds=60)
def git_status(repo_path: str) -> dict:
    return subprocess.run(["git", "status"], cwd=repo_path)
```

### Configuration

No external config - all via decorator parameters:
- `ttl_seconds`: Time-to-live (default: 5.0)
- `max_size`: Maximum entries (default: 1000)
- `dependencies`: File glob patterns

## Related ADRs

- [ADR-003](./ADR-003-budget-service-reservation-semantics.md) - Budget impacts of caching

## Notes

### Magic Number Audit

| File | Line | Value | Recommendation |
|------|------|-------|----------------|
| `cache.py` | 91-108 | `TRANSIENT_FIELDS` set | Good - centralized, but may miss edge cases |
| `cache.py` | 164 | `ttl_seconds: float = 5.0` | Default is quite short |
| `cache.py` | 165 | `max_size: int = 1000` | Should be configurable for large deployments |
| `cache.py` | 130 | `arg > 1e15` | Magic number for timestamp detection - document why |

### Stub Implementations

None identified - caching appears fully implemented.

### Future Considerations

- Distributed cache (Redis) for multi-instance deployments
- Persistent cache (disk-based) for survives restart
- Cache warming on startup
- More granular invalidation (per-key)
