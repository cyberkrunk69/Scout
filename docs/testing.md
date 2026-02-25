# Testing Guide

This document covers the testing strategy for Scout, including integration tests and performance benchmarks.

## Running Tests

### Prerequisites

Install test dependencies:

```bash
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Categories

```bash
# Integration tests for LLM providers
pytest tests/scout/llm/test_integration_providers.py -v

# Integration tests for search index
pytest tests/scout/test_integration_search.py -v

# Batch pipeline performance tests
pytest tests/scout/batch/test_batch_performance.py -v

# Unit tests for batch pipeline
pytest tests/scout/batch/test_batch_pipeline.py -v
```

## Integration Tests

### LLM Provider Integration Tests

Location: `tests/scout/llm/test_integration_providers.py`

These tests exercise the full request/response flow with a mock HTTP server.

**What they test:**
- Provider request/response flow with mock server
- Error handling (rate limits, HTTP errors)
- Key rotation logic
- Concurrent provider calls
- Provider registry operations

**Mock server features:**
- Simulates LLM API responses
- Tracks all requests received
- Configurable error responses for testing failure handling

**Run:**

```bash
pytest tests/scout/llm/test_integration_providers.py -v
```

### Search Index Integration Tests

Location: `tests/scout/test_integration_search.py`

These tests use temporary SQLite databases to test the FTS5 search backend.

**What they test:**
- Index creation and search
- Document add/update/delete operations
- Field weights configuration
- Search with confidence filtering
- Large document sets (1000+ documents)
- Edge cases (empty queries, special characters)

**Database:**
- Uses temporary directories for test isolation
- Each test gets a fresh database
- Tests verify persistence across SearchIndex instances

**Run:**

```bash
pytest tests/scout/test_integration_search.py -v
```

## Performance Tests

Location: `tests/scout/batch/test_batch_performance.py`

These tests measure execution time and establish performance baselines.

### Running Performance Tests

```bash
# Run all performance tests with verbose output
pytest tests/scout/batch/test_batch_performance.py -v -s

# Run specific performance test
pytest tests/scout/batch/test_batch_performance.py::TestBatchPipelinePerformance::test_pipeline_100_items_sequential -v -s
```

### Expected Baselines

| Test | Expected Time | Threshold |
|------|--------------|-----------|
| 100 items sequential | ~100-200ms | < 1000ms |
| 100 items parallel | ~50-100ms | < 500ms |
| 500 items stress test | ~500-1000ms | < 5000ms |
| Context set/get (1000 ops) | < 50ms | < 100ms |
| Progress events (200) | < 500ms | < 1000ms |

### Understanding Performance Output

The tests print detailed performance metrics:

```
=== Sequential 100 Items Performance ===
Total time: 145.23ms
Avg per task: 1.45ms
Throughput: 688.66 tasks/sec
```

**Key metrics:**
- **Total time**: Complete execution time for all tasks
- **Avg per task**: Average time per individual task
- **Throughput**: Tasks processed per second

### Performance Consistency

The consistency test runs multiple iterations to measure variance:

```
=== Consistency Test (5 runs of 50 items) ===
Durations: ['45.12ms', '43.89ms', '44.56ms', '46.01ms', '44.23ms']
Average: 44.76ms
Std Dev: 0.78ms
CV: 1.74%
```

**Key metrics:**
- **CV (Coefficient of Variation)**: Standard deviation as percentage of mean
- **Expected CV**: < 20% for stable performance

## CI Integration

### Adding Tests to CI

Example GitHub Actions workflow:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run unit tests
        run: pytest tests/scout/ -v --ignore=tests/scout/batch/test_batch_performance.py
      - name: Run integration tests
        run: pytest tests/scout/test_integration_search.py tests/scout/llm/test_integration_providers.py -v
      - name: Run performance tests
        run: pytest tests/scout/batch/test_batch_performance.py -v -s
```

### Performance Regression Detection

To detect performance regressions:

1. Run performance tests regularly (e.g., on each release)
2. Track metrics over time
3. Set alerts for > 20% degradation
4. Consider benchmarks for critical paths

## Writing New Tests

### Test File Structure

Follow the existing patterns:

```python
"""Description of what these tests cover."""

import pytest
import asyncio
# ... other imports

class TestComponentName:
    """Tests for ComponentName."""

    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        # Act
        # Assert
        pass

    @pytest.mark.asyncio
    async def test_async_functionality(self):
        """Test async functionality."""
        # Arrange
        # Act
        # Assert
        pass
```

### Test Fixtures

Use fixtures for common setup:

```python
@pytest.fixture
def temp_resource():
    """Fixture that provides a temporary resource."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
```

### Performance Test Guidelines

1. **Use `time.perf_counter()`** for accurate timing
2. **Print results** with `-s` flag for visibility
3. **Set reasonable thresholds** based on expected performance
4. **Test consistency** by running multiple iterations
5. **Document expected baselines** in comments

## Troubleshooting

### Tests Fail in CI but Pass Locally

- Check for timing-sensitive tests (increase thresholds)
- Verify environment differences (Python version, OS)
- Look for race conditions in async tests

### Performance Tests Unstable

- Increase iteration count for consistency tests
- Check for background processes consuming resources
- Consider running with reduced parallelism

### Mock Server Issues

- Ensure port availability (uses dynamic port allocation)
- Check for firewall blocking localhost connections
- Verify mock server starts before tests run
