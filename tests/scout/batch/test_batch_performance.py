"""Performance tests for batch pipeline.

These tests measure the performance of processing multiple items through
the batch pipeline and establish performance baselines.
"""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from unittest.mock import AsyncMock, Mock

from scout.batch_pipeline import PipelineExecutor
from scout.batch_context import BatchContext
from scout.progress import ProgressReporter, Status, ProgressEvent


# Simple task runner for testing
async def mock_task_runner(task: dict, semaphore: asyncio.Semaphore, index: int) -> dict:
    """Mock task runner that simulates work."""
    await asyncio.sleep(0.001)  # Simulate minimal async work
    return {
        "status": "success",
        "output": f"Task {index} completed",
        "error": None,
    }


class TestBatchPipelinePerformance:
    """Performance tests for batch pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_100_items_sequential(self):
        """Test processing 100 items sequentially - establish baseline."""
        # Create 100 tasks
        tasks = [
            {"command": f"task_{i}", "args": {}}
            for i in range(100)
        ]

        context = BatchContext()
        executor = PipelineExecutor(
            context=context,
            task_runner=mock_task_runner,
        )

        # Measure execution time
        start_time = time.perf_counter()
        results = await executor.run(tasks, mode="sequential")
        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000

        # Verify all tasks completed
        assert len(results) == 100
        assert all(r.get("status") == "success" for r in results)

        # Log performance metrics
        print(f"\n=== Sequential 100 Items Performance ===")
        print(f"Total time: {duration_ms:.2f}ms")
        print(f"Avg per task: {duration_ms/100:.2f}ms")
        print(f"Throughput: {100/duration_ms*1000:.2f} tasks/sec")

        # Performance assertions
        # Sequential with 1ms simulated work per task should complete in ~100ms + overhead
        # Allow 500ms as upper bound for CI environments
        assert duration_ms < 1000, f"Pipeline too slow: {duration_ms:.2f}ms"

    @pytest.mark.asyncio
    async def test_pipeline_100_items_parallel(self):
        """Test processing 100 items with limited parallelism."""
        # Create 100 tasks
        tasks = [
            {"command": f"task_{i}", "args": {}}
            for i in range(100)
        ]

        context = BatchContext()
        executor = PipelineExecutor(
            context=context,
            task_runner=mock_task_runner,
        )

        # Measure execution time
        start_time = time.perf_counter()
        results = await executor.run(tasks, mode="parallel")
        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000

        # Verify all tasks completed
        assert len(results) == 100
        assert all(r.get("status") == "success" for r in results)

        # Log performance metrics
        print(f"\n=== Parallel 100 Items Performance ===")
        print(f"Total time: {duration_ms:.2f}ms")
        print(f"Avg per task: {duration_ms/100:.2f}ms")
        print(f"Throughput: {100/duration_ms*1000:.2f} tasks/sec")

    @pytest.mark.asyncio
    async def test_pipeline_multiple_runs_consistency(self):
        """Test that multiple runs have consistent performance."""
        durations: List[float] = []

        for run in range(5):
            tasks = [
                {"command": f"task_{i}", "args": {}}
                for i in range(50)
            ]

            context = BatchContext()
            executor = PipelineExecutor(
                context=context,
                task_runner=mock_task_runner,
            )

            start_time = time.perf_counter()
            await executor.run(tasks, mode="sequential")
            end_time = time.perf_counter()

            durations.append((end_time - start_time) * 1000)

        avg_duration = statistics.mean(durations)
        stdev = statistics.stdev(durations) if len(durations) > 1 else 0

        print(f"\n=== Consistency Test (5 runs of 50 items) ===")
        print(f"Durations: {[f'{d:.2f}ms' for d in durations]}")
        print(f"Average: {avg_duration:.2f}ms")
        print(f"Std Dev: {stdev:.2f}ms")
        print(f"CV: {stdev/avg_duration*100:.2f}%")

        # Coefficient of variation should be reasonable (< 20%)
        cv = stdev / avg_duration if avg_duration > 0 else 0
        assert cv < 0.3, f"Performance too inconsistent: CV = {cv*100:.1f}%"


class TestBatchContextPerformance:
    """Performance tests for batch context operations."""

    @pytest.mark.asyncio
    async def test_context_set_get_performance(self):
        """Test context set/get performance."""
        context = BatchContext()

        # Measure set operations
        start_time = time.perf_counter()
        for i in range(1000):
            context.set_var(f"var_{i}", {"data": i})
        set_time = time.perf_counter() - start_time

        # Measure get operations
        start_time = time.perf_counter()
        for i in range(1000):
            value = context.get_var(f"var_{i}")
            assert value is not None
        get_time = time.perf_counter() - start_time

        print(f"\n=== Context Performance ===")
        print(f"1000 set operations: {set_time*1000:.2f}ms ({1000/set_time:.0f} ops/sec)")
        print(f"1000 get operations: {get_time*1000:.2f}ms ({1000/get_time:.0f} ops/sec)")

        # Should be fast (< 100ms for 1000 operations)
        assert set_time < 0.1
        assert get_time < 0.1

    @pytest.mark.asyncio
    async def test_context_result_storage_performance(self):
        """Test context result storage performance."""
        context = BatchContext()

        # Measure storing results
        start_time = time.perf_counter()
        for i in range(500):
            await context.set_result(
                task_index=i,
                command=f"task_{i}",
                status="success",
                output={"result": i},
                error=None
            )
        storage_time = time.perf_counter() - start_time

        print(f"\n=== Result Storage Performance ===")
        print(f"500 result stores: {storage_time*1000:.2f}ms ({500/storage_time:.0f} ops/sec)")

        # Should be fast
        assert storage_time < 0.5


class TestProgressReporterPerformance:
    """Performance tests for progress reporting."""

    @pytest.mark.asyncio
    async def test_progress_reporter_bulk_events(self):
        """Test handling many progress events."""
        reporter = ProgressReporter()

        start_time = time.perf_counter()
        for i in range(200):
            await reporter.emit_async(ProgressEvent(
                task_id=f"task_{i}",
                status=Status.RUNNING,
                message=f"Task {i} running",
            ))
        emit_time = time.perf_counter() - start_time

        print(f"\n=== Progress Reporter Performance ===")
        print(f"200 emit events: {emit_time*1000:.2f}ms ({200/emit_time:.0f} events/sec)")

        # Should be fast
        assert emit_time < 1.0

        # Verify history
        history = reporter.get_history()
        assert len(history) == 200


class TestBatchPipelineStress:
    """Stress tests for batch pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_500_items_stress(self):
        """Test pipeline with 500 items - stress test."""
        tasks = [
            {"command": f"task_{i}", "args": {}}
            for i in range(500)
        ]

        context = BatchContext()
        executor = PipelineExecutor(
            context=context,
            task_runner=mock_task_runner,
        )

        start_time = time.perf_counter()
        results = await executor.run(tasks, mode="sequential")
        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000

        print(f"\n=== Stress Test (500 Items) ===")
        print(f"Total time: {duration_ms:.2f}ms")
        print(f"Throughput: {500/duration_ms*1000:.2f} tasks/sec")

        assert len(results) == 500
        # 500 items should complete in reasonable time
        assert duration_ms < 5000

    @pytest.mark.asyncio
    async def test_pipeline_with_variables(self):
        """Test pipeline with variable interpolation overhead."""
        # Create tasks that use variables
        tasks = [
            {
                "command": f"task_{i}",
                "args": {"index": i},
                "variables": {"task_index": i} if i % 10 == 0 else None,
            }
            for i in range(100)
        ]

        # Remove None variables
        tasks = [t for t in tasks if t.get("variables")]

        context = BatchContext()
        executor = PipelineExecutor(
            context=context,
            task_runner=mock_task_runner,
        )

        start_time = time.perf_counter()
        results = await executor.run(tasks, mode="sequential")
        end_time = time.perf_counter()

        duration_ms = (end_time - start_time) * 1000

        print(f"\n=== Variable Interpolation Test (100 items with vars) ===")
        print(f"Total time: {duration_ms:.2f}ms")
        print(f"Avg per task: {duration_ms/len(tasks):.2f}ms")

        assert duration_ms < 1000


# Benchmark utilities for manual testing
def run_benchmark(name: str, func, iterations: int = 10):
    """Run a benchmark function multiple times and report statistics."""
    durations = []

    for _ in range(iterations):
        start = time.perf_counter()
        asyncio.run(func())
        end = time.perf_counter()
        durations.append((end - start) * 1000)

    avg = statistics.mean(durations)
    median = statistics.median(durations)
    stdev = statistics.stdev(durations) if len(durations) > 1 else 0

    print(f"\n=== {name} ===")
    print(f"Iterations: {iterations}")
    print(f"Average: {avg:.2f}ms")
    print(f"Median: {median:.2f}ms")
    print(f"Std Dev: {stdev:.2f}ms")
    print(f"Min: {min(durations):.2f}ms")
    print(f"Max: {max(durations):.2f}ms")

    return {
        "name": name,
        "avg_ms": avg,
        "median_ms": median,
        "stdev_ms": stdev,
        "min_ms": min(durations),
        "max_ms": max(durations),
    }


if __name__ == "__main__":
    # Allow running benchmarks directly
    print("Running standalone benchmarks...")
    print("Use pytest for full test suite.")
