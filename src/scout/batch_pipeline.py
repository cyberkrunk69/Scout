from __future__ import annotations
"""Pipeline executor for stateful batch operations.

Self-Healing Features:
- Progress reporting with whimsy (following vivarium/scout/ui/whimsy.py patterns)
- Circuit breaker pattern for failure detection
- Retry with exponential backoff
- Dependency-aware parallel execution
"""
import asyncio
import json
import time
from typing import Any, Optional, Callable
from scout.batch_context import BatchContext
from scout.batch_expression import ExpressionEvaluator
from scout.progress import ProgressReporter, Status, ProgressEvent, format_deterministic
from scout.circuit_breaker import CircuitBreaker, CircuitBreakerManager, CircuitBreakerConfig
from scout.retry import RetryConfig, RetryContext, is_retryable


class PipelineExecutor:
    """
    Execute batch tasks with conditionals, variables, and early exit.
    
    Features:
    - Sequential or parallel execution
    - if/skip_if/stop_if conditionals
    - store_as to save results to context
    - ${var} variable interpolation
    - Early exit on stop_if truthy
    - Auto-JSON: Automatically inject --json flags for commands that need structured output
    """
    
    # Commands that need JSON output for variable interpolation
    COMMANDS_NEED_JSON = {
        "git_status": "json",
        "git_branch": "json",
        "git_diff": "json",
        "git_log": "json",
        "git_show": "json",
        "plan": "json_output",
        "lint": "json_output",
        "audit": "json_output",
        "run": "json_output",
        "nav": "json_output",
        "doc_sync": "json_output",
        "roast": "json_output",
        "query": "json_output",
        "ci_guard": "json_output",
        "brief": "json_output",
        "status": "json_output",
        "index": "json_output",
        "validate": "json_output",
    }
    
    def __init__(
        self,
        context: BatchContext,
        task_runner: Callable,
        reporter: Optional[ProgressReporter] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
        retry_config: Optional[RetryConfig] = None,
    ):
        self.context = context
        self.task_runner = task_runner
        self.evaluator = ExpressionEvaluator(context)
        
        # Self-healing components
        self.reporter = reporter or ProgressReporter()
        self.circuit_breaker = circuit_breaker
        self.retry_config = retry_config or RetryConfig()
        
        # Track cost
        self._total_cost = 0.0
    
    async def run(self, tasks: list[dict], mode: str = "sequential") -> list[dict]:
        """
        Run tasks through the pipeline with self-healing features.
        
        Features:
        - Progress reporting with whimsy
        - Circuit breaker for failure handling
        - Retry with exponential backoff
        - Dependency-aware parallel execution
        
        Args:
            tasks: List of task dicts with command, args, if, skip_if, stop_if, store_as
            mode: "sequential" (default) or "parallel"
        
        Returns:
            List of result dicts in order
        """
        results = []
        start_time = time.time()
        
        # Handle mode from first task or default
        if tasks and isinstance(tasks[0], dict):
            mode = tasks[0].get("mode", mode)
        
        # Check for initial variables in first task
        if tasks and isinstance(tasks[0], dict):
            if "variables" in tasks[0]:
                for k, v in tasks[0]["variables"].items():
                    self.context.set_var(k, v)
        
        # Emit start event
        start_time = time.time()
        await self.reporter.emit_async(ProgressEvent(
            task_id="batch:start",
            status=Status.RUNNING,
            message=f"Starting {len(tasks)} tasks in {mode} mode",
            metadata={"total_tasks": len(tasks), "mode": mode}
        ))
        
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            
            # Check circuit breaker before execution
            if self.circuit_breaker and not self.circuit_breaker.can_execute():
                await self.reporter.emit_async(ProgressEvent(
                    task_id=f"task:{i}",
                    status=Status.CIRCUIT_OPEN,
                    message="Circuit breaker open - stopping execution",
                    metadata={"failure_count": self.circuit_breaker.stats.failures}
                ))
                break
            
            # Check for early exit
            early_exit = await self.context.check_early_exit()
            if early_exit:
                await self.reporter.emit_async(ProgressEvent(
                    task_id=f"task:{i}",
                    status=Status.SKIPPED,
                    message=f"Pipeline stopped: {early_exit['reason']}",
                    metadata={"stopped_at": i, "reason": early_exit["reason"]}
                ))
                break
            
            # Interpolate args
            if "args" in task:
                task["args"] = self.evaluator.interpolate_args(task["args"])
            
            # Auto-inject JSON flags for commands that need structured output
            task = self._inject_json_flags(task)
            
            # Check if conditionals
            if "if" in task:
                condition_result = self.evaluator.evaluate_condition(task["if"])
                if not condition_result:
                    await self.reporter.emit_async(ProgressEvent(
                        task_id=f"task:{i}",
                        status=Status.SKIPPED,
                        message=f"Skipped - condition false: {task['if']}",
                        metadata={"condition": task["if"], "result": condition_result}
                    ))
                    continue
            
            if "skip_if" in task:
                condition_result = self.evaluator.evaluate_condition(task["skip_if"])
                if condition_result:
                    await self.reporter.emit_async(ProgressEvent(
                        task_id=f"task:{i}",
                        status=Status.SKIPPED,
                        message=f"Skipped - skip_if true: {task['skip_if']}",
                        metadata={"condition": task["skip_if"], "result": condition_result}
                    ))
                    continue
            
            # Emit running event
            task_id = task.get("command", "unknown")
            await self.reporter.emit_async(ProgressEvent(
                task_id=f"task:{i}:{task_id}",
                status=Status.RUNNING,
                message=f"Executing {task_id}",
                metadata={"task_index": i, "command": task_id, "args": task.get("args", {})}
            ))
            
            # Execute task with retry
            task_start = time.time()
            result = await self._run_task_with_retry(task, i)
            task_duration = (time.time() - task_start) * 1000
            
            # Update circuit breaker based on result
            if self.circuit_breaker:
                if result.get("status") == "success":
                    await self.circuit_breaker.record_success()
                else:
                    await self.circuit_breaker.record_failure(Exception(result.get("error", "Unknown error")))
            
            # Emit result event
            result_status = Status.SUCCESS if result.get("status") == "success" else Status.FAILURE
            await self.reporter.emit_async(ProgressEvent(
                task_id=f"task:{i}:{task_id}",
                status=result_status,
                message=result.get("error") or f"Completed {task_id}",
                metadata={
                    "duration_ms": int(task_duration),
                    "status": result.get("status"),
                    "error": result.get("error"),
                }
            ))
            
            results.append(result)
            
            # Handle extract_steps - auto-spawn sub-batch from plan output
            if task.get("extract_steps") and result.get("output"):
                # Lazy import to avoid circular dependency
                from vivarium.scout.batch_subbatch import SubBatchOrchestrator
                sub_orchestrator = SubBatchOrchestrator()
                sub_result = await sub_orchestrator.execute_plan_steps(
                    plan_output=result["output"],
                    task_runner=self.task_runner,
                    context=self.context,
                    mode=task.get("subbatch_mode", "sequential")
                )
                # Add sub-batch results to main result
                result["sub_batch"] = sub_result
            
            # Store result if store_as specified
            if task.get("store_as"):
                self.context.set_var(task["store_as"], result)
            
            # Store in context
            await self.context.set_result(
                i,
                task.get("command", "unknown"),
                result.get("status", "unknown"),
                result.get("output"),
                result.get("error")
            )
            
            # Check stop_if after execution
            if "stop_if" in task:
                if self.evaluator.evaluate_condition(task["stop_if"]):
                    await self.context.set_early_exit(i, f"stop_if: {task['stop_if']}")
                    break
        
        # Emit completion event
        total_duration = (time.time() - start_time) * 1000
        success_count = sum(1 for r in results if r.get("status") == "success")
        failure_count = len(results) - success_count
        
        await self.reporter.emit_async(ProgressEvent(
            task_id="batch:complete",
            status=Status.COMPLETE,
            message=f"Completed {success_count}/{len(results)} tasks",
            metadata={
                "success_count": success_count,
                "failure_count": failure_count,
                "total": len(results),
                "total_duration_ms": int(total_duration),
                "circuit_breaker": self.circuit_breaker.get_report() if self.circuit_breaker else None,
            }
        ))
        
        return results
    
    async def _run_task_sync(self, task: dict, index: int) -> dict:
        """Run a single task synchronously (for sequential mode."""
        return await self.task_runner(task, asyncio.Semaphore(1), index)
    
    async def _run_task_with_retry(self, task: dict, index: int) -> dict:
        """Run a single task with retry logic."""
        retry_ctx = RetryContext(
            task_id=f"task:{index}",
            config=self.retry_config,
            reporter=self.reporter
        )
        
        while retry_ctx.can_retry:
            try:
                retry_ctx.stats.attempts += 1
                
                # Execute the task
                result = await self.task_runner(task, asyncio.Semaphore(1), index)
                
                # Check for success
                if result.get("status") == "success":
                    return result
                
                # Task failed - check if retryable
                error = Exception(result.get("error", "Unknown error"))
                if not is_retryable(error, self.retry_config):
                    return result
                
                # Retry with backoff
                if retry_ctx.can_retry:
                    await retry_ctx.wait_before_retry()
                    
            except Exception as e:
                retry_ctx.stats.last_error = e
                
                if not is_retryable(e, self.retry_config):
                    return {"status": "error", "error": str(e), "output": None}
                
                if retry_ctx.can_retry:
                    await retry_ctx.wait_before_retry()
                else:
                    return {"status": "error", "error": str(e), "output": None}
        
        # All retries exhausted
        return {
            "status": "error",
            "error": f"Failed after {retry_ctx.stats.attempts} attempts: {retry_ctx.stats.last_error}",
            "output": None
        }
    
    def _inject_json_flags(self, task: dict) -> dict:
        """
        Automatically inject JSON flags for commands that need structured output.
        
        This enables transparent variable interpolation without requiring users
        to manually specify --json flags for every command.
        """
        command = task.get("command", "")
        args = task.get("args", {})
        
        # Check if this command needs JSON
        json_flag = self.COMMANDS_NEED_JSON.get(command)
        if not json_flag:
            return task
        
        # Skip if already has the flag set
        if args.get(json_flag):
            return task
        
        # Skip for plan if not structured/extract_steps
        if command == "plan":
            if not args.get("structured") and not task.get("extract_steps"):
                return task
        
        # Inject the JSON flag
        args = dict(args)  # Don't mutate original
        args[json_flag] = True
        task["args"] = args
        
        return task
    
    def set_reporter(self, reporter: ProgressReporter) -> None:
        """Set the progress reporter."""
        self.reporter = reporter
    
    def set_circuit_breaker(self, cb: CircuitBreaker) -> None:
        """Set the circuit breaker."""
        self.circuit_breaker = cb
    
    def get_reporter(self) -> ProgressReporter:
        """Get the progress reporter."""
        return self.reporter
    
    def get_context(self) -> BatchContext:
        """Get the batch execution context.

        Returns:
            The BatchContext instance for this pipeline.
        """
        return self.context
    
    def to_dict(self) -> dict:
        """Export pipeline state."""
        return {
            "context": self.context.to_dict(),
            "mode": getattr(self, '_mode', 'sequential'),
            "circuit_breaker": self.circuit_breaker.get_report() if self.circuit_breaker else None,
            "progress_events": len(self.reporter.get_history()),
        }
