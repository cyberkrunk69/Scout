"""Plan executor for web automation tasks.

This module provides the WebPlanExecutor class that coordinates execution of
multi-step web automation plans using the browser agent.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Dict, List, Optional

from scout.execution.actions import (
    PlanContext,
    PlanResult,
    StepResult,
    WebStep,
)

logger = logging.getLogger(__name__)


def _calculate_backoff_delay(attempt: int, base_delay: float, max_delay: float, jitter_factor: float) -> float:
    """Calculate exponential backoff delay with jitter.

    Args:
        attempt: Current retry attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap in seconds
        jitter_factor: Jitter as fraction of base delay

    Returns:
        Delay in seconds to wait before next retry
    """
    delay = min(max_delay, base_delay * (2 ** attempt) + random.uniform(0, jitter_factor * base_delay))
    return delay


class WebPlanExecutor:
    """Executes web task plans using the browser agent.

    Execution Traces:
    - Happy: All steps execute successfully, data extracted
    - Failure: Step fails after retries, returns partial results
    - Edge: Browser crashes, executor catches and returns error

    Args:
        browser_agent_tool: The browser_act function to call for each step
        max_retries: Number of retries per step (default 1)
        base_delay: Base delay for exponential backoff in seconds (default 1.0)
        max_delay: Maximum delay cap for exponential backoff in seconds (default 60.0)
        jitter_factor: Jitter as fraction of base delay (default 0.1)
    """

    def __init__(
        self,
        browser_agent_tool: Any,
        max_retries: int = 1,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter_factor: float = 0.1,
    ):
        """Initialize the WebPlanExecutor.

        Args:
            browser_agent_tool: Callable that executes browser actions
            max_retries: Maximum retry attempts per step (default 1)
            base_delay: Base delay for exponential backoff in seconds
            max_delay: Maximum delay cap for exponential backoff in seconds
            jitter_factor: Jitter as fraction of base delay
        """
        self.browser_agent = browser_agent_tool
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter_factor = jitter_factor

    async def execute_plan(
        self,
        steps: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        plan_id: Optional[str] = None,
    ) -> PlanResult:
        """Execute a plan with retry policy.

        Execution Traces:
        - Happy: All steps complete successfully, returns PlanResult with success=True
        - Failure: Step fails after retries, returns PlanResult with success=False
        - Edge: Browser unavailable, returns PlanResult with error details

        Args:
            steps: List of step dictionaries (from parser)
            context: Optional execution context
            plan_id: Optional plan identifier for audit logging

        Returns:
            PlanResult with execution results
        """
        start_time = time.time()

        # Initialize context
        plan_context = PlanContext(
            extracted_data={},
            current_url=None,
            cookies=None,
            session_id=None,
            plan_id=plan_id,
        )
        if context:
            plan_context.extracted_data.update(context.get("extracted_data", {}))
            plan_context.current_url = context.get("current_url")

        # Track results
        step_results: List[StepResult] = []
        steps_executed = 0
        steps_failed = 0
        total_cost = 0.0

        logger.info(f"Starting plan execution: plan_id={plan_id}, steps={len(steps)}")

        for i, step_dict in enumerate(steps):
            step_index = i + 1

            # Handle both dict and WebStep formats
            if isinstance(step_dict, dict):
                if step_dict.get("command") == "browser_act":
                    step = WebStep(
                        action=step_dict.get("action", "click"),
                        target=step_dict.get("target"),
                        value=step_dict.get("value"),
                        url=step_dict.get("url"),
                        condition=step_dict.get("condition"),
                        step_index=step_index,
                    )
                else:
                    # Non-browser step - WebPlanExecutor only handles browser actions
                    # Non-web actions (e.g., run_command, file operations) should be
                    # executed via PlanExecutor which supports adapters for each action type
                    logger.debug(f"Skipping non-browser step (handled by PlanExecutor): {step_dict.get('command')}")
                    continue
            else:
                step = step_dict

            # Execute step with retries
            result = await self._execute_step_with_retry(step, plan_context)
            step_results.append(result)

            # Accumulate cost
            total_cost += result.cost

            if result.success:
                steps_executed += 1
                # Update context with extracted data
                if result.output and "extracted" in result.output:
                    key = f"step_{step_index}"
                    plan_context.extracted_data[key] = result.output["extracted"]
            else:
                steps_failed += 1
                logger.warning(
                    f"Step {step_index} failed after {result.retry_count} retries: {result.error}"
                )

            # Update current URL in context
            if result.output and "current_url" in result.output:
                plan_context.current_url = result.output["current_url"]

        total_duration_ms = int((time.time() - start_time) * 1000)

        return PlanResult(
            success=steps_failed == 0,
            plan_id=plan_id,
            steps_executed=steps_executed,
            steps_failed=steps_failed,
            step_results=step_results,
            extracted_data=plan_context.extracted_data,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
        )

    async def _execute_step_with_retry(
        self,
        step: WebStep,
        context: PlanContext,
    ) -> StepResult:
        """Execute a single step with exponential backoff retry logic.

        Execution Traces:
        - Happy: Step succeeds on first attempt
        - Failure: Step fails, retries with exponential backoff
        - Edge: Browser error on first try, succeeds on retry after backoff

        Args:
            step: WebStep to execute
            context: Current execution context

        Returns:
            StepResult with execution outcome
        """
        retry_count = 0
        last_error = None
        last_failure_reason = None
        last_step_cost = 0.0

        for attempt in range(self.max_retries + 1):
            try:
                # Convert WebStep to browser_act parameters
                params = step.to_browser_act_params()

                # Add session_id if available
                if context.session_id:
                    params["session_id"] = context.session_id

                # Execute the browser action
                result = await self.browser_agent(**params)

                # Extract cost from ToolOutput
                step_cost = 0.0
                if hasattr(result, 'cost_usd'):
                    step_cost = result.cost_usd
                last_step_cost = step_cost

                # Check if successful
                if result.success:
                    return StepResult(
                        step_index=step.step_index,
                        action=step.action,
                        success=True,
                        output=result.data if hasattr(result, 'data') else None,
                        error=None,
                        failure_reason=None,
                        retry_count=retry_count,
                        cost=step_cost,
                        target=step.target,
                    )
                else:
                    last_error = result.error if hasattr(result, 'error') else str(result)
                    last_failure_reason = last_error
                    retry_count = attempt + 1

            except Exception as e:
                last_error = str(e)
                last_failure_reason = f"Exception: {e}"
                retry_count = attempt + 1
                logger.warning(f"Step {step.step_index} attempt {attempt + 1} failed: {e}")

            # Apply exponential backoff before retry (if not last attempt)
            if attempt < self.max_retries:
                delay = _calculate_backoff_delay(
                    attempt,
                    self.base_delay,
                    self.max_delay,
                    self.jitter_factor
                )
                logger.info(
                    f"Step {step.step_index} retry {attempt + 1}/{self.max_retries} "
                    f"after {delay:.2f}s backoff (error: {last_error})"
                )
                await asyncio.sleep(delay)

        # All retries exhausted - return last known cost
        return StepResult(
            step_index=step.step_index,
            action=step.action,
            success=False,
            output=None,
            error=last_error,
            failure_reason=last_failure_reason,
            retry_count=retry_count,
            cost=last_step_cost,
            target=step.target,
        )


def plan_result_to_dict(result: PlanResult) -> Dict[str, Any]:
    """Convert PlanResult to dictionary for JSON serialization.

    Execution Traces:
    - Happy: Converts successfully to dict
    - Failure: N/A (always returns dict)
    - Edge: N/A

    Args:
        result: PlanResult to convert

    Returns:
        Dictionary representation
    """
    return {
        "success": result.success,
        "plan_id": result.plan_id,
        "steps_executed": result.steps_executed,
        "steps_failed": result.steps_failed,
        "total_cost": result.total_cost,
        "total_duration_ms": result.total_duration_ms,
        "extracted_data": result.extracted_data,
        "step_results": [
            {
                "step": sr.step_index,
                "action": sr.action,
                "success": sr.success,
                "output": sr.output,
                "error": sr.error,
                "failure_reason": sr.failure_reason,
                "retry_count": sr.retry_count,
                "cost": sr.cost,
                "target": sr.target,
            }
            for sr in result.step_results
        ],
    }
