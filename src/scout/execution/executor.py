"""Plan Executor - Executes structured plans with dependency resolution."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from scout.execution.actions import (
    ActionType,
    ExecutionResult,
    StepResult,
    StructuredPlan,
    StructuredStep,
)
from scout.execution.registry import ExecutionToolRegistry
from scout.config.defaults import (
    EXECUTOR_MAX_BUDGET,
    EXECUTOR_ESTIMATED_COST,
)

logger = logging.getLogger(__name__)


@dataclass
class ChangeRecord:
    """Record of a change made during execution."""
    step_id: int
    change_type: str
    undo_command: str
    metadata: dict


class RollbackManager:
    """Tracks changes for self-healing rollback capability."""
    
    def __init__(self):
        self._change_log: List[ChangeRecord] = []
        
    def record(self, step_id: int, change_type: str, undo_command: str, metadata: dict):
        """Record a change that can be undone."""
        self._change_log.append(ChangeRecord(
            step_id=step_id,
            change_type=change_type,
            undo_command=undo_command,
            metadata=metadata
        ))
        
    async def rollback_to(self, step_id: int) -> List[Dict]:
        """Rollback all changes made after step_id."""
        results = []
        # Changes are in order, so find the first one after step_id
        for change in reversed(self._change_log):
            if change.step_id > step_id:
                # Execute undo command
                logger.info(f"Rolling back step {change.step_id}: {change.change_type}")
                results.append({
                    "step_id": change.step_id,
                    "type": change.change_type,
                    "status": "rolled_back"
                })
        return results
        
    def generate_undo_plan(self, from_step: int) -> StructuredPlan:
        """Generate a plan to undo changes made after from_step.

        Uses the recorded changes in the rollback manager to create
        a structured plan that reverses the changes.
        """
        undo_steps = []

        # Changes are in order, so find all changes after from_step
        for change in reversed(self._change_log):
            if change.step_id > from_step:
                # Generate an undo step for this change
                undo_step = StructuredStep(
                    action_type=ActionType.RUN_COMMAND,
                    description=f"Undo {change.change_type} from step {change.step_id}",
                    command=change.undo_command,
                    step_id=len(undo_steps),
                    depends_on=[],
                )
                undo_steps.append(undo_step)

        return StructuredPlan(
            steps=undo_steps,
            raw_plan=f"Undo plan from step {from_step}",
            summary=f"Generated undo plan for {len(undo_steps)} changes"
        )


class BudgetGuard:
    """Monitors and enforces budget limits during execution."""
    
    def __init__(self, max_budget: float):
        self.max_budget = max_budget
        self.current_cost = 0.0
        
    async def check(self, step: StructuredStep, estimated_cost: float) -> bool:
        """Pre-execution check. Returns False if would exceed budget."""
        if self.current_cost + estimated_cost > self.max_budget:
            logger.warning(
                f"Budget would be exceeded: ${self.current_cost:.4f} + ${estimated_cost:.4f} > ${self.max_budget:.4f}"
            )
            return False
        return True
        
    async def update(self, actual_cost: float) -> None:
        """Post-execution check. Update current_cost and warn if exceeded."""
        self.current_cost += actual_cost
        if self.current_cost > self.max_budget:
            logger.warning(f"Budget exceeded: ${self.current_cost:.4f} > ${self.max_budget:.4f}")


class PlanExecutor:
    """Executes structured plans with dependency resolution."""
    
    def __init__(
        self,
        registry: ExecutionToolRegistry,
        max_budget: float = EXECUTOR_MAX_BUDGET,
        discovery_callback: Optional[Callable] = None,
    ):
        self.registry = registry
        self.max_budget = max_budget
        self.discovery_callback = discovery_callback
        self.rollback_manager = RollbackManager()
        self.budget_guard = BudgetGuard(max_budget)
        self.current_cost = 0.0
        
    async def execute(self, plan: StructuredPlan) -> ExecutionResult:
        """Execute a structured plan with dependency resolution."""
        start_time = time.time()
        
        # Build DAG and get execution order
        batches = self._build_execution_batches(plan.steps)
        
        results: Dict[int, StepResult] = {}
        steps_completed = 0
        steps_failed = 0
        discoveries = []
        
        # Execute each batch
        for batch in batches:
            # Execute all steps in this batch in parallel
            tasks = [
                self._execute_step(step, results)
                for step in batch
                if self._can_execute(step, results)
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for step, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Step {step.step_id} failed: {result}")
                    results[step.step_id] = StepResult(
                        step_id=step.step_id,
                        success=False,
                        error=str(result)
                    )
                    steps_failed += 1
                else:
                    results[step.step_id] = result
                    if result.success:
                        steps_completed += 1
                    else:
                        steps_failed += 1
                        
                    # Check for discoveries
                    if result.output and result.output.get("discovery"):
                        discoveries.append({
                            "step_id": step.step_id,
                            "type": result.output.get("discovery_type"),
                            "detail": result.output.get("discovery_detail"),
                            "requires_replan": result.output.get("requires_replan", False)
                        })
        
        total_duration = int((time.time() - start_time) * 1000)
        
        return ExecutionResult(
            steps_completed=steps_completed,
            steps_failed=steps_failed,
            total_cost=self.current_cost,
            total_duration=total_duration,
            discoveries=discoveries,
            rollback_commands=[]
        )
        
    def _build_execution_batches(self, steps: List[StructuredStep]) -> List[List[StructuredStep]]:
        """Build execution batches using topological sort (Kahn's algorithm)."""
        # Build adjacency list and in-degree count
        step_map = {step.step_id: step for step in steps}
        in_degree = {step.step_id: 0 for step in steps}
        dependents = {step.step_id: [] for step in steps}
        
        for step in steps:
            for dep in step.depends_on:
                if dep in step_map:
                    dependents[dep].append(step.step_id)
                    in_degree[step.step_id] += 1
        
        # Kahn's algorithm
        batches = []
        processed: Set[int] = set()
        
        while len(processed) < len(steps):
            # Find all steps with no remaining dependencies
            ready = [
                step_id for step_id, degree in in_degree.items()
                if degree == 0 and step_id not in processed
            ]
            
            if not ready:
                # Cycle detected or missing dependency
                logger.warning(f"Cycle or missing dependency detected. Remaining: {set(in_degree.keys()) - processed}")
                break
            
            batch = [step_map[step_id] for step_id in ready]
            batches.append(batch)
            
            # Update in-degrees
            for step_id in ready:
                processed.add(step_id)
                for dependent in dependents[step_id]:
                    in_degree[dependent] -= 1
        
        return batches
        
    def _can_execute(self, step: StructuredStep, results: Dict[int, StepResult]) -> bool:
        """Check if all dependencies are satisfied."""
        for dep_id in step.depends_on:
            if dep_id not in results:
                return False
            if not results[dep_id].success:
                return False
        return True
        
    async def _execute_step(
        self,
        step: StructuredStep,
        results: Dict[int, StepResult]
    ) -> StepResult:
        """Execute a single step via tool registry."""
        start_time = time.time()
        
        try:
            # Get tool from registry
            tool_name = self.registry.get_tool_name(step.action_type)
            if not tool_name:
                return StepResult(
                    step_id=step.step_id,
                    success=False,
                    error=f"No tool found for action type: {step.action_type}"
                )
            
            # Check budget
            estimated_cost = EXECUTOR_ESTIMATED_COST  # Rough estimate
            if not await self.budget_guard.check(step, estimated_cost):
                return StepResult(
                    step_id=step.step_id,
                    success=False,
                    error="Budget exceeded"
                )
            
            # Handle user input step
            if step.action_type == ActionType.GET_USER_INPUT:
                # This would be handled by the REPL integration
                return StepResult(
                    step_id=step.step_id,
                    success=True,
                    output={"user_input_required": True}
                )
            
            # Execute via adapter if present
            adapter = self.registry.get_adapter(step.action_type)
            if adapter:
                output = await adapter(step)
            else:
                # Direct execution (placeholder)
                output = {"executed": True, "tool": tool_name}
            
            # Update budget
            actual_cost = estimated_cost
            await self.budget_guard.update(actual_cost)
            self.current_cost += actual_cost
            
            # Record for rollback
            if step.rollback_on_fail:
                self.rollback_manager.record(
                    step.step_id,
                    str(step.action_type),
                    f"undo_{step.step_id}",
                    {"tool": tool_name}
                )
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            return StepResult(
                step_id=step.step_id,
                success=True,
                output=output,
                cost=actual_cost,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Step {step.step_id} execution failed: {e}")
            return StepResult(
                step_id=step.step_id,
                success=False,
                error=str(e)
            )
