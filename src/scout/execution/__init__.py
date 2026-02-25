"""Execution module for scout-core.

This module provides the core execution engine for running structured plans,
including action definitions, execution safety, plan parsing, and tool mapping.
"""

from .actions import (
    ActionType,
    StructuredStep,
    StructuredPlan,
    StepResult,
    ExecutionResult,
    WebStep,
    PlanContext,
    PlanResult,
)
from .executor import (
    PlanExecutor,
    ChangeRecord,
    RollbackManager,
    BudgetGuard,
)
from .registry import (
    ToolContract,
    ExecutionToolRegistry,
)
from .safety import (
    SafetyViolation,
    SafetyGuard,
)
from .llm_prose_parser import (
    ParsingError,
    LLMError,
    LLMClientProtocol,
    CostTrackerProtocol,
    LlmProseParser,
)
from .mapper import (
    StepToToolMapper,
)

__all__ = [
    # Actions
    "ActionType",
    "StructuredStep",
    "StructuredPlan",
    "StepResult",
    "ExecutionResult",
    "WebStep",
    "PlanContext",
    "PlanResult",
    # Executor
    "PlanExecutor",
    "ChangeRecord",
    "RollbackManager",
    "BudgetGuard",
    # Registry
    "ToolContract",
    "ExecutionToolRegistry",
    # Safety
    "SafetyViolation",
    "SafetyGuard",
    # LLM Prose Parser
    "ParsingError",
    "LLMError",
    "LLMClientProtocol",
    "CostTrackerProtocol",
    "LlmProseParser",
    # Mapper
    "StepToToolMapper",
]
