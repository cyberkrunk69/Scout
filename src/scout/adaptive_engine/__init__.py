"""
Adaptive Engine - Core execution abstraction layer.

This module provides the foundation for adaptive execution with:
- State management (PlanningContext/StateManager)
- Trigger registry for pivot detection
- Quality gates (Threshold, Ballot)
- Cost controller for budget management
- Tool protocols

Phase 1: Core extraction from existing Scout modules.
"""

from .cost import CostController, Reservation
from .gates import (
    BallotGate,
    BallotVote,
    GateDecision,
    GateRegistry,
    GateResult,
    GateStage,
    QualityGate,
    ThresholdGate,
)
from .protocols import AsyncTool, Tool, ToolContext, ToolResult
from .state import PlanningContext, StateManager
from .triggers import (
    PIVOT_TRIGGERS,
    DEFAULT_PIVOT_THRESHOLD,
    PIVOT_FEEDBACK_FILE,
    TriggerConfig,
    TriggerRegistry,
    compute_optimal_threshold,
    create_default_registry,
    log_pivot_outcome,
)

# Public API
__all__ = [
    # State
    "PlanningContext",
    "StateManager",  # Alias for PlanningContext
    
    # Triggers
    "TriggerRegistry",
    "TriggerConfig",
    "create_default_registry",
    "PIVOT_TRIGGERS",
    "DEFAULT_PIVOT_THRESHOLD",
    "PIVOT_FEEDBACK_FILE",
    "log_pivot_outcome",
    "compute_optimal_threshold",
    
    # Gates
    "GateRegistry",
    "QualityGate",
    "ThresholdGate",
    "BallotGate",
    "Ballot",
    "BallotVote",
    "GateDecision",
    "GateStage",
    "GateResult",
    
    # Cost
    "CostController",
    "Reservation",
    
    # Protocols
    "Tool",
    "AsyncTool",
    "ToolContext",
    "ToolResult",
]

# Version info
__version__ = "1.0.0"
