"""
Budget Service - Centralized cost management for Scout.

Extracts budget handling from router.py with reservation semantics:
- Check if budget allows operation
- Reserve budget before LLM call
- Commit actual cost or rollback on failure

Phase 2 of Unified Tool Framework.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from scout.audit import AuditLog, get_audit
from scout.config import ScoutConfig


# Token cost estimates (moved from router.py)
TOKENS_PER_SMALL_FILE = 500
COST_PER_MILLION_8B = 0.20
COST_PER_MILLION_70B = 0.90
BRIEF_COST_PER_FILE = 0.005
TASK_NAV_ESTIMATED_COST = 0.002
DRAFT_COST_PER_FILE = 0.0004


class BudgetError(Exception):
    """Base exception for budget operations."""
    pass


class InsufficientBudgetError(BudgetError):
    """Raised when reservation fails due to insufficient budget."""

    def __init__(self, requested: float, available: float, operation: str):
        self.requested = requested
        self.available = available
        self.operation = operation
        super().__init__(
            f"Insufficient budget for {operation}: requested ${requested:.4f}, "
            f"available ${available:.4f}"
        )


class BudgetReservationTimeoutError(BudgetError):
    """Raised when reservation times out."""
    pass


@dataclass
class Reservation:
    """Represents a reserved budget for an operation."""
    id: str
    operation: str
    estimated_cost: float
    actual_cost: float = 0.0
    committed: bool = False
    created_at: float = field(default_factory=time.time)
    committed_at: Optional[float] = None


@dataclass
class CostEntry:
    """A single cost entry for tracking."""
    operation: str
    estimated_cost: float
    actual_cost: float
    output_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class BudgetService:
    """
    Centralized budget management with reservation semantics.

    Provides:
    - Check if budget allows operation
    - Reserve budget before LLM call (context manager)
    - Commit actual cost or rollback on failure
    - Track spend history
    """

    def __init__(
        self,
        config: ScoutConfig,
        audit: Optional[AuditLog] = None,
    ):
        self.config = config
        self.audit = audit or get_audit()
        self._reservations: Dict[str, Reservation] = {}
        self._spent_this_hour: float = 0.0
        self._hour_start: float = time.time()

    def check(self, estimated_cost: float, operation: str) -> bool:
        """
        Check if budget allows operation without reserving.

        Returns True if within limits, False otherwise.
        """
        hourly_budget = self._get_hourly_budget()
        current_spend = self._get_current_spend()
        remaining = hourly_budget - current_spend

        # Check per-event limit
        max_per_event = self.config.effective_max_cost()
        if estimated_cost > max_per_event:
            return False

        # Check hourly budget (with reserved amounts)
        reserved = sum(r.estimated_cost for r in self._reservations.values() if not r.committed)
        available = remaining - reserved

        return available >= estimated_cost

    def reserve(
        self,
        estimated_cost: float,
        operation: str,
        timeout_seconds: Optional[int] = None,
    ) -> BudgetReservation:
        """
        Reserve budget for operation.

        Raises InsufficientBudgetError if insufficient budget.
        """
        hourly_budget = self._get_hourly_budget()
        current_spend = self._get_current_spend()
        remaining = hourly_budget - current_spend

        # Check per-event limit
        max_per_event = self.config.effective_max_cost()
        if estimated_cost > max_per_event:
            raise InsufficientBudgetError(
                estimated_cost, max_per_event, operation
            )

        # Check available (including existing reservations)
        reserved = sum(r.estimated_cost for r in self._reservations.values() if not r.committed)
        available = remaining - reserved

        if available < estimated_cost:
            raise InsufficientBudgetError(estimated_cost, available, operation)

        # Create reservation
        timeout = timeout_seconds or self.config.get_budget_config().get(
            "reservation_timeout_seconds", 30
        )
        reservation_id = str(uuid.uuid4())[:8]
        reservation = Reservation(
            id=reservation_id,
            operation=operation,
            estimated_cost=estimated_cost,
        )
        self._reservations[reservation_id] = reservation

        # Log reservation
        self.audit.log(
            "budget_reserved",
            operation=operation,
            estimated_cost=estimated_cost,
            reservation_id=reservation_id,
            available_budget=available - estimated_cost,
        )

        return BudgetReservation(reservation_id, self, timeout)

    def commit(self, reservation: Reservation, actual_cost: float) -> None:
        """
        Commit actual cost against reservation.

        If actual_cost exceeds estimated by more than allow_overage_percent,
        the reservation is extended but logged as overage.
        """
        if reservation.committed:
            return

        # Calculate overage
        allow_percent = self.config.get_budget_config().get("allow_overage_percent", 10)
        max_allowed = reservation.estimated_cost * (1 + allow_percent / 100)

        reservation.actual_cost = actual_cost
        reservation.committed = True
        reservation.committed_at = time.time()

        self._spent_this_hour += actual_cost

        # Log commit
        self.audit.log(
            "budget_committed",
            operation=reservation.operation,
            estimated_cost=reservation.estimated_cost,
            actual_cost=actual_cost,
            reservation_id=reservation.id,
            overage=actual_cost - reservation.estimated_cost if actual_cost > reservation.estimated_cost else 0,
        )

    def rollback(self, reservation: Reservation) -> None:
        """Release reserved budget without charging."""
        if reservation.committed:
            return

        # Remove from reservations
        reservation_id = reservation.id
        if reservation_id in self._reservations:
            del self._reservations[reservation_id]

        # Log rollback
        self.audit.log(
            "budget_rolled_back",
            operation=reservation.operation,
            estimated_cost=reservation.estimated_cost,
            reservation_id=reservation_id,
        )

    def get_remaining(self) -> float:
        """Current remaining hourly budget."""
        hourly_budget = self._get_hourly_budget()
        current_spend = self._get_current_spend()
        reserved = sum(r.estimated_cost for r in self._reservations.values() if not r.committed)
        return max(0, hourly_budget - current_spend - reserved)

    def get_spend_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get detailed spend history from audit."""
        # Query audit for budget-related events
        events = self.audit.query(
            since=None,  # This would need actual time filtering
            event_type="budget_committed",
        )

        history = []
        for event in events:
            if isinstance(event, dict):
                history.append({
                    "operation": event.get("operation"),
                    "estimated_cost": event.get("estimated_cost"),
                    "actual_cost": event.get("actual_cost"),
                    "timestamp": event.get("timestamp", 0),
                })

        return history

    def _get_hourly_budget(self) -> float:
        """Get the effective hourly budget."""
        return min(
            self.config.get("limits.hourly_budget") or 1.0,
            self.config.get("limits.hard_safety_cap") or 10.0,
        )

    def _get_current_spend(self) -> float:
        """Get current hour's total spend."""
        # Check if hour has rolled over
        hour_seconds = 3600
        if time.time() - self._hour_start >= hour_seconds:
            self._spent_this_hour = 0.0
            self._hour_start = time.time()

        return self._spent_this_hour + sum(
            r.actual_cost for r in self._reservations.values() if r.committed
        )


class BudgetReservation:
    """Context manager for budget reservation."""

    def __init__(self, reservation_id: str, service: BudgetService, timeout: int = 30):
        self.reservation_id = reservation_id
        self.service = service
        self.timeout = timeout
        self._reservation: Optional[Reservation] = None

    def __enter__(self) -> Reservation:
        self._reservation = self.service._reservations.get(self.reservation_id)
        if not self._reservation:
            raise BudgetError(f"Reservation {self.reservation_id} not found")
        return self._reservation

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            # Exception occurred - rollback
            if self._reservation:
                self.service.rollback(self._reservation)
            return False

        if self._reservation and not self._reservation.committed:
            # No exception but not committed - auto-commit with estimated cost
            self.service.commit(self._reservation, self._reservation.estimated_cost)

        return False


# === Cost Estimation Utilities ===

def estimate_tokens_for_file(path: Path) -> int:
    """Quick symbol/code size estimate for cost prediction."""
    try:
        if not path.exists():
            return TOKENS_PER_SMALL_FILE
        content = path.read_text(encoding="utf-8", errors="replace")
        # Rough: ~4 chars per token for code
        return max(100, min(len(content) // 4, 5000))
    except OSError:
        return TOKENS_PER_SMALL_FILE


def estimate_cascade_cost(files: List[Path]) -> float:
    """
    Predict cost BEFORE any LLM calls.
    Conservative estimate: over-estimate slightly to stay under budget.
    """
    token_estimate = sum(estimate_tokens_for_file(f) for f in files)
    base_cost = token_estimate * COST_PER_MILLION_8B / 1_000_000
    # Add 20% buffer for potential 70B escalations
    return base_cost * 1.2


# === Legacy Compatibility ===

def check_budget_with_message(
    config: ScoutConfig,
    estimated_cost: float = 0.01,
    audit: Optional[AuditLog] = None,
    model: Optional[str] = None,
) -> bool:
    """
    Legacy function for backward compatibility with router.py.

    Checks if operation can proceed within hourly budget.
    Returns True if OK, False if blocked.
    """
    service = BudgetService(config, audit)
    return service.check(estimated_cost, "legacy_check")


# === Module-level convenience ===

_default_service: Optional[BudgetService] = None


def get_budget_service(config: Optional[ScoutConfig] = None) -> BudgetService:
    """Get default budget service."""
    global _default_service
    if _default_service is None:
        from scout.config import ScoutConfig
        _default_service = BudgetService(config or ScoutConfig())
    return _default_service
