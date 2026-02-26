"""
Cost Controller for the Adaptive Engine.

Provides budget management for execution, tracking reservations and spending
across multiple components. This is a skeleton for Phase 1 that will be
expanded in later phases.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Reservation:
    """Represents a reserved budget allocation.
    
    Attributes:
        id: Unique reservation identifier
        amount: Reserved amount in USD
        spent: Amount already spent from this reservation
        created_at: ISO timestamp of reservation creation
    """
    id: str
    amount: float
    spent: float = 0.0
    created_at: str = field(default_factory=lambda: __import__("datetime").datetime.utcnow().isoformat())


class CostController:
    """
    Manages budget reservations and spending for the adaptive engine.
    
    Provides a reservation-based system where components can reserve budget
    before spending, enabling rollback of unspent allocations.
    
    Phase 1: Minimal implementation with reserve/spend/rollback.
    Later phases will add: quota enforcement, cost prediction, logging.
    """

    def __init__(self, total_budget: Optional[float] = None):
        """Initialize the cost controller.
        
        Args:
            total_budget: Optional total budget cap. If None, no cap is enforced.
        """
        self.total_budget = total_budget
        self._reservations: dict[str, Reservation] = {}
        self._total_spent: float = 0.0

    def reserve(self, amount: float) -> str:
        """Reserve a budget allocation.
        
        Args:
            amount: Amount to reserve in USD
            
        Returns:
            Reservation ID that can be used to spend or rollback
            
        Raises:
            ValueError: If amount is negative or exceeds available budget
        """
        if amount < 0:
            raise ValueError("Reservation amount must be non-negative")

        # Check total budget cap
        if self.total_budget is not None:
            available = self.total_budget - self._total_spent
            reserved = sum(r.amount - r.spent for r in self._reservations.values())
            available -= reserved
            if amount > available:
                raise ValueError(
                    f"Cannot reserve ${amount:.2f}: only ${available:.2f} available "
                    f"(total: ${self.total_budget:.2f}, spent: ${self._total_spent:.2f})"
                )

        reservation_id = f"res_{uuid.uuid4().hex[:12]}"
        self._reservations[reservation_id] = Reservation(
            id=reservation_id,
            amount=amount,
        )
        return reservation_id

    def spend(self, reservation_id: str, amount: float) -> None:
        """Record spending against a reservation.
        
        Args:
            reservation_id: ID returned by reserve()
            amount: Amount spent
            
        Raises:
            ValueError: If reservation doesn't exist or exceeds reservation
        """
        if reservation_id not in self._reservations:
            raise ValueError(f"Reservation {reservation_id} not found")

        reservation = self._reservations[reservation_id]

        if amount < 0:
            raise ValueError("Spend amount must be non-negative")

        new_spent = reservation.spent + amount
        if new_spent > reservation.amount:
            raise ValueError(
                f"Cannot spend ${amount:.2f}: reservation only has "
                f"${reservation.amount - reservation.spent:.2f} remaining"
            )

        reservation.spent = new_spent
        self._total_spent += amount

    def rollback(self, reservation_id: str) -> float:
        """Rollback an unused reservation, returning available amount.
        
        Args:
            reservation_id: ID returned by reserve()
            
        Returns:
            Amount that was returned to available budget
            
        Raises:
            ValueError: If reservation doesn't exist
        """
        if reservation_id not in self._reservations:
            raise ValueError(f"Reservation {reservation_id} not found")

        reservation = self._reservations[reservation_id]
        unspent = reservation.amount - reservation.spent
        del self._reservations[reservation_id]
        return unspent

    def get_available(self) -> float:
        """Get currently available budget.
        
        Returns:
            Available amount in USD
        """
        if self.total_budget is None:
            # Return infinity-like value (very large number)
            return float("inf")

        reserved = sum(r.amount - r.spent for r in self._reservations.values())
        return self.total_budget - self._total_spent - reserved

    def get_status(self) -> dict:
        """Get current budget status.
        
        Returns:
            Dict with total_budget, total_spent, available, reservation_count
        """
        return {
            "total_budget": self.total_budget,
            "total_spent": self._total_spent,
            "available": self.get_available(),
            "reservation_count": len(self._reservations),
        }


__all__ = [
    "CostController",
    "Reservation",
]
