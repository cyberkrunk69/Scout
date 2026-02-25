"""Tests for Budget Service."""

import pytest
from unittest.mock import MagicMock, patch

from scout.llm.budget import (
    BudgetService,
    BudgetReservation,
    BudgetError,
    InsufficientBudgetError,
    Reservation,
)


def test_reservation_creation():
    """Test Reservation dataclass."""
    reservation = Reservation(
        id="test123",
        operation="test_op",
        estimated_cost=0.01,
    )
    
    assert reservation.id == "test123"
    assert reservation.operation == "test_op"
    assert reservation.estimated_cost == 0.01
    assert reservation.committed is False
    assert reservation.actual_cost == 0.0


def test_budget_error_exceptions():
    """Test budget error classes."""
    # Test InsufficientBudgetError
    err = InsufficientBudgetError(requested=0.05, available=0.01, operation="test")
    assert "test" in str(err)
    assert err.requested == 0.05
    assert err.available == 0.01


@pytest.fixture
def mock_config():
    """Create a mock ScoutConfig."""
    config = MagicMock()
    config.effective_max_cost.return_value = 0.10
    config.get_budget_config.return_value = {
        "reservation_timeout_seconds": 30,
        "allow_overage_percent": 10,
    }
    config.get.side_effect = lambda key: {
        "limits.hourly_budget": 1.0,
        "limits.hard_safety_cap": 10.0,
    }.get(key)
    return config


@pytest.fixture
def mock_audit():
    """Create a mock AuditLog."""
    audit = MagicMock()
    audit.log = MagicMock()
    audit.query.return_value = []
    return audit


def test_budget_service_check(mock_config, mock_audit):
    """Test budget check method."""
    service = BudgetService(config=mock_config, audit=mock_audit)
    
    # Should return True when within budget
    assert service.check(0.01, "test_op") is True
    
    # Should return False when over per-event limit
    mock_config.effective_max_cost.return_value = 0.05
    assert service.check(0.10, "test_op") is False


def test_budget_service_reserve(mock_config, mock_audit):
    """Test budget reservation."""
    service = BudgetService(config=mock_config, audit=mock_audit)
    
    budget_res = service.reserve(0.01, "test_op")
    
    assert isinstance(budget_res, BudgetReservation)
    assert budget_res.reservation_id is not None
    assert mock_audit.log.called


def test_budget_service_insufficient_budget(mock_config, mock_audit):
    """Test insufficient budget error."""
    service = BudgetService(config=mock_config, audit=mock_audit)
    
    # Set up to fail: estimated cost > available
    mock_config.get.side_effect = lambda key: {
        "limits.hourly_budget": 0.01,
        "limits.hard_safety_cap": 10.0,
    }.get(key)
    
    with pytest.raises(InsufficientBudgetError):
        service.reserve(0.05, "test_op")


def test_budget_reservation_context_manager(mock_config, mock_audit):
    """Test BudgetReservation as context manager."""
    service = BudgetService(config=mock_config, audit=mock_audit)
    
    budget_res = service.reserve(0.01, "test_op")
    
    # Test __enter__
    with budget_res as res:
        assert res is not None
        assert res.estimated_cost == 0.01
    
    # After exiting without exception, should auto-commit
    assert res.committed is True


def test_budget_reservation_exception_rollback(mock_config, mock_audit):
    """Test that exceptions cause rollback."""
    service = BudgetService(config=mock_config, audit=mock_audit)
    
    budget_res = service.reserve(0.01, "test_op")
    reservation_id = budget_res.reservation_id
    
    # Test __exit__ with exception
    try:
        with budget_res:
            raise ValueError("test error")
    except ValueError:
        pass
    
    # Should have rolled back
    assert reservation_id not in service._reservations


def test_budget_service_get_remaining(mock_config, mock_audit):
    """Test get_remaining method."""
    service = BudgetService(config=mock_config, audit=mock_audit)
    
    remaining = service.get_remaining()
    assert remaining >= 0


def test_budget_service_get_spend_history(mock_config, mock_audit):
    """Test get_spend_history method."""
    service = BudgetService(config=mock_config, audit=mock_audit)
    
    history = service.get_spend_history(hours=1)
    assert isinstance(history, list)
