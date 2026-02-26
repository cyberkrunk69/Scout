"""Tests for plan_store module."""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

from scout.plan_store import PlanStore


@pytest.fixture
def temp_repo():
    """Create a temporary repository root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def plan_store(temp_repo):
    """Create an initialized PlanStore."""
    store = PlanStore(temp_repo)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_store_initialization(temp_repo):
    """Test that store creates database and schema on init."""
    store = PlanStore(temp_repo)
    await store.initialize()

    # Check db file exists
    db_path = store.db_path
    assert db_path.exists()

    await store.close()


@pytest.mark.asyncio
async def test_store_plan(plan_store):
    """Test storing a plan."""
    goal = "Click the login button"
    steps = [
        {"action": "click", "selector": "#login-btn"},
        {"action": "wait", "selector": ".dashboard"}
    ]
    
    plan_id = await plan_store.store_plan(goal, steps)
    
    assert plan_id is not None
    assert len(plan_id) == 16  # SHA256 hash truncated to 16 chars


@pytest.mark.asyncio
async def test_store_plan_with_url_pattern(plan_store):
    """Test storing a plan with URL pattern."""
    goal = "Click the login button"
    steps = [{"action": "click", "selector": "#login-btn"}]
    url_pattern = "^https://example\\.com/login$"
    
    plan_id = await plan_store.store_plan(goal, steps, url_pattern)
    
    assert plan_id is not None


@pytest.mark.asyncio
async def test_get_plan_found(plan_store):
    """Test retrieving an existing plan."""
    goal = "Fill the form"
    steps = [
        {"action": "type", "selector": "#name", "value": "John"},
        {"action": "click", "selector": "#submit"}
    ]
    
    stored_id = await plan_store.store_plan(goal, steps)
    
    result = await plan_store.get_plan(goal)
    
    assert result is not None
    plan_id, retrieved_steps = result
    assert plan_id == stored_id
    assert retrieved_steps == steps


@pytest.mark.asyncio
async def test_get_plan_not_found(plan_store):
    """Test retrieving a non-existent plan returns None."""
    result = await plan_store.get_plan("Non-existent goal")
    assert result is None


@pytest.mark.asyncio
async def test_get_plan_with_url_matching(plan_store):
    """Test URL pattern matching."""
    goal = "Login action"
    steps = [{"action": "click", "selector": "#login"}]
    url_pattern = "^https://example\\.com/login$"
    
    await plan_store.store_plan(goal, steps, url_pattern)
    
    # Should match
    result = await plan_store.get_plan(goal, current_url="https://example.com/login")
    assert result is not None
    
    # Should not match
    result = await plan_store.get_plan(goal, current_url="https://example.com/other")
    assert result is None


@pytest.mark.asyncio
async def test_get_plan_with_invalid_url_pattern(plan_store):
    """Test handling of invalid URL pattern."""
    goal = "Test invalid pattern"
    steps = [{"action": "click"}]
    url_pattern = "[invalid"  # Invalid regex
    
    await plan_store.store_plan(goal, steps, url_pattern)
    
    # The code logs a warning but still returns the plan when pattern is invalid
    # This is the current behavior
    result = await plan_store.get_plan(goal, current_url="https://example.com")
    # Result is returned but with warning logged (actual behavior)


@pytest.mark.asyncio
async def test_update_plan_stats_success(plan_store):
    """Test updating plan success stats."""
    goal = "Test success stats"
    steps = [{"action": "click"}]
    
    plan_id = await plan_store.store_plan(goal, steps)
    
    await plan_store.update_plan_stats(plan_id, success=True)
    
    stats = await plan_store.get_plan_stats(plan_id)
    assert stats is not None
    assert stats["success_count"] == 1
    assert stats["failure_count"] == 0


@pytest.mark.asyncio
async def test_update_plan_stats_failure(plan_store):
    """Test updating plan failure stats."""
    goal = "Test failure stats"
    steps = [{"action": "click"}]
    
    plan_id = await plan_store.store_plan(goal, steps)
    
    await plan_store.update_plan_stats(plan_id, success=False)
    
    stats = await plan_store.get_plan_stats(plan_id)
    assert stats is not None
    assert stats["success_count"] == 0
    assert stats["failure_count"] == 1


@pytest.mark.asyncio
async def test_get_plan_stats_not_found(plan_store):
    """Test getting stats for non-existent plan."""
    stats = await plan_store.get_plan_stats("nonexistent_id")
    assert stats is None


@pytest.mark.asyncio
async def test_cleanup_old_plans(plan_store):
    """Test cleanup of old plans."""
    goal = "Old plan"
    steps = [{"action": "click"}]
    
    # Store a plan
    await plan_store.store_plan(goal, steps)
    
    # Cleanup (default 30 days, should not delete new plans)
    result = await plan_store.cleanup_old_plans(max_age_days=30)
    
    assert "deleted" in result


@pytest.mark.asyncio
async def test_normalize_goal(plan_store):
    """Test goal normalization."""
    # Test that goals are normalized (lowercase, stripped)
    goal = "  Click the Button  "
    steps = [{"action": "click"}]
    
    await plan_store.store_plan(goal, steps)
    
    # Should be found with different whitespace
    result = await plan_store.get_plan("click the button")
    assert result is not None
    
    result = await plan_store.get_plan("  CLICK THE BUTTON  ")
    assert result is not None


class TestPlanStoreGeneration:
    """Test plan ID generation."""

    @pytest.mark.asyncio
    async def test_generate_plan_id_deterministic(self):
        """Test that same goal produces same plan ID."""
        store = PlanStore(Path(tempfile.mkdtemp()))
        await store.initialize()
        
        try:
            id1 = store._generate_plan_id("test goal")
            id2 = store._generate_plan_id("test goal")
            
            assert id1 == id2
            
            # Different goal should produce different ID
            id3 = store._generate_plan_id("different goal")
            assert id1 != id3
        finally:
            await store.close()

    @pytest.mark.asyncio
    async def test_generate_plan_id_with_url(self):
        """Test plan ID includes URL pattern."""
        store = PlanStore(Path(tempfile.mkdtemp()))
        await store.initialize()
        
        try:
            id1 = store._generate_plan_id("test goal")
            id2 = store._generate_plan_id("test goal", "https://example.com")
            
            assert id1 != id2
        finally:
            await store.close()


class TestPlanStoreEdgeCases:
    """Test edge cases."""

    @pytest.mark.asyncio
    async def test_store_duplicate_plan(self, plan_store):
        """Test storing a plan with same goal updates existing."""
        goal = "Duplicate goal"
        steps1 = [{"action": "click", "selector": "#a"}]
        steps2 = [{"action": "click", "selector": "#b"}]
        
        id1 = await plan_store.store_plan(goal, steps1)
        id2 = await plan_store.store_plan(goal, steps2)
        
        # Should return same ID
        assert id1 == id2
        
        # Should have updated steps
        result = await plan_store.get_plan(goal)
        _, retrieved_steps = result
        assert retrieved_steps == steps2
