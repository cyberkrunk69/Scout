"""Comprehensive tests for plan_state module."""

import asyncio
import json
import logging
import os
import pytest
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from scout.plan_state import (
    PlanStateManager,
    generate_plan_id,
    acquire_lock,
    release_lock,
    is_locked,
    _get_lock_config,
    _acquire_lock_atomic,
    LOCKS_DIR,
    PLAN_STATE_DIR,
)


class TestLockFunctions:
    """Test lock acquisition and release functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.original_locks_dir = LOCKS_DIR
        # Patch LOCKS_DIR to use temp directory
        import scout.plan_state as plan_state
        plan_state.LOCKS_DIR = Path(self.test_dir) / "locks"
        # Ensure locks directory exists
        (Path(self.test_dir) / "locks").mkdir(parents=True, exist_ok=True)
        
        # Patch logger to avoid structlog-style logging issues
        self.mock_logger = MagicMock()
        plan_state.logger = self.mock_logger

    def teardown_method(self):
        """Clean up test fixtures."""
        import scout.plan_state as plan_state
        plan_state.LOCKS_DIR = self.original_locks_dir
        # Restore original logger
        plan_state.logger = logging.getLogger(__name__)

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_lock_config_defaults(self):
        """Test default lock configuration."""
        with patch.dict(os.environ, {}, clear=True):
            config = _get_lock_config()
            assert "timeout" in config
            assert "stale_hours" in config
            assert config["timeout"] == 30  # Default
            assert config["stale_hours"] == 1  # Default

    def test_get_lock_config_from_env(self):
        """Test lock configuration from environment variables."""
        with patch.dict(os.environ, {
            "SCOUT_PLAN_LOCK_TIMEOUT": "60",
            "SCOUT_PLAN_STALE_LOCK_HOURS": "2"
        }):
            config = _get_lock_config()
            assert config["timeout"] == 60
            assert config["stale_hours"] == 2

    def test_acquire_lock_atomic_success(self):
        """Test successful atomic lock acquisition."""
        lock_path = Path(self.test_dir) / "locks" / "test_plan.lock"
        result = _acquire_lock_atomic(lock_path)
        assert result is True
        assert lock_path.exists()

    def test_acquire_lock_atomic_already_exists(self):
        """Test atomic lock acquisition when lock already exists."""
        lock_path = Path(self.test_dir) / "locks" / "test_plan.lock"
        lock_path.mkdir(parents=True)
        
        result = _acquire_lock_atomic(lock_path)
        assert result is False

    def test_acquire_lock_atomic_os_error(self):
        """Test atomic lock acquisition with OS error."""
        lock_path = Path("/root/locks/test.lock")  # Permission denied
        
        result = _acquire_lock_atomic(lock_path)
        assert result is False

    def test_acquire_lock_success(self):
        """Test successful lock acquisition."""
        plan_id = "test_plan_123"
        result = acquire_lock(plan_id, timeout=5)
        assert result is True
        assert is_locked(plan_id)

    def test_acquire_lock_twice_fails(self):
        """Test that acquiring lock twice fails."""
        plan_id = "test_plan_456"
        result1 = acquire_lock(plan_id, timeout=5)
        assert result1 is True
        
        # Try to acquire again - should fail
        result2 = acquire_lock(plan_id, timeout=1)
        assert result2 is False

    def test_release_lock_success(self):
        """Test successful lock release."""
        plan_id = "test_plan_789"
        acquire_lock(plan_id, timeout=5)
        
        result = release_lock(plan_id)
        assert result is True
        assert not is_locked(plan_id)

    def test_release_lock_not_exists(self):
        """Test releasing non-existent lock."""
        # Note: The current implementation returns True even if lock doesn't exist
        # as long as no exception is raised
        result = release_lock("nonexistent_plan")
        assert result is True  # Current behavior: returns True

    def test_is_locked_exists(self):
        """Test is_locked returns True for existing lock."""
        plan_id = "test_plan_is_locked"
        acquire_lock(plan_id, timeout=5)
        assert is_locked(plan_id) is True

    def test_is_locked_not_exists(self):
        """Test is_locked returns False for non-existent lock."""
        assert is_locked("nonexistent_plan") is False


class TestGeneratePlanId:
    """Test plan ID generation."""

    def test_generate_plan_id_basic(self):
        """Test basic plan ID generation."""
        plan_id = generate_plan_id("test request")
        assert plan_id.startswith("testrequest_")
        assert len(plan_id) > len("testrequest_")

    def test_generate_plan_id_unique(self):
        """Test that generated IDs are unique."""
        ids = [generate_plan_id("same request") for _ in range(10)]
        assert len(set(ids)) == 10  # All unique

    def test_generate_plan_id_special_chars(self):
        """Test plan ID generation with special characters."""
        plan_id = generate_plan_id("test @#$% request!")
        # Should only contain alphanumeric from prefix
        prefix = plan_id.split("_")[0]
        assert prefix.isalnum()

    def test_generate_plan_id_long_request(self):
        """Test plan ID generation with long request."""
        long_request = "a" * 100
        plan_id = generate_plan_id(long_request)
        # Prefix should only be 20 chars
        prefix = plan_id.split("_")[0]
        assert len(prefix) <= 20


class TestPlanStateManager:
    """Test PlanStateManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.repo_root = Path(self.test_dir)

        # Patch the directory functions to use our test dir
        import scout.plan_state as plan_state
        self.original_active = plan_state._active_dir
        self.original_completed = plan_state._completed_dir
        self.original_archived = plan_state._archived_dir
        
        # Patch logger to avoid structlog-style logging issues
        self.mock_logger = MagicMock()
        self.original_logger = plan_state.logger
        plan_state.logger = self.mock_logger

        # Lambda that mimics the original functions (including mkdir)
        plan_state._active_dir = lambda r: (r / ".scout" / "plans" / "active").mkdir(parents=True, exist_ok=True) or (r / ".scout" / "plans" / "active")
        plan_state._completed_dir = lambda r: (r / ".scout" / "plans" / "completed").mkdir(parents=True, exist_ok=True) or (r / ".scout" / "plans" / "completed")
        plan_state._archived_dir = lambda r: (r / ".scout" / "plans" / "archived").mkdir(parents=True, exist_ok=True) or (r / ".scout" / "plans" / "archived")

    def teardown_method(self):
        """Clean up test fixtures."""
        import scout.plan_state as plan_state
        plan_state._active_dir = self.original_active
        plan_state._completed_dir = self.original_completed
        plan_state._archived_dir = self.original_archived
        # Restore original logger
        plan_state.logger = self.original_logger
        
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @pytest.mark.asyncio
    async def test_save_plan_state(self):
        """Test saving plan state."""
        manager = PlanStateManager(self.repo_root)
        
        # Create a mock context
        context = Mock()
        context.request = "Test request"
        context.depth = 0
        context.max_depth = 3
        context.summary = "Test summary"
        context.parent_goals = []
        context.constraints = []
        context.discoveries = []
        context.pivots_needed = []
        context.sub_plan_outcomes = []
        context.is_pivoting = False
        context.pivot_reason = ""
        context.replan_required = False
        
        plan_id = await manager.save_plan_state(context)
        
        assert plan_id is not None
        assert plan_id.startswith("testrequest_")
        
        # Check file was created
        plan_file = self.repo_root / ".scout" / "plans" / "active" / f"{plan_id}.json"
        assert plan_file.exists()
        
        # Check content
        data = json.loads(plan_file.read_text())
        assert data["request"] == "Test request"
        assert data["depth"] == 0

    @pytest.mark.asyncio
    async def test_load_plan_state(self):
        """Test loading plan state."""
        manager = PlanStateManager(self.repo_root)
        
        # Create and save a context
        context = Mock()
        context.request = "Load test request"
        context.depth = 1
        context.max_depth = 5
        context.summary = "Test summary"
        context.parent_goals = ["goal1"]
        context.constraints = ["constraint1"]
        context.discoveries = ["discovery1"]
        context.pivots_needed = []
        context.sub_plan_outcomes = []
        context.is_pivoting = False
        context.pivot_reason = ""
        context.replan_required = False
        
        plan_id = await manager.save_plan_state(context)
        
        # Load the context
        loaded_context = await manager.load_plan_state(plan_id)
        
        assert loaded_context is not None
        assert loaded_context.request == "Load test request"
        assert loaded_context.depth == 1
        assert loaded_context.plan_id == plan_id

    @pytest.mark.asyncio
    async def test_load_plan_state_not_found(self):
        """Test loading non-existent plan returns None."""
        manager = PlanStateManager(self.repo_root)
        
        result = await manager.load_plan_state("nonexistent_plan_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_transition_plan_to_completed(self):
        """Test transitioning plan to completed state."""
        manager = PlanStateManager(self.repo_root)
        
        # Create and save a context
        context = Mock()
        context.request = "Transition test"
        context.depth = 0
        context.max_depth = 3
        context.summary = ""
        context.parent_goals = []
        context.constraints = []
        context.discoveries = []
        context.pivots_needed = []
        context.sub_plan_outcomes = []
        context.is_pivoting = False
        context.pivot_reason = ""
        context.replan_required = False
        
        plan_id = await manager.save_plan_state(context)
        
        # Transition to completed
        result = await manager.transition_plan(plan_id, "completed")
        
        assert result is True
        
        # Check plan is in completed directory
        completed_file = self.repo_root / ".scout" / "plans" / "completed" / f"{plan_id}.json"
        assert completed_file.exists()
        
        # Check original is gone
        active_file = self.repo_root / ".scout" / "plans" / "active" / f"{plan_id}.json"
        assert not active_file.exists()

    @pytest.mark.asyncio
    async def test_transition_plan_to_archived(self):
        """Test transitioning plan to archived state."""
        manager = PlanStateManager(self.repo_root)
        
        context = Mock()
        context.request = "Archive test"
        context.depth = 0
        context.max_depth = 3
        context.summary = ""
        context.parent_goals = []
        context.constraints = []
        context.discoveries = []
        context.pivots_needed = []
        context.sub_plan_outcomes = []
        context.is_pivoting = False
        context.pivot_reason = ""
        context.replan_required = False
        
        plan_id = await manager.save_plan_state(context)
        
        result = await manager.transition_plan(plan_id, "archived")
        
        assert result is True
        
        archived_file = self.repo_root / ".scout" / "plans" / "archived" / f"{plan_id}.json"
        assert archived_file.exists()

    @pytest.mark.asyncio
    async def test_transition_plan_invalid_state(self):
        """Test transitioning to invalid state fails."""
        manager = PlanStateManager(self.repo_root)
        
        context = Mock()
        context.request = "Invalid state test"
        context.depth = 0
        context.max_depth = 3
        context.summary = ""
        context.parent_goals = []
        context.constraints = []
        context.discoveries = []
        context.pivots_needed = []
        context.sub_plan_outcomes = []
        context.is_pivoting = False
        context.pivot_reason = ""
        context.replan_required = False
        
        plan_id = await manager.save_plan_state(context)
        
        result = await manager.transition_plan(plan_id, "invalid_state")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_transition_plan_missing_source(self):
        """Test transitioning missing plan fails."""
        manager = PlanStateManager(self.repo_root)
        
        result = await manager.transition_plan("nonexistent_plan", "completed")
        
        assert result is False

    @pytest.mark.asyncio
    async def test_list_active_plans(self):
        """Test listing active plans."""
        manager = PlanStateManager(self.repo_root)
        
        # Create multiple plans
        for i in range(3):
            context = Mock()
            context.request = f"Plan {i}"
            context.depth = 0
            context.max_depth = 3
            context.summary = ""
            context.parent_goals = []
            context.constraints = []
            context.discoveries = []
            context.pivots_needed = []
            context.sub_plan_outcomes = []
            context.is_pivoting = False
            context.pivot_reason = ""
            context.replan_required = False
            await manager.save_plan_state(context)
        
        active = manager.list_active_plans()
        assert len(active) >= 3

    @pytest.mark.asyncio
    async def test_list_completed_plans(self):
        """Test listing completed plans."""
        manager = PlanStateManager(self.repo_root)
        
        # Create and complete a plan
        context = Mock()
        context.request = "Completed plan"
        context.depth = 0
        context.max_depth = 3
        context.summary = ""
        context.parent_goals = []
        context.constraints = []
        context.discoveries = []
        context.pivots_needed = []
        context.sub_plan_outcomes = []
        context.is_pivoting = False
        context.pivot_reason = ""
        context.replan_required = False
        
        plan_id = await manager.save_plan_state(context)
        await manager.transition_plan(plan_id, "completed")
        
        completed = manager.list_completed_plans()
        assert plan_id in completed

    @pytest.mark.asyncio
    async def test_list_archived_plans(self):
        """Test listing archived plans."""
        manager = PlanStateManager(self.repo_root)
        
        context = Mock()
        context.request = "Archived plan"
        context.depth = 0
        context.max_depth = 3
        context.summary = ""
        context.parent_goals = []
        context.constraints = []
        context.discoveries = []
        context.pivots_needed = []
        context.sub_plan_outcomes = []
        context.is_pivoting = False
        context.pivot_reason = ""
        context.replan_required = False
        
        plan_id = await manager.save_plan_state(context)
        await manager.transition_plan(plan_id, "archived")
        
        archived = manager.list_archived_plans()
        assert plan_id in archived

    @pytest.mark.asyncio
    async def test_cleanup_old_plans_with_active(self):
        """Test cleanup is skipped when active plans exist."""
        manager = PlanStateManager(self.repo_root)
        
        # Create an active plan
        context = Mock()
        context.request = "Active plan"
        context.depth = 0
        context.max_depth = 3
        context.summary = ""
        context.parent_goals = []
        context.constraints = []
        context.discoveries = []
        context.pivots_needed = []
        context.sub_plan_outcomes = []
        context.is_pivoting = False
        context.pivot_reason = ""
        context.replan_required = False
        await manager.save_plan_state(context)
        
        result = manager.cleanup_old_plans()
        
        assert result["skipped"] is True
        assert result["reason"] == "active_plans_present"

    @pytest.mark.asyncio
    async def test_cleanup_old_plans_no_plans(self):
        """Test cleanup with no plans."""
        manager = PlanStateManager(self.repo_root)
        
        result = manager.cleanup_old_plans()
        
        assert result["skipped"] is False
        assert result["archived"] == 0
        assert result["deleted"] == 0


class TestGetPlanStateManager:
    """Test get_plan_state_manager function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Patch logger to avoid structlog-style logging issues
        import scout.plan_state as plan_state
        self.mock_logger = MagicMock()
        self.original_logger = plan_state.logger
        plan_state.logger = self.mock_logger

    def teardown_method(self):
        """Clean up test fixtures."""
        import scout.plan_state as plan_state
        plan_state._plan_state_manager = None
        # Restore original logger
        plan_state.logger = self.original_logger
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_get_plan_state_manager_default(self):
        """Test getting manager with default root."""
        import scout.plan_state as plan_state
        plan_state._plan_state_manager = None
        
        # Change to test directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.test_dir)
            manager = plan_state.get_plan_state_manager()
            assert manager is not None
            assert isinstance(manager, PlanStateManager)
        finally:
            os.chdir(original_cwd)

    def test_get_plan_state_manager_with_path(self):
        """Test getting manager with specified root."""
        import scout.plan_state as plan_state
        plan_state._plan_state_manager = None
        
        manager = plan_state.get_plan_state_manager(self.test_dir)
        assert manager is not None
        assert str(manager.repo_root) == str(Path(self.test_dir))

    def test_get_plan_state_manager_singleton(self):
        """Test that manager is a singleton."""
        import scout.plan_state as plan_state
        plan_state._plan_state_manager = None
        
        manager1 = plan_state.get_plan_state_manager(self.test_dir)
        manager2 = plan_state.get_plan_state_manager(self.test_dir)
        
        assert manager1 is manager2


class TestPlanStateManagerErrorPaths:
    """Test PlanStateManager error handling paths."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.repo_root = Path(self.test_dir)

        import scout.plan_state as plan_state
        self.original_active = plan_state._active_dir
        self.original_completed = plan_state._completed_dir
        self.original_archived = plan_state._archived_dir
        self.mock_logger = MagicMock()
        self.original_logger = plan_state.logger
        plan_state.logger = self.mock_logger

        plan_state._active_dir = lambda r: (r / ".scout" / "plans" / "active").mkdir(parents=True, exist_ok=True) or (r / ".scout" / "plans" / "active")
        plan_state._completed_dir = lambda r: (r / ".scout" / "plans" / "completed").mkdir(parents=True, exist_ok=True) or (r / ".scout" / "plans" / "completed")
        plan_state._archived_dir = lambda r: (r / ".scout" / "plans" / "archived").mkdir(parents=True, exist_ok=True) or (r / ".scout" / "plans" / "archived")

    def teardown_method(self):
        """Clean up test fixtures."""
        import scout.plan_state as plan_state
        plan_state._active_dir = self.original_active
        plan_state._completed_dir = self.original_completed
        plan_state._archived_dir = self.original_archived
        plan_state.logger = self.original_logger
        
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @pytest.mark.asyncio
    async def test_load_plan_state_corrupted(self):
        """Test loading corrupted plan state returns None."""
        manager = PlanStateManager(self.repo_root)
        
        # Create a corrupted plan file
        active_dir = self.repo_root / ".scout" / "plans" / "active"
        active_dir.mkdir(parents=True, exist_ok=True)
        plan_file = active_dir / "corrupted_plan.json"
        plan_file.write_text("{ invalid json }")
        
        result = await manager.load_plan_state("corrupted_plan")
        
        assert result is None
        
        # Check it was archived
        archived_file = self.repo_root / ".scout" / "plans" / "archived" / "corrupted_plan.json"
        assert archived_file.exists()

    @pytest.mark.asyncio
    async def test_transition_plan_lock_failed(self):
        """Test transition fails when lock cannot be acquired."""
        manager = PlanStateManager(self.repo_root)
        
        # Create and save a context
        context = Mock()
        context.request = "Lock test"
        context.depth = 0
        context.max_depth = 3
        context.summary = ""
        context.parent_goals = []
        context.constraints = []
        context.discoveries = []
        context.pivots_needed = []
        context.sub_plan_outcomes = []
        context.is_pivoting = False
        context.pivot_reason = ""
        context.replan_required = False
        
        plan_id = await manager.save_plan_state(context)
        
        # Acquire the lock to block transition
        import scout.plan_state as plan_state
        plan_state.acquire_lock(plan_id, timeout=30)
        
        try:
            result = await manager.transition_plan(plan_id, "completed")
            assert result is False
        finally:
            plan_state.release_lock(plan_id)
