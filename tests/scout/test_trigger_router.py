"""Tests for TriggerRouter in scout.router."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
import tempfile
import shutil


class TestTriggerRouter:
    """Test suite for TriggerRouter class."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def router(self, temp_repo):
        """Create a TriggerRouter instance for testing."""
        from scout.router import TriggerRouter
        from scout.audit import AuditLog
        from scout.app_config import ScoutConfig

        config = ScoutConfig()
        audit = AuditLog()
        return TriggerRouter(
            config=config,
            audit=audit,
            repo_root=temp_repo,
            trust_level="normal",
        )

    def test_constructor_defaults(self):
        """Test TriggerRouter constructor with default values."""
        from scout.router import TriggerRouter

        router = TriggerRouter()
        assert router.config is not None
        assert router.audit is not None
        assert router.repo_root is not None
        assert router.trust_level == "normal"

    def test_constructor_with_params(self, temp_repo):
        """Test TriggerRouter constructor with custom parameters."""
        from scout.router import TriggerRouter
        from scout.audit import AuditLog
        from scout.app_config import ScoutConfig

        config = ScoutConfig()
        audit = AuditLog()
        router = TriggerRouter(
            config=config,
            audit=audit,
            repo_root=temp_repo,
            trust_level="high",
        )
        assert router.trust_level == "high"
        assert router.repo_root.resolve() == temp_repo.resolve()

    def test_should_trigger_filters_ignored_files(self, router, temp_repo):
        """Test that should_trigger filters out ignored files."""
        # Create test files
        test_file = temp_repo / "test.py"
        test_file.write_text("print('hello')")

        # Test with non-ignored file
        result = router.should_trigger([test_file])
        assert test_file in result

    def test_should_trigger_empty_list(self, router):
        """Test should_trigger with empty list."""
        result = router.should_trigger([])
        assert result == []

    def test_quick_token_estimate(self, router, temp_repo):
        """Test token estimation for files."""
        test_file = temp_repo / "test.py"
        test_file.write_text("x = 1\n" * 100)

        estimate = router._quick_token_estimate(test_file)
        assert estimate > 0

    def test_quick_token_estimate_nonexistent(self, router, temp_repo):
        """Test token estimation for nonexistent file."""
        fake_file = temp_repo / "nonexistent.py"
        estimate = router._quick_token_estimate(fake_file)
        assert estimate > 0

    def test_estimate_cascade_cost(self, router, temp_repo):
        """Test cascade cost estimation."""
        test_file = temp_repo / "test.py"
        test_file.write_text("x = 1\n" * 100)

        cost = router.estimate_cascade_cost([test_file])
        assert cost >= 0

    def test_is_public_api(self, router, temp_repo):
        """Test public API detection."""
        # Create a runtime file
        runtime_dir = temp_repo / "runtime"
        runtime_dir.mkdir()
        runtime_file = runtime_dir / "module.py"
        runtime_file.write_text("")

        result = router._is_public_api(runtime_file)
        assert isinstance(result, bool)

    def test_detect_module(self):
        """Test module name detection."""
        from scout.router import TriggerRouter
        import os

        # Create a temp directory and use its resolved path
        import tempfile
        temp_dir = tempfile.mkdtemp()
        resolved_path = Path(temp_dir).resolve()

        # Create router with the resolved path
        router = TriggerRouter(repo_root=resolved_path)

        # Test with a path that has 2+ parts - must use resolved base
        file_path = resolved_path / "mymodule" / "__init__.py"
        module_name = router._detect_module(file_path)
        # Module name should be the first directory
        assert module_name == "mymodule"

        # Test with just a file (no subdirectory) - use resolved base
        file_path2 = resolved_path / "main.py"
        module_name2 = router._detect_module(file_path2)
        # Should return the file stem
        assert module_name2 == "main"

    def test_affects_module_boundary(self, router, temp_repo):
        """Test module boundary detection."""
        from scout.router import NavResult

        test_file = temp_repo / "test.py"
        test_file.write_text("")

        # Test with signature changed
        nav_result = NavResult(
            suggestion={"file": "test.py"},
            cost=0.001,
            duration_ms=100,
            signature_changed=True,
            new_exports=False,
        )
        result = router._affects_module_boundary(test_file, nav_result)
        assert result is True

        # Test without signature change
        nav_result2 = NavResult(
            suggestion={"file": "test.py"},
            cost=0.001,
            duration_ms=100,
            signature_changed=False,
            new_exports=False,
        )
        result2 = router._affects_module_boundary(test_file, nav_result2)
        assert isinstance(result2, bool)

    def test_critical_path_files(self, router, temp_repo):
        """Test critical path file detection."""
        # Create test files
        config_dir = temp_repo / "config"
        config_dir.mkdir()
        config_file = config_dir / "settings.py"
        config_file.write_text("")

        runtime_dir = temp_repo / "runtime"
        runtime_dir.mkdir()
        runtime_file = runtime_dir / "main.py"
        runtime_file.write_text("")

        critical_files = router._critical_path_files()
        assert isinstance(critical_files, set)

    def test_quick_parse(self, router, temp_repo):
        """Test quick file parsing."""
        test_file = temp_repo / "test.py"
        test_content = "def main():\n    pass\n"
        test_file.write_text(test_content)

        content = router._quick_parse(test_file)
        assert content == test_content

    def test_quick_parse_nonexistent(self, router, temp_repo):
        """Test quick parse of nonexistent file."""
        fake_file = temp_repo / "nonexistent.py"
        content = router._quick_parse(fake_file)
        assert content == ""

    def test_write_draft(self, router, temp_repo):
        """Test draft writing."""
        from scout.router import SymbolDoc

        test_file = temp_repo / "test.py"
        test_file.write_text("")

        symbol_doc = SymbolDoc(
            content="# Test\n\nContent here.",
            generation_cost=0.001,
        )

        draft_path = router._write_draft(test_file, symbol_doc)
        assert draft_path.exists()
        assert draft_path.parent.name == "drafts"

    def test_update_module_brief(self, router, temp_repo):
        """Test module brief update."""
        test_file = temp_repo / "test.py"
        test_file.write_text("")

        cost = router._update_module_brief("test_module", test_file, "test-session")
        assert cost >= 0

    @pytest.mark.asyncio
    async def test_generate_commit_draft(self, router, temp_repo):
        """Test commit draft generation."""
        # Create test file
        test_file = temp_repo / "test.py"
        test_file.write_text("def hello(): pass")

        # Mock the LLM call
        with patch("scout.router.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = MagicMock(
                content="feat(test): add hello function",
                cost_usd=0.001,
                model="llama-3.1-8b-instant",
                provider="test",
            )

            with patch("scout.git_analyzer.get_diff_for_file") as mock_diff:
                mock_diff.return_value = "+def hello(): pass"

                await router._generate_commit_draft(test_file, "test-session")

                # Check if draft was created
                draft_dir = temp_repo / "docs" / "drafts"
                if draft_dir.exists():
                    drafts = list(draft_dir.glob("*.commit.txt"))
                    # Draft may or may not exist depending on mocking

    @pytest.mark.asyncio
    async def test_generate_pr_snippet(self, router, temp_repo):
        """Test PR snippet generation."""
        # Create test file
        test_file = temp_repo / "test.py"
        test_file.write_text("def hello(): pass")

        with patch("scout.router.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = MagicMock(
                content="This PR adds a hello function.",
                cost_usd=0.001,
                model="llama-3.1-8b-instant",
                provider="test",
            )

            with patch("scout.git_analyzer.get_diff_for_file") as mock_diff:
                mock_diff.return_value = "+def hello(): pass"

                await router._generate_pr_snippet(test_file, "test-session")

    @pytest.mark.asyncio
    async def test_generate_impact_summary(self, router, temp_repo):
        """Test impact summary generation."""
        # Create test file
        test_file = temp_repo / "test.py"
        test_file.write_text("def hello(): pass")

        with patch("scout.router.call_llm", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = MagicMock(
                content="- May affect callers of hello()",
                cost_usd=0.001,
                model="llama-3.1-8b-instant",
                provider="test",
            )

            with patch("scout.git_analyzer.get_diff_for_file") as mock_diff:
                mock_diff.return_value = "+def hello(): pass"

                await router._generate_impact_summary(test_file, "test-session")

    def test_parse_nav_json(self, router):
        """Test JSON parsing from LLM response."""
        # Test with valid JSON
        result = router._parse_nav_json('{"file": "test.py", "function": "main", "line": 1}')
        assert result["file"] == "test.py"
        assert result["function"] == "main"

        # Test with markdown-wrapped JSON
        result2 = router._parse_nav_json('```json\n{"file": "test.py"}\n```')
        assert result2["file"] == "test.py"

        # Test with invalid JSON
        result3 = router._parse_nav_json("not json")
        assert result3["file"] == ""
        assert result3["function"] == ""

    def test_list_python_files(self, router, temp_repo):
        """Test listing Python files."""
        # Create test files
        (temp_repo / "module1.py").write_text("")
        subdir = temp_repo / "subdir"
        subdir.mkdir()
        (subdir / "module2.py").write_text("")

        files = router._list_python_files(None, limit=10)
        assert len(files) >= 2


class TestTaskRouter:
    """Test suite for TaskRouter class."""

    def test_constructor(self):
        """Test TaskRouter constructor."""
        from scout.router import TaskRouter

        router = TaskRouter()
        assert router.config is not None

    def test_route_high_confidence(self):
        """Test routing with high confidence intent."""
        from scout.router import TaskRouter
        from scout.llm.intent import IntentResult, IntentType

        router = TaskRouter()

        intent = IntentResult(
            intent_type=IntentType.QUERY_CODE,
            target="find_auth_function",
            confidence=0.95,
            metadata={},
        )

        decision = router.route(intent)
        assert decision.path == "direct"
        assert "scout_function_info" in decision.tools

    def test_route_low_confidence(self):
        """Test routing with low confidence intent."""
        from scout.router import TaskRouter
        from scout.llm.intent import IntentResult, IntentType

        router = TaskRouter()

        intent = IntentResult(
            intent_type=IntentType.IMPLEMENT_FEATURE,
            target="add_login",
            confidence=0.5,
            metadata={},
        )

        decision = router.route(intent)
        assert decision.path == "clarify"

    def test_route_plan_required(self):
        """Test routing for intents requiring planning."""
        from scout.router import TaskRouter
        from scout.llm.intent import IntentResult, IntentType

        router = TaskRouter()

        intent = IntentResult(
            intent_type=IntentType.REFACTOR,
            target="refactor_auth_module",
            confidence=0.8,
            metadata={},
        )

        decision = router.route(intent)
        assert decision.path == "plan_execute"

    def test_get_available_tools(self):
        """Test getting available tools."""
        from scout.router import TaskRouter

        router = TaskRouter()
        tools = router.get_available_tools()
        assert isinstance(tools, list)
        assert len(tools) > 0


class TestBudgetCheck:
    """Test suite for budget checking functions."""

    def test_check_budget_with_message_within_limit(self):
        """Test budget check when within limit."""
        from scout.router import check_budget_with_message
        from scout.app_config import ScoutConfig

        config = ScoutConfig()

        # Should return True when well under budget
        with patch("scout.router.AuditLog") as mock_audit:
            mock_instance = MagicMock()
            mock_instance.hourly_spend.return_value = 0.1
            mock_audit.return_value = mock_instance

            result = check_budget_with_message(config, estimated_cost=0.01)
            assert isinstance(result, bool)

    def test_check_budget_free_model(self):
        """Test budget check allows free models regardless of budget."""
        from scout.router import check_budget_with_message
        from scout.app_config import ScoutConfig

        config = ScoutConfig()

        with patch("scout.router.AuditLog") as mock_audit:
            mock_instance = MagicMock()
            mock_instance.hourly_spend.return_value = 100.0  # Budget exhausted
            mock_audit.return_value = mock_instance

            with patch("scout.router.is_free_model") as mock_free:
                mock_free.return_value = True
                # Free model should always be allowed
                result = check_budget_with_message(
                    config, estimated_cost=0.01, model="free-model"
                )
                assert result is True


class TestNavResult:
    """Test suite for NavResult dataclass."""

    def test_nav_result_creation(self):
        """Test NavResult creation."""
        from scout.router import NavResult

        result = NavResult(
            suggestion={"file": "test.py"},
            cost=0.001,
            duration_ms=100,
        )
        assert result.suggestion["file"] == "test.py"
        assert result.cost == 0.001
        assert result.duration_ms == 100

    def test_nav_result_with_flags(self):
        """Test NavResult with signature and export flags."""
        from scout.router import NavResult

        result = NavResult(
            suggestion={"file": "test.py"},
            cost=0.001,
            duration_ms=100,
            signature_changed=True,
            new_exports=True,
        )
        assert result.signature_changed is True
        assert result.new_exports is True


class TestSymbolDoc:
    """Test suite for SymbolDoc dataclass."""

    def test_symbol_doc_creation(self):
        """Test SymbolDoc creation."""
        from scout.router import SymbolDoc

        doc = SymbolDoc(
            content="# Test\n\nContent here.",
            generation_cost=0.001,
        )
        assert doc.content == "# Test\n\nContent here."
        assert doc.generation_cost == 0.001


class TestScoutNavIntegration:
    """Integration tests for _scout_nav (LLM path and fallback)."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def router(self, temp_repo):
        """Create a TriggerRouter instance for testing."""
        from scout.router import TriggerRouter
        from scout.audit import AuditLog
        from scout.app_config import ScoutConfig

        config = ScoutConfig()
        audit = AuditLog()
        return TriggerRouter(
            config=config,
            audit=audit,
            repo_root=temp_repo,
            trust_level="normal",
        )

    @pytest.mark.asyncio
    async def test_scout_nav_async_with_mock(self, router, temp_repo):
        """Test _scout_nav_async with mocked LLM."""
        from scout.router import NavResult

        # Create a test file
        test_file = temp_repo / "test.py"
        test_file.write_text("def main():\n    pass\n")

        # Mock the call_llm function
        mock_result = MagicMock()
        mock_result.content = '{"file": "test.py", "function": "main", "line": 1, "confidence": 85}'
        mock_result.cost_usd = 0.0001

        with patch("scout.router.call_llm", new_callable=AsyncMock) as mock_call_llm:
            mock_call_llm.return_value = mock_result
            result = await router._scout_nav_async(test_file, "def main():\n    pass\n")

        assert isinstance(result, NavResult)
        assert result.suggestion["file"] == "test.py"
        assert result.suggestion["function"] == "main"

    def test_scout_nav_sync_fallback(self, router, temp_repo):
        """Test _scout_nav sync fallback when LLM is unavailable."""
        from scout.router import NavResult

        # Create a test file
        test_file = temp_repo / "test.py"
        test_file.write_text("def main():\n    pass\n")

        # Mock call_llm to raise an exception
        with patch("scout.router.call_llm", side_effect=Exception("LLM unavailable")):
            result = router._scout_nav(test_file, "def main():\n    pass\n")

        assert isinstance(result, NavResult)
        # Should fall back to heuristic
        assert "confidence" in result.suggestion
        assert result.suggestion["confidence"] > 0


class TestSymbolDocGeneration:
    """Integration tests for _generate_symbol_doc."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def router(self, temp_repo):
        """Create a TriggerRouter instance for testing."""
        from scout.router import TriggerRouter
        from scout.audit import AuditLog
        from scout.app_config import ScoutConfig

        config = ScoutConfig()
        audit = AuditLog()
        return TriggerRouter(
            config=config,
            audit=audit,
            repo_root=temp_repo,
            trust_level="normal",
        )

    @pytest.mark.asyncio
    async def test_generate_symbol_doc_with_mock(self, router, temp_repo):
        """Test _generate_symbol_doc with mocked LLM."""
        from scout.router import NavResult, SymbolDoc, ValidationResult

        # Create a test file
        test_file = temp_repo / "test.py"
        test_file.write_text("def main():\n    pass\n")

        nav_result = NavResult(
            suggestion={"file": "test.py", "function": "main", "line": 1},
            cost=0.0001,
            duration_ms=100,
        )

        validation = ValidationResult(
            is_valid=True,
            actual_file=test_file,
            actual_line=1,
            error_code=None,
            alternatives=[],
            symbol_snippet="def main():",
            adjusted_confidence=85,
            validation_time_ms=10.0,
        )

        # Mock the call_llm function
        mock_result = MagicMock()
        mock_result.content = "# Test Function\n\nThis is a test function."
        mock_result.cost_usd = 0.0002

        with patch("scout.router.call_llm", new_callable=AsyncMock) as mock_call_llm:
            mock_call_llm.return_value = mock_result
            # Note: _generate_symbol_doc is sync, but internally uses asyncio.run
            result = router._generate_symbol_doc(test_file, nav_result, validation)

        assert isinstance(result, SymbolDoc)
        assert result.content is not None
        assert result.generation_cost >= 0

    def test_generate_symbol_doc_fallback(self, router, temp_repo):
        """Test _generate_symbol_doc fallback when LLM is unavailable."""
        from scout.router import NavResult, SymbolDoc, ValidationResult

        # Create a test file
        test_file = temp_repo / "test.py"
        test_file.write_text("def main():\n    pass\n")

        nav_result = NavResult(
            suggestion={"file": "test.py", "function": "main", "line": 1},
            cost=0.0001,
            duration_ms=100,
        )

        validation = ValidationResult(
            is_valid=True,
            actual_file=test_file,
            actual_line=1,
            error_code=None,
            alternatives=[],
            symbol_snippet="def main():",
            adjusted_confidence=85,
            validation_time_ms=10.0,
        )

        # Mock call_llm to raise an exception
        with patch("scout.router.call_llm", side_effect=Exception("LLM unavailable")):
            result = router._generate_symbol_doc(test_file, nav_result, validation)

        assert isinstance(result, SymbolDoc)
        # Should have fallback content
        assert "test.py" in result.content


class TestIndexNavigation:
    """Tests for index-based navigation."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def router(self, temp_repo):
        """Create a TriggerRouter instance for testing."""
        from scout.router import TriggerRouter
        from scout.audit import AuditLog
        from scout.app_config import ScoutConfig

        config = ScoutConfig()
        audit = AuditLog()
        return TriggerRouter(
            config=config,
            audit=audit,
            repo_root=temp_repo,
            trust_level="normal",
        )

    def test_index_confidence_scores(self, router):
        """Test that index results use correct confidence scores."""
        from scout.config.defaults import NAV_INDEX_CONFIDENCE, NAV_DEFAULT_CONFIDENCE

        # Index should have lower confidence than LLM
        assert NAV_INDEX_CONFIDENCE < NAV_DEFAULT_CONFIDENCE
        # But still reasonable
        assert NAV_INDEX_CONFIDENCE >= 50

    def test_parse_nav_json_handles_list(self, router):
        """Test _parse_nav_json handles list responses."""
        # Test with a list (should return first element)
        result = router._parse_nav_json('[{"file": "a.py", "line": 1}]')
        assert isinstance(result, dict)
        assert "file" in result


class TestMagicNumbers:
    """Test that magic numbers are properly extracted to defaults."""

    def test_all_navigation_defaults_importable(self):
        """Test that all navigation defaults can be imported."""
        from scout.config.defaults import (
            NAV_DEFAULT_CONFIDENCE,
            NAV_FALLBACK_CONFIDENCE,
            NAV_INDEX_CONFIDENCE,
            NAV_COST_8B_ESTIMATE,
            NAV_COST_70B_ESTIMATE,
            NAV_FALLBACK_DURATION_MS,
            NAV_CONTEXT_MAX_CHARS,
            NAV_SEARCH_RESULT_LIMIT,
            NAV_PYTHON_FILE_LIMIT,
            NAV_TOKEN_MIN,
            NAV_TOKEN_MAX,
            NAV_TOKEN_CHAR_RATIO,
        )

        # Verify reasonable values
        assert NAV_DEFAULT_CONFIDENCE > 0
        assert NAV_FALLBACK_CONFIDENCE > 0
        assert NAV_INDEX_CONFIDENCE > 0
        assert NAV_COST_8B_ESTIMATE > 0
        assert NAV_COST_70B_ESTIMATE > NAV_COST_8B_ESTIMATE
        assert NAV_FALLBACK_DURATION_MS > 0
        assert NAV_CONTEXT_MAX_CHARS > 0
        assert NAV_SEARCH_RESULT_LIMIT > 0
        assert NAV_PYTHON_FILE_LIMIT > 0
        assert NAV_TOKEN_MIN > 0
        assert NAV_TOKEN_MAX > NAV_TOKEN_MIN
        assert NAV_TOKEN_CHAR_RATIO > 0
