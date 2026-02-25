"""Tests for analysis/hotspots.py module."""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from scout.analysis.hotspots import (
    compute_hotspot_score,
    get_error_rates,
    get_file_churn,
    get_impact_counts,
    scout_hotspots,
    _extract_module_from_event,
    DEFAULT_CHURN_WEIGHT,
    DEFAULT_ERROR_WEIGHT,
    DEFAULT_IMPACT_WEIGHT,
)


class TestComputeHotspotScore:
    """Tests for compute_hotspot_score function."""

    def test_basic_calculation(self):
        """Test basic score calculation with default weights."""
        score = compute_hotspot_score(
            churn=5,
            errors=3,
            impact_count=10,
            max_churn=10,
            max_errors=10,
            max_impact=50,
        )

        # churn=5/10=0.5, errors=3/10=0.3, impact=10/50=0.2
        # weighted: 0.4*0.5 + 0.4*0.3 + 0.2*0.2 = 0.2 + 0.12 + 0.04 = 0.36
        assert score == 0.36

    def test_max_values_normalized(self):
        """Test that max values are normalized to 1.0."""
        score = compute_hotspot_score(
            churn=10,
            errors=10,
            impact_count=50,
            max_churn=10,
            max_errors=10,
            max_impact=50,
        )

        # All normalized to 1.0: 0.4*1 + 0.4*1 + 0.2*1 = 1.0
        assert score == 1.0

    def test_zero_values(self):
        """Test with all zero values."""
        score = compute_hotspot_score(
            churn=0,
            errors=0,
            impact_count=0,
            max_churn=10,
            max_errors=10,
            max_impact=50,
        )

        assert score == 0.0

    def test_exceeding_max_values(self):
        """Test that values exceeding max are capped at 1.0."""
        score = compute_hotspot_score(
            churn=100,
            errors=100,
            impact_count=500,
            max_churn=10,
            max_errors=10,
            max_impact=50,
        )

        # All capped at 1.0: 0.4*1 + 0.4*1 + 0.2*1 = 1.0
        assert score == 1.0

    def test_custom_weights(self):
        """Test with custom weights."""
        score = compute_hotspot_score(
            churn=5,
            errors=5,
            impact_count=25,
            max_churn=10,
            max_errors=10,
            max_impact=50,
            churn_weight=0.5,
            error_weight=0.3,
            impact_weight=0.2,
        )

        # churn=5/10=0.5, errors=5/10=0.5, impact=25/50=0.5
        # weighted: 0.5*0.5 + 0.3*0.5 + 0.2*0.5 = 0.25 + 0.15 + 0.1 = 0.5
        assert score == 0.5

    def test_zero_max_values(self):
        """Test handling of zero max values."""
        score = compute_hotspot_score(
            churn=5,
            errors=3,
            impact_count=10,
            max_churn=0,
            max_errors=0,
            max_impact=0,
        )

        # All should be 0 when max is 0
        assert score == 0.0


class TestGetFileChurn:
    """Tests for get_file_churn function."""

    @patch("subprocess.run")
    def test_basic_git_log(self, mock_run):
        """Test basic git log parsing."""
        mock_result = MagicMock()
        mock_result.stdout = "file1.py\nfile2.py\nfile1.py\n"
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = get_file_churn(days=30, repo_path=Path("/test/repo"))

        assert result == {"file1.py": 2, "file2.py": 1}
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_empty_output(self, mock_run):
        """Test with empty git output."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.returncode = 0
        mock_run.return_value = mock_result

        result = get_file_churn(days=30)

        assert result == {}

    @patch("subprocess.run")
    def test_non_git_directory(self, mock_run):
        """Test handling of non-git directory."""
        mock_run.side_effect = Exception("Not a git repo")

        result = get_file_churn(days=30, repo_path=Path("/not/a/repo"))

        assert result == {}

    @patch("subprocess.run")
    def test_git_not_installed(self, mock_run):
        """Test when git is not installed."""
        mock_run.side_effect = FileNotFoundError("git not found")

        result = get_file_churn(days=30)

        assert result == {}


class TestGetErrorRates:
    """Tests for get_error_rates function."""

    @patch("scout.analysis.hotspots.AuditLog")
    def test_basic_error_extraction(self, mock_audit_class):
        """Test basic error rate extraction from audit log."""
        mock_log = MagicMock()
        mock_log.query.side_effect = [
            [  # validation_fail
                {"files": ["src/module1.py"], "reason": "validation error"},
                {"files": ["src/module2.py"], "reason": "another error"},
            ],
            [  # llm_error
                {"files": ["src/module1.py"], "reason": "API error"},
            ],
            [],  # llm_retry - empty
            [],  # node_failed - empty
        ]
        mock_audit_class.return_value = mock_log

        result = get_error_rates(days=30, audit_path=Path("/fake/audit.jsonl"))

        # 1 from validation + 1 from llm_error = 2
        assert result.get("src/module1.py") == 2
        assert result.get("src/module2.py") == 1

    @patch("scout.analysis.hotspots.AuditLog")
    def test_audit_log_init_failure(self, mock_audit_class):
        """Test handling when audit log cannot be initialized."""
        mock_audit_class.side_effect = Exception("Audit init failed")

        result = get_error_rates(days=30)

        assert result == {}

    @patch("scout.analysis.hotspots.AuditLog")
    def test_query_failure(self, mock_audit_class):
        """Test handling when query fails for an event type."""
        mock_log = MagicMock()
        # First query succeeds, second fails
        mock_log.query.side_effect = [
            [{"files": ["src/module1.py"]}],
            Exception("Query failed"),
            [],
            [],
        ]
        mock_audit_class.return_value = mock_log

        result = get_error_rates(days=30)

        # Should still return results from successful query
        assert result.get("src/module1.py") == 1


class TestExtractModuleFromEvent:
    """Tests for _extract_module_from_event function."""

    def test_extract_from_files_field(self):
        """Test extraction from 'files' field."""
        event = {"files": ["src/auth.py", "src/utils.py"]}

        result = _extract_module_from_event(event)

        assert result == "src/auth.py"

    def test_extract_from_files_empty(self):
        """Test with empty files list."""
        event = {"files": []}

        result = _extract_module_from_event(event)

        assert result is None

    def test_extract_from_stack_trace(self):
        """Test extraction from stack trace in 'reason'."""
        event = {
            "reason": 'File "/Users/test/src/module.py", line 42, in test_func'
        }

        result = _extract_module_from_event(event)

        assert result == "/Users/test/src/module.py"

    def test_extract_from_raw_brief_path(self):
        """Test extraction from raw_brief_path."""
        event = {
            "raw_brief_path": "/home/user/project/src/auth.py:123"
        }

        result = _extract_module_from_event(event)

        assert "src/auth.py" in result

    def test_extract_makes_relative(self):
        """Test that absolute paths are made relative."""
        event = {
            "raw_brief_path": "/Users/user/project/vivarium/src/scout/module.py"
        }

        result = _extract_module_from_event(event)

        # Should extract relative path
        assert "module.py" in result

    def test_no_matching_data(self):
        """Test with event that has no extractable module."""
        event = {"unrelated": "data"}

        result = _extract_module_from_event(event)

        assert result is None


class TestGetImpactCounts:
    """Tests for get_impact_counts function."""

    @patch("scout.analysis.hotspots.impact_analysis")
    def test_basic_impact(self, mock_impact):
        """Test basic impact counting."""
        mock_impact.side_effect = [
            ["func_a", "func_b", "func_c"],  # file1.py
            ["func_x", "func_y"],  # file2.py
        ]

        result = get_impact_counts(
            files=["file1.py", "file2.py"],
            repo_path=Path("/test/repo"),
        )

        assert result["file1.py"] == 3
        assert result["file2.py"] == 2

    @patch("scout.analysis.hotspots.impact_analysis")
    def test_impact_analysis_failure(self, mock_impact):
        """Test handling when impact analysis fails."""
        mock_impact.side_effect = Exception("Analysis failed")

        result = get_impact_counts(
            files=["file1.py"],
            repo_path=Path("/test/repo"),
        )

        assert result["file1.py"] == 0


class TestScoutHotspots:
    """Tests for scout_hotspots main function."""

    @patch("scout.analysis.hotspots.get_impact_counts")
    @patch("scout.analysis.hotspots.get_error_rates")
    @patch("scout.analysis.hotspots.get_file_churn")
    def test_basic_hotspot_detection(
        self, mock_churn, mock_errors, mock_impact
    ):
        """Test basic hotspot detection."""
        # Setup mocks
        mock_churn.return_value = {
            "src/auth.py": 10,
            "src/utils.py": 5,
        }
        mock_errors.return_value = {
            "src/auth.py": 8,
        }
        mock_impact.return_value = {
            "src/auth.py": 20,
            "src/utils.py": 10,
        }

        result = scout_hotspots(
            days=30,
            limit=10,
            include_impact=True,
        )

        assert "hotspots" in result
        assert "metadata" in result
        assert result["metadata"]["days_analyzed"] == 30
        assert result["metadata"]["total_files_evaluated"] == 2

    @patch("scout.analysis.hotspots.get_impact_counts")
    @patch("scout.analysis.hotspots.get_error_rates")
    @patch("scout.analysis.hotspots.get_file_churn")
    def test_hotspots_sorted_by_score(
        self, mock_churn, mock_errors, mock_impact
    ):
        """Test that hotspots are sorted by score descending."""
        mock_churn.return_value = {"src/a.py": 10, "src/b.py": 1}
        mock_errors.return_value = {"src/a.py": 10, "src/b.py": 1}
        mock_impact.return_value = {"src/a.py": 50, "src/b.py": 10}

        result = scout_hotspots(days=30, limit=10)

        # a.py should have higher score than b.py
        hotspots = result["hotspots"]
        assert len(hotspots) == 2
        assert hotspots[0]["score"] >= hotspots[1]["score"]

    @patch("scout.analysis.hotspots.get_impact_counts")
    @patch("scout.analysis.hotspots.get_error_rates")
    @patch("scout.analysis.hotspots.get_file_churn")
    def test_limit_results(
        self, mock_churn, mock_errors, mock_impact
    ):
        """Test result limiting."""
        mock_churn.return_value = {f"src/file{i}.py": i for i in range(20)}
        mock_errors.return_value = {}
        mock_impact.return_value = {}

        result = scout_hotspots(days=30, limit=5)

        assert len(result["hotspots"]) == 5

    @patch("scout.analysis.hotspots.get_impact_counts")
    @patch("scout.analysis.hotspots.get_error_rates")
    @patch("scout.analysis.hotspots.get_file_churn")
    def test_without_impact(
        self, mock_churn, mock_errors, mock_impact
    ):
        """Test without impact analysis."""
        mock_churn.return_value = {"src/a.py": 10}
        mock_errors.return_value = {"src/a.py": 5}
        mock_impact.return_value = {}

        result = scout_hotspots(days=30, include_impact=False)

        mock_impact.assert_not_called()

    @patch("scout.analysis.hotspots.get_impact_counts")
    @patch("scout.analysis.hotspots.get_error_rates")
    @patch("scout.analysis.hotspots.get_file_churn")
    def test_empty_data(
        self, mock_churn, mock_errors, mock_impact
    ):
        """Test with no data."""
        mock_churn.return_value = {}
        mock_errors.return_value = {}
        mock_impact.return_value = {}

        result = scout_hotspots(days=30)

        assert result["hotspots"] == []
        assert result["metadata"]["total_files_evaluated"] == 0
