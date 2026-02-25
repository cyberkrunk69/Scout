"""Tests for analysis hotspots module."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path


class TestHotspots:
    """Test hotspot detection functions."""

    def test_get_file_churn_returns_dict(self):
        """Test that get_file_churn returns a dictionary."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="file1.py\nfile2.py\nfile1.py\n")
            from scout.analysis.hotspots import get_file_churn
            result = get_file_churn(days=30, repo_path=Path("/tmp"))
            assert isinstance(result, dict)

    def test_compute_hotspot_score_basic(self):
        """Test hotspot score calculation with mock data."""
        from scout.analysis.hotspots import compute_hotspot_score
        
        # Mock data
        file_churn = {"file1.py": 10, "file2.py": 5}
        error_rates = {"file1.py": 0.1, "file2.py": 0.05}
        impact_scores = {"file1.py": 8.0, "file2.py": 3.0}
        
        scores = compute_hotspot_score(file_churn, error_rates, impact_scores)
        
        assert "file1.py" in scores
        assert "file2.py" in scores
        # file1.py should have higher score due to more churn, errors, and impact
        assert scores["file1.py"] > scores["file2.py"]

    def test_compute_hotspot_score_with_weights(self):
        """Test hotspot score calculation with custom weights."""
        from scout.analysis.hotspots import compute_hotspot_score
        
        file_churn = {"file1.py": 10}
        error_rates = {"file1.py": 0.1}
        impact_scores = {"file1.py": 5.0}
        
        # Test with custom weights
        scores = compute_hotspot_score(
            file_churn, 
            error_rates, 
            impact_scores,
            churn_weight=0.5,
            error_weight=0.3,
            impact_weight=0.2
        )
        
        assert "file1.py" in scores

    def test_get_error_rates_returns_dict(self):
        """Test that get_error_rates returns a dictionary."""
        from scout.analysis.hotspots import get_error_rates
        
        # Mock audit log
        mock_audit = MagicMock()
        mock_audit.get_events.return_value = []
        
        result = get_error_rates(mock_audit, days=30)
        assert isinstance(result, dict)

    def test_scout_hotspots_function_exists(self):
        """Test that scout_hotspots function exists and is callable."""
        from scout.analysis.hotspots import scout_hotspots
        assert callable(scout_hotspots)
