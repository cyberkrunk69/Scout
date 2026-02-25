"""Tests for env_validator.py module."""

import os
import pytest
from unittest.mock import patch, MagicMock

from scout.env_validator import (
    EnvValidationResult,
    validate_environment,
    get_router_status,
)


class TestEnvValidationResult:
    """Tests for EnvValidationResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = EnvValidationResult(
            deepseek_key_set=True,
            minimax_key_set=False,
            scout_llm_mode="auto",
            is_valid=True,
            missing_required=[],
            warnings=["warning1"],
        )

        d = result.to_dict()

        assert d["deepseek_configured"] is True
        assert d["minimax_configured"] is False
        assert d["scout_llm_mode"] == "auto"
        assert d["is_valid"] is True
        assert d["missing_required"] == []
        assert d["warnings"] == ["warning1"]


class TestValidateEnvironment:
    """Tests for validate_environment function."""

    def test_free_mode_with_deepseek_key(self):
        """Test free mode with DeepSeek API key set."""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key-123",
            "SCOUT_LLM_MODE": "free"
        }, clear=True):
            result = validate_environment()

            assert result.deepseek_key_set is True
            assert result.scout_llm_mode == "free"
            assert result.is_valid is True

    def test_free_mode_without_deepseek_key(self):
        """Test free mode without DeepSeek API key."""
        with patch.dict(os.environ, {
            "SCOUT_LLM_MODE": "free"
        }, clear=True):
            result = validate_environment()

            assert result.deepseek_key_set is False
            assert result.is_valid is False
            assert "DEEPSEEK_API_KEY" in result.missing_required[0]

    def test_paid_mode_with_minimax_key(self):
        """Test paid mode with MiniMax API key set."""
        with patch.dict(os.environ, {
            "MINIMAX_API_KEY": "test-key-456",
            "SCOUT_LLM_MODE": "paid"
        }, clear=True):
            result = validate_environment()

            assert result.minimax_key_set is True
            assert result.scout_llm_mode == "paid"
            assert result.is_valid is True

    def test_paid_mode_without_minimax_key(self):
        """Test paid mode without MiniMax API key."""
        with patch.dict(os.environ, {
            "SCOUT_LLM_MODE": "paid"
        }, clear=True):
            result = validate_environment()

            assert result.minimax_key_set is False
            assert result.is_valid is False
            assert "MINIMAX_API_KEY" in result.missing_required[0]

    def test_auto_mode_with_both_keys(self):
        """Test auto mode with both API keys."""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key-123",
            "MINIMAX_API_KEY": "test-key-456",
            "SCOUT_LLM_MODE": "auto"
        }, clear=True):
            result = validate_environment()

            assert result.deepseek_key_set is True
            assert result.minimax_key_set is True
            assert result.scout_llm_mode == "auto"
            assert result.is_valid is True
            assert result.missing_required == []

    def test_auto_mode_without_any_keys(self):
        """Test auto mode without any API keys."""
        with patch.dict(os.environ, {
            "SCOUT_LLM_MODE": "auto"
        }, clear=True):
            result = validate_environment()

            assert result.deepseek_key_set is False
            assert result.minimax_key_set is False
            assert result.is_valid is False
            assert "At least one of" in result.missing_required[0]
            assert len(result.warnings) > 0

    def test_default_mode_is_auto(self):
        """Test that default mode is auto when not set."""
        with patch.dict(os.environ, {}, clear=True):
            result = validate_environment()

            assert result.scout_llm_mode == "auto"

    def test_unknown_mode_warning(self):
        """Test warning for unknown mode."""
        with patch.dict(os.environ, {
            "SCOUT_LLM_MODE": "unknown_mode"
        }, clear=True):
            result = validate_environment()

            assert any("Unknown SCOUT_LLM_MODE" in w for w in result.warnings)

    def test_unused_deepseek_key_warning(self):
        """Test warning when DeepSeek key is unused in paid mode."""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key-123",
            "SCOUT_LLM_MODE": "paid",
            "MINIMAX_API_KEY": "test-key-456"
        }, clear=True):
            result = validate_environment()

            assert any("DEEPSEEK_API_KEY is set but" in w for w in result.warnings)

    def test_unused_minimax_key_warning(self):
        """Test warning when MiniMax key is unused in free mode."""
        with patch.dict(os.environ, {
            "MINIMAX_API_KEY": "test-key-456",
            "SCOUT_LLM_MODE": "free",
            "DEEPSEEK_API_KEY": "test-key-123"
        }, clear=True):
            result = validate_environment()

            assert any("MINIMAX_API_KEY is set but" in w for w in result.warnings)


class TestGetRouterStatus:
    """Tests for get_router_status function."""

    @patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "test-key",
        "MINIMAX_API_KEY": "",
        "SCOUT_LLM_MODE": "free"
    }, clear=True)
    def test_basic_status(self):
        """Test basic router status."""
        with patch("scout.env_validator.validate_environment") as mock_validate:
            mock_validate.return_value = MagicMock(
                deepseek_key_set=True,
                minimax_key_set=False,
                scout_llm_mode="free",
                is_valid=True,
                missing_required=[],
                warnings=[],
            )

            result = get_router_status()

            assert result["mode"] == "free"
            assert result["deepseek_configured"] is True
            assert result["minimax_configured"] is False
            assert result["is_valid"] is True

    @patch.dict(os.environ, {}, clear=True)
    def test_status_with_missing_keys(self):
        """Test status when keys are missing."""
        with patch("scout.env_validator.validate_environment") as mock_validate:
            mock_validate.return_value = MagicMock(
                deepseek_key_set=False,
                minimax_key_set=False,
                scout_llm_mode="auto",
                is_valid=False,
                missing_required=["API key missing"],
                warnings=["warning"],
            )

            result = get_router_status()

            assert result["missing_required"] == ["API key missing"]
            assert result["warnings"] == ["warning"]

    @patch.dict(os.environ, {
        "DEEPSEEK_API_KEY": "test-key",
    }, clear=True)
    def test_status_default_rpm(self):
        """Test default RPM values when router not available."""
        with patch("scout.env_validator.validate_environment") as mock_validate:
            mock_validate.return_value = MagicMock(
                deepseek_key_set=True,
                minimax_key_set=False,
                scout_llm_mode="auto",
                is_valid=True,
                missing_required=[],
                warnings=[],
            )

            # Without router imported, should return defaults
            result = get_router_status()

            assert result["deepseek_rpm_used"] == 0
            assert result["deepseek_rpm_limit"] == 50
            assert result["deepseek_reset_in"] == 0
