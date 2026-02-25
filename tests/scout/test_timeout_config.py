"""Tests for timeout_config.py module."""

import os
import pytest
from unittest.mock import patch

from scout.timeout_config import (
    TimeoutConfig,
    TimeoutError,
    get_timeout_config,
    set_timeout_config,
)


class TestTimeoutConfig:
    """Tests for TimeoutConfig class."""

    def test_default_values(self):
        """Test default timeout values."""
        config = TimeoutConfig()

        assert config.connect_timeout == 10.0
        assert config.read_timeout == 60.0
        assert config.deepseek_connect == 10.0
        assert config.deepseek_read == 60.0
        assert config.minimax_connect == 15.0
        assert config.minimax_read == 90.0

    def test_custom_values(self):
        """Test custom timeout values."""
        config = TimeoutConfig(
            connect_timeout=5.0,
            read_timeout=30.0,
            deepseek_connect=5.0,
            deepseek_read=30.0,
            minimax_connect=10.0,
            minimax_read=60.0,
        )

        assert config.connect_timeout == 5.0
        assert config.read_timeout == 30.0

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_defaults(self):
        """Test from_env with no environment variables."""
        config = TimeoutConfig.from_env()

        assert config.connect_timeout == 10.0
        assert config.read_timeout == 60.0

    @patch.dict(os.environ, {
        "SCOUT_CONNECT_TIMEOUT": "5.0",
        "SCOUT_READ_TIMEOUT": "30.0",
    }, clear=True)
    def test_from_env_with_values(self):
        """Test from_env with environment variables."""
        config = TimeoutConfig.from_env()

        assert config.connect_timeout == 5.0
        assert config.read_timeout == 30.0

    @patch.dict(os.environ, {
        "SCOUT_DEEPSEEK_CONNECT_TIMEOUT": "15.0",
        "SCOUT_DEEPSEEK_READ_TIMEOUT": "120.0",
    }, clear=True)
    def test_from_env_deepseek_overrides(self):
        """Test DeepSeek-specific overrides from env."""
        config = TimeoutConfig.from_env()

        assert config.deepseek_connect == 15.0
        assert config.deepseek_read == 120.0

    @patch.dict(os.environ, {
        "SCOUT_MINIMAX_CONNECT_TIMEOUT": "20.0",
        "SCOUT_MINIMAX_READ_TIMEOUT": "180.0",
    }, clear=True)
    def test_from_env_minimax_overrides(self):
        """Test MiniMax-specific overrides from env."""
        config = TimeoutConfig.from_env()

        assert config.minimax_connect == 20.0
        assert config.minimax_read == 180.0

    def test_for_provider_deepseek(self):
        """Test for_provider with deepseek."""
        config = TimeoutConfig()

        connect, read = config.for_provider("deepseek")

        assert connect == 10.0
        assert read == 60.0

    def test_for_provider_minimax(self):
        """Test for_provider with minimax."""
        config = TimeoutConfig()

        connect, read = config.for_provider("minimax")

        assert connect == 15.0
        assert read == 90.0

    def test_for_provider_unknown(self):
        """Test for_provider with unknown provider uses defaults."""
        config = TimeoutConfig()

        connect, read = config.for_provider("unknown")

        assert connect == 10.0  # Default
        assert read == 60.0     # Default


class TestTimeoutError:
    """Tests for TimeoutError exception."""

    def test_exception(self):
        """Test TimeoutError can be raised."""
        with pytest.raises(TimeoutError):
            raise TimeoutError("Request timed out")

    def test_exception_with_timeout_value(self):
        """Test TimeoutError with timeout value."""
        err = TimeoutError("Request timeout after 60s")

        assert "60s" in str(err)


class TestGetTimeoutConfig:
    """Tests for get_timeout_config and set_timeout_config."""

    def test_get_timeout_config_singleton(self):
        """Test that get_timeout_config returns singleton."""
        # Reset global
        import scout.timeout_config as tc_module
        tc_module._config = None

        config1 = get_timeout_config()
        config2 = get_timeout_config()

        assert config1 is config2

    def test_set_timeout_config(self):
        """Test setting custom timeout config."""
        # Reset global
        import scout.timeout_config as tc_module
        tc_module._config = None

        custom_config = TimeoutConfig(connect_timeout=1.0, read_timeout=5.0)
        set_timeout_config(custom_config)

        config = get_timeout_config()

        assert config.connect_timeout == 1.0
        assert config.read_timeout == 5.0

    def test_set_and_get(self):
        """Test set then get."""
        # Reset global
        import scout.timeout_config as tc_module
        tc_module._config = None

        custom = TimeoutConfig(connect_timeout=99.0)
        set_timeout_config(custom)

        result = get_timeout_config()

        assert result.connect_timeout == 99.0
