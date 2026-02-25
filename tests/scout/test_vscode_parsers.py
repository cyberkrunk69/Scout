"""Tests for VS Code extension parsers (Copilot, Cline, Continue).

Uses mock JSON files for testing parsing logic without requiring
actual VS Code installation.
"""

import json
import pytest
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path

from scout.vscode_parsers import (
    AgentParser,
    CopilotParser,
    ClineParser,
    ContinueParser,
    ChatMessage,
    ParsedSession,
    ParseResult,
    PARSER_REGISTRY,
    get_parser,
    parse_agent,
)
from scout.vscode_storage import (
    normalise_path,
    parse_json_file,
    hash_workspace_path,
)


class TestNormalisePath:
    """Tests for normalise_path utility."""

    def test_expands_user(self):
        """Test that ~ is expanded to home directory."""
        path = normalise_path("~/test")
        assert "~" not in str(path)
        assert path.is_absolute()

    def test_resolves_relative(self):
        """Test that relative paths are resolved."""
        path = normalise_path("test/../test/file.py")
        assert ".." not in str(path)

    def test_handles_path_object(self):
        """Test that Path objects are handled correctly."""
        path_obj = Path("test/file.py")
        result = normalise_path(path_obj)
        assert isinstance(result, Path)


class TestParseJsonFile:
    """Tests for parse_json_file utility."""

    def test_parses_valid_json(self, tmp_path):
        """Test parsing valid JSON file."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}')

        result = parse_json_file(test_file)
        assert result == {"key": "value"}

    def test_returns_none_for_missing_file(self, tmp_path):
        """Test that missing file returns None."""
        result = parse_json_file(tmp_path / "nonexistent.json")
        assert result is None

    def test_returns_none_for_invalid_json(self, tmp_path):
        """Test that invalid JSON returns None."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text("{ invalid json }")

        result = parse_json_file(test_file)
        assert result is None


class TestHashWorkspacePath:
    """Tests for hash_workspace_path utility."""

    def test_consistent_hash(self):
        """Test that same path produces same hash."""
        hash1 = hash_workspace_path("/test/path")
        hash2 = hash_workspace_path("/test/path")
        assert hash1 == hash2

    def test_different_paths_different_hashes(self):
        """Test that different paths produce different hashes."""
        hash1 = hash_workspace_path("/test/path1")
        hash2 = hash_workspace_path("/test/path2")
        assert hash1 != hash2


class TestChatMessage:
    """Tests for ChatMessage dataclass."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        msg = ChatMessage(role="user", content="Hello")
        result = msg.to_dict()
        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_to_dict_with_timestamp(self):
        """Test to_dict with timestamp."""
        ts = datetime(2024, 1, 15, 10, 30, 0)
        msg = ChatMessage(role="user", content="Hello", timestamp=ts)
        result = msg.to_dict()
        assert result["timestamp"] == "2024-01-15T10:30:00"

    def test_to_dict_with_model(self):
        """Test to_dict with model."""
        msg = ChatMessage(role="assistant", content="Hi", model="gpt-4")
        result = msg.to_dict()
        assert result["model"] == "gpt-4"


class TestParsedSession:
    """Tests for ParsedSession dataclass."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        session = ParsedSession(
            session_id="test-123",
            agent="copilot",
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi"),
            ],
        )
        result = session.to_dict()
        assert result["session_id"] == "test-123"
        assert result["agent"] == "copilot"
        assert len(result["messages"]) == 2
        assert result["success"] is True

    def test_to_dict_with_error(self):
        """Test to_dict when session failed."""
        session = ParsedSession(
            session_id="test-123",
            agent="copilot",
            messages=[],
            success=False,
            error="Parse error",
        )
        result = session.to_dict()
        assert result["success"] is False
        assert result["error"] == "Parse error"


class TestParseResult:
    """Tests for ParseResult dataclass."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = ParseResult(
            agent="copilot",
            sessions=[],
            total_sessions=10,
            successful=8,
            failed=2,
            duration_ms=100.0,
        )
        assert result.success_rate == 80.0

    def test_success_rate_zero_sessions(self):
        """Test success rate with zero sessions."""
        result = ParseResult(
            agent="copilot",
            sessions=[],
            total_sessions=0,
            successful=0,
            failed=0,
            duration_ms=0.0,
        )
        assert result.success_rate == 0.0


class TestCopilotParser:
    """Tests for CopilotParser class."""

    def test_agent_id(self):
        """Test agent ID is copilot."""
        parser = CopilotParser()
        assert parser.agent_id == "copilot"

    def test_list_sessions_no_storage(self):
        """Test list_sessions returns empty when no storage."""
        parser = CopilotParser()
        sessions = parser.list_sessions()
        assert sessions == []

    def test_parse_session_no_storage(self):
        """Test parse_session returns error when no storage."""
        parser = CopilotParser()
        session = parser.parse_session("test-123")
        assert session.success is False
        assert "not found" in session.error.lower()


class TestClineParser:
    """Tests for ClineParser class."""

    def test_agent_id(self):
        """Test agent ID is cline."""
        parser = ClineParser()
        assert parser.agent_id == "cline"

    def test_list_sessions_no_storage(self):
        """Test list_sessions returns empty when no storage."""
        parser = ClineParser()
        sessions = parser.list_sessions()
        assert sessions == []

    def test_parse_session_no_storage(self):
        """Test parse_session returns error when no storage."""
        parser = ClineParser()
        session = parser.parse_session("test-123")
        assert session.success is False


class TestContinueParser:
    """Tests for ContinueParser class."""

    def test_agent_id(self):
        """Test agent ID is continue."""
        parser = ContinueParser()
        assert parser.agent_id == "continue"

    def test_list_sessions_no_storage(self):
        """Test list_sessions returns empty when no storage."""
        parser = ContinueParser()
        sessions = parser.list_sessions()
        assert sessions == []


class TestParserRegistry:
    """Tests for parser registry."""

    def test_registry_contains_all_agents(self):
        """Test registry contains all supported agents."""
        assert "copilot" in PARSER_REGISTRY
        assert "cline" in PARSER_REGISTRY
        assert "continue" in PARSER_REGISTRY

    def test_get_parser_returns_correct_type(self):
        """Test get_parser returns correct parser class."""
        parser = get_parser("copilot")
        assert isinstance(parser, CopilotParser)

        parser = get_parser("cline")
        assert isinstance(parser, ClineParser)

        parser = get_parser("continue")
        assert isinstance(parser, ContinueParser)

    def test_get_parser_case_insensitive(self):
        """Test get_parser is case insensitive."""
        parser = get_parser("COPILOT")
        assert isinstance(parser, CopilotParser)

        parser = get_parser("Copilot")
        assert isinstance(parser, CopilotParser)

    def test_get_parser_unsupported_agent(self):
        """Test get_parser returns None for unsupported agent."""
        parser = get_parser("unsupported")
        assert parser is None


class TestParseAgent:
    """Tests for parse_agent function."""

    def test_parse_agent_unsupported_raises(self):
        """Test parse_agent raises ValueError for unsupported agent."""
        with pytest.raises(ValueError) as exc_info:
            parse_agent("unsupported")
        assert "Unsupported agent" in str(exc_info.value)


# Mock data fixtures for integration tests

@pytest.fixture
def mock_copilot_session(tmp_path):
    """Create mock Copilot session file."""
    session_data = {
        "sessionId": "test-session-123",
        "createdAt": 1705312200000,  # 2024-01-15T10:30:00Z
        "updatedAt": 1705312800000,
        "requests": [
            {
                "message": {
                    "text": "Help me write a function"
                },
                "response": {
                    "message": {
                        "content": "Here's a function for you"
                    },
                    "model": "gpt-4"
                }
            },
            {
                "message": {
                    "text": "Make it handle errors"
                },
                "response": {
                    "message": {
                        "content": "Added error handling"
                    },
                    "model": "gpt-4"
                }
            }
        ]
    }
    return session_data


@pytest.fixture
def mock_cline_api_history(tmp_path):
    """Create mock Cline API conversation history."""
    history_data = [
        {
            "role": "user",
            "content": "Create a new file"
        },
        {
            "role": "assistant",
            "content": "I'll create that file for you",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "tool_use",
                    "name": "Write",
                    "input": {"file_path": "test.py", "content": "..."}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "File created successfully"
        }
    ]
    return history_data


@pytest.fixture
def mock_continue_sessions(tmp_path):
    """Create mock Continue sessions.json file."""
    sessions_data = [
        {
            "id": "session-1",
            "title": "Code review",
            "messages": [
                {"role": "user", "content": "Review this PR"},
                {"role": "assistant", "content": "Looks good!"}
            ],
            "created_at": "2024-01-15T10:00:00",
            "updated_at": "2024-01-15T10:30:00"
        },
        {
            "id": "session-2",
            "title": "Bug fix",
            "messages": [
                {"role": "user", "content": "Fix the bug"},
                {"role": "assistant", "content": "Fixed!"}
            ],
            "created_at": "2024-01-14T15:00:00",
            "updated_at": "2024-01-14T15:30:00"
        }
    ]
    return sessions_data


@pytest.fixture
def mock_continue_db(tmp_path):
    """Create mock Continue history.db SQLite file."""
    db_path = tmp_path / "history.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY,
            title TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY,
            session_id INTEGER,
            role TEXT,
            content TEXT,
            model TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    cursor.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (1, "Test Session", "2024-01-15T10:00:00", "2024-01-15T10:30:00")
    )

    cursor.execute(
        "INSERT INTO messages (session_id, role, content, model) VALUES (?, ?, ?, ?)",
        (1, "user", "Hello", "gpt-4")
    )

    cursor.execute(
        "INSERT INTO messages (session_id, role, content, model) VALUES (?, ?, ?, ?)",
        (1, "assistant", "Hi there!", "gpt-4")
    )

    conn.commit()
    conn.close()

    return db_path


class TestCopilotParserWithMock:
    """Integration tests for CopilotParser with mock data."""

    def test_parse_mock_session(self, mock_copilot_session, monkeypatch):
        """Test parsing a mock Copilot session file."""
        # Create temporary mock storage
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "github.copilot" / "chatSessions"
            storage_path.mkdir(parents=True)

            # Write mock session file
            session_file = storage_path / "session-test-123.json"
            with open(session_file, "w") as f:
                json.dump(mock_copilot_session, f)

            # Create parser with mock storage
            parser = CopilotParser()
            monkeypatch.setattr(parser, "get_storage_path", lambda: storage_path.parent)

            session = parser.parse_session("test-123")

            assert session.success is True
            assert len(session.messages) == 4  # 2 user + 2 assistant
            assert session.messages[0].role == "user"
            assert "write a function" in session.messages[0].content.lower()
            assert session.messages[1].role == "assistant"


class TestClineParserWithMock:
    """Integration tests for ClineParser with mock data."""

    def test_parse_mock_api_history(self, mock_cline_api_history):
        """Test parsing mock Cline API history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "saoudrizwan.claude-dev" / "tasks" / "test-task"
            storage_path.mkdir(parents=True)

            # Write mock API history
            history_file = storage_path / "api_conversation_history.json"
            with open(history_file, "w") as f:
                json.dump(mock_cline_api_history, f)

            parser = ClineParser()
            original_get_storage = parser.get_storage_path
            parser.get_storage_path = lambda: storage_path.parent.parent

            session = parser.parse_session("test-task")

            assert session.success is True
            assert len(session.messages) == 3
            assert session.messages[0].role == "user"
            assert session.messages[1].role == "assistant"
            assert session.messages[1].tool_calls is not None


class TestContinueParserWithMock:
    """Integration tests for ContinueParser with mock data."""

    def test_parse_mock_sessions_json(self, mock_continue_sessions):
        """Test parsing mock Continue sessions.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / ".continue"
            storage_path.mkdir(parents=True)

            # Write mock sessions
            sessions_file = storage_path / "sessions.json"
            with open(sessions_file, "w") as f:
                json.dump(mock_continue_sessions, f)

            parser = ContinueParser()
            parser.get_storage_path = lambda: storage_path

            sessions = parser.list_sessions()
            assert len(sessions) == 2

            session = parser.parse_session("0")
            assert session.success is True
            assert len(session.messages) == 2
