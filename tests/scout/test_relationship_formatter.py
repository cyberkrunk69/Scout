"""Unit tests for relationship_formatter module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scout.doc_sync.relationship_formatter import (
    RelationshipData,
    format_callers,
    format_callees,
    format_examples,
    format_inheritance,
    generate_mermaid_inheritance,
    get_relationship_data,
    inject_relationship_sections,
)


class TestFormatCallers:
    """Tests for format_callers function."""

    def test_empty_callers_returns_empty_string(self):
        """Empty caller list should return empty string."""
        result = format_callers([])
        assert result == ""

    def test_single_caller_formatted(self):
        """Single caller should be formatted correctly."""
        from scout.graph import CallerInfo

        callers = [
            CallerInfo(
                symbol="module.py::process_data",
                file=Path("module.py"),
                line=42,
                call_type="direct",
                ambiguous=False,
            )
        ]
        result = format_callers(callers)
        assert "## Callers" in result
        assert "module.py::process_data" in result
        assert "(line 42)" in result

    def test_truncation_with_max_items(self):
        """Test that results are truncated when exceeding max_items."""
        from scout.graph import CallerInfo

        callers = [
            CallerInfo(
                symbol=f"module.py::func_{i}",
                file=Path("module.py"),
                line=i,
                call_type="direct",
                ambiguous=False,
            )
            for i in range(10)
        ]
        result = format_callers(callers, max_items=3)
        assert "func_0" in result
        assert "func_2" in result
        assert "and 7 more" in result
        assert "func_5" not in result

    def test_ambiguous_symbol_marked(self):
        """Ambiguous symbols should be marked."""
        from scout.graph import CallerInfo

        callers = [
            CallerInfo(
                symbol="ambiguous_func",
                file=Path("module.py"),
                line=10,
                call_type="direct",
                ambiguous=True,
            )
        ]
        result = format_callers(callers)
        assert "(AMBIGUOUS)" in result


class TestFormatCallees:
    """Tests for format_callees function."""

    def test_empty_callees_returns_empty_string(self):
        """Empty callee list should return empty string."""
        result = format_callees([])
        assert result == ""

    def test_callees_formatted(self):
        """Callees should be formatted correctly."""
        from scout.graph import CalleeInfo

        callees = [
            CalleeInfo(
                symbol="helper.py::validate",
                file=Path("helper.py"),
                line=15,
                call_type="direct",
                ambiguous=False,
            )
        ]
        result = format_callees(callees)
        assert "## Callees" in result
        assert "helper.py::validate" in result


class TestFormatExamples:
    """Tests for format_examples function."""

    def test_empty_usages_returns_empty_string(self):
        """Empty usage list should return empty string."""
        result = format_examples([])
        assert result == ""

    def test_usage_formatted_as_code_block(self):
        """Usages should be formatted as code blocks."""
        from scout.graph import UsageInfo

        usages = [
            UsageInfo(
                symbol="process_data",
                file=Path("main.py"),
                line=25,
                usage_type="calls",
            )
        ]
        result = format_examples(usages)
        assert "## Usage Examples" in result
        assert "```python" in result
        assert "# main.py:25" in result


class TestFormatInheritance:
    """Tests for format_inheritance function."""

    def test_empty_returns_empty_string(self):
        """Empty inheritance data should return empty string."""
        result = format_inheritance([], [])
        assert result == ""

    def test_base_classes_formatted(self):
        """Base classes should be formatted correctly."""
        result = format_inheritance(
            base_classes=["base.py::BaseClass"],
            subclasses=[],
        )
        assert "## Inheritance" in result
        assert "**Base classes:**" in result
        assert "BaseClass" in result

    def test_subclasses_formatted(self):
        """Subclasses should be formatted correctly."""
        result = format_inheritance(
            base_classes=[],
            subclasses=["derived.py::DerivedClass"],
        )
        assert "## Inheritance" in result
        assert "**Subclasses:**" in result
        assert "DerivedClass" in result


class TestGenerateMermaidInheritance:
    """Tests for generate_mermaid_inheritance function."""

    def test_empty_returns_empty_string(self):
        """Empty inheritance should return empty string."""
        result = generate_mermaid_inheritance("MyClass", [], [])
        assert result == ""

    def test_single_inheritance(self):
        """Single inheritance should generate valid Mermaid."""
        result = generate_mermaid_inheritance(
            "MyClass",
            base_classes=["base.py::BaseClass"],
            subclasses=[],
        )
        assert "```mermaid" in result
        assert "classDiagram" in result
        assert "BaseClass <|-- MyClass" in result

    def test_subclass_generation(self):
        """Subclass should generate valid Mermaid."""
        result = generate_mermaid_inheritance(
            "MyClass",
            base_classes=[],
            subclasses=["derived.py::DerivedClass"],
        )
        assert "MyClass <|-- DerivedClass" in result


class TestGetRelationshipData:
    """Tests for get_relationship_data function."""

    @patch("scout.doc_sync.relationship_formatter._ensure_graph_loaded")
    def test_returns_empty_when_graph_unavailable(self, mock_load):
        """Should return empty data when graph is not available."""
        mock_load.return_value = None
        result = get_relationship_data(
            symbol_name="test_func",
            symbol_type="function",
            file_path=Path("test.py"),
        )
        assert result.callers == []
        assert result.callees == []
        assert result.usages == []

    @patch("scout.doc_sync.relationship_formatter._get_callers")
    @patch("scout.doc_sync.relationship_formatter._get_callees")
    @patch("scout.doc_sync.relationship_formatter._find_usages")
    @patch("scout.doc_sync.relationship_formatter._ensure_graph_loaded")
    def test_queries_graph_when_available(
        self, mock_load, mock_find_usages, mock_get_callees, mock_get_callers
    ):
        """Should query graph when available."""
        from scout.graph import CallerInfo, CalleeInfo, UsageInfo

        mock_load.return_value = {
            "nodes": {"test.py::test_func": {}},
            "edges": [],
            "symbolIndex": {},
        }
        mock_get_callers.return_value = [
            CallerInfo(
                symbol="caller.py::caller_func",
                file=Path("caller.py"),
                line=10,
            )
        ]
        mock_get_callees.return_value = [
            CalleeInfo(
                symbol="helper.py::helper_func",
                file=Path("helper.py"),
                line=20,
            )
        ]
        mock_find_usages.return_value = [
            UsageInfo(
                symbol="test_func",
                file=Path("main.py"),
                line=30,
                usage_type="calls",
            )
        ]

        result = get_relationship_data(
            symbol_name="test_func",
            symbol_type="function",
            file_path=Path("test.py"),
        )

        assert len(result.callers) == 1
        assert len(result.callees) == 1
        assert len(result.usages) == 1


class TestInjectRelationshipSections:
    """Tests for inject_relationship_sections function."""

    def test_returns_unchanged_when_disabled_via_config(self):
        """Should return content unchanged when disabled via config."""
        mock_config = MagicMock()
        mock_config.get_doc_relationships_config.return_value = {
            "enabled": False
        }

        content = "# Test\n\nSome content"
        result = inject_relationship_sections(
            content=content,
            symbol_name="test_func",
            symbol_type="function",
            file_path=Path("test.py"),
            repo_root=Path.cwd(),
            config=mock_config,
        )
        assert result == content

    @patch("scout.doc_sync.relationship_formatter.get_relationship_data")
    def test_injects_sections_when_enabled(self, mock_get_data):
        """Should inject sections when enabled."""
        from scout.graph import CallerInfo

        mock_config = MagicMock()
        mock_config.get_doc_relationships_config.return_value = {
            "enabled": True,
            "include_callers": True,
            "include_callees": True,
            "include_examples": True,
            "include_inheritance": False,
            "include_mermaid": False,
            "max_related_items": 5,
        }
        mock_get_data.return_value = RelationshipData(
            callers=[
                CallerInfo(
                    symbol="caller.py::caller_func",
                    file=Path("caller.py"),
                    line=10,
                )
            ],
            callees=[],
            usages=[],
            base_classes=[],
            subclasses=[],
        )

        content = "# Test Function\n\nSome documentation."
        result = inject_relationship_sections(
            content=content,
            symbol_name="test_func",
            symbol_type="function",
            file_path=Path("test.py"),
            repo_root=Path.cwd(),
            config=mock_config,
        )

        assert "## Callers" in result
        assert "caller.py::caller_func" in result

    def test_no_injection_for_class_inheritance_when_not_class(self):
        """Should not inject inheritance for non-class symbols."""
        mock_config = MagicMock()
        mock_config.get_doc_relationships_config.return_value = {
            "enabled": True,
            "include_callers": False,
            "include_callees": False,
            "include_examples": False,
            "include_inheritance": True,
            "include_mermaid": True,
            "max_related_items": 5,
        }

        content = "# Test Function\n\nSome documentation."
        result = inject_relationship_sections(
            content=content,
            symbol_name="test_func",
            symbol_type="function",  # Not a class
            file_path=Path("test.py"),
            repo_root=Path.cwd(),
            config=mock_config,
        )

        # Should return unchanged since there are no callers/callees/examples
        # and inheritance is only for classes
        assert result == content
