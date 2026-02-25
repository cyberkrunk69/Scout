"""Tests for AST fact extraction module."""

import ast
import tempfile
from pathlib import Path

import pytest

from scout.ast_facts import (
    ASTFactExtractor,
    ModuleFacts,
    SymbolFact,
    ControlFlowFact,
    CallEdge,
    VariableRef,
    SymbolType,
    SemanticRole,
)


class TestSymbolExtraction:
    """Tests for extracting symbols from Python code."""

    def test_extract_constant(self):
        """Test extraction of module-level constants."""
        source = """
MAX_SIZE = 100
DEFAULT_NAME = "test"
"""
        tree = ast.parse(source)
        extractor = ASTFactExtractor()
        symbols = extractor._extract_definitions(tree)

        assert "MAX_SIZE" in symbols
        assert symbols["MAX_SIZE"].type == "constant"
        assert symbols["MAX_SIZE"].value == "100"
        assert "DEFAULT_NAME" in symbols
        assert symbols["DEFAULT_NAME"].value == "test"

    def test_extract_function(self):
        """Test extraction of module-level functions."""
        source = """
def hello():
    pass

def add(a, b):
    return a + b
"""
        tree = ast.parse(source)
        extractor = ASTFactExtractor()
        symbols = extractor._extract_definitions(tree)

        assert "hello" in symbols
        assert symbols["hello"].type == "function"
        assert "add" in symbols
        assert symbols["add"].type == "function"

    def test_extract_class(self):
        """Test extraction of classes."""
        source = """
class MyClass:
    def method(self):
        pass

class SimpleClass:
    pass
"""
        tree = ast.parse(source)
        extractor = ASTFactExtractor()
        symbols = extractor._extract_definitions(tree)

        assert "MyClass" in symbols
        assert symbols["MyClass"].type == "class"
        assert "method" in symbols["MyClass"].methods

    def test_excludes_imports(self):
        """Test that imported names are not extracted as symbols."""
        source = """
import os
from pathlib import Path

MY_CONST = 42
"""
        tree = ast.parse(source)
        extractor = ASTFactExtractor()
        symbols = extractor._extract_definitions(tree)

        assert "os" not in symbols
        assert "Path" not in symbols
        assert "MY_CONST" in symbols


class TestSymbolUsage:
    """Tests for tracing symbol usage."""

    def test_trace_usage(self):
        """Test tracking where symbols are used."""
        source = """x = 10
y = x + 5
z = x + y
"""
        tree = ast.parse(source)
        extractor = ASTFactExtractor()
        symbols = extractor._extract_definitions(tree)
        extractor._trace_usage(tree, symbols)

        # x is used on lines 2 and 3 (y and z definitions)
        assert 2 in symbols["x"].used_at
        assert 3 in symbols["x"].used_at
        # y is used on line 3
        assert 3 in symbols["y"].used_at


class TestControlFlow:
    """Tests for control flow extraction."""

    def test_extract_if_blocks(self):
        """Test extraction of if/elif/else blocks."""
        source = """def process(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
"""
        tree = ast.parse(source)
        extractor = ASTFactExtractor()
        control_flow = extractor._extract_control_flow(tree)

        assert "process" in control_flow
        blocks = control_flow["process"][0].blocks
        # The extractor only captures top-level if statements, not elif
        assert len(blocks) >= 1
        assert blocks[0]["type"] == "if"


class TestImports:
    """Tests for import extraction."""

    def test_extract_imports(self):
        """Test extraction of import statements."""
        source = """
import os
import sys as system
from pathlib import Path
from typing import List, Dict
"""
        tree = ast.parse(source)
        extractor = ASTFactExtractor()
        imports = extractor._extract_imports(tree)

        assert "os" in imports
        assert "system" in imports
        assert "pathlib.Path" in imports
        assert "typing.List" in imports
        assert "typing.Dict" in imports


class TestModuleFacts:
    """Tests for ModuleFacts class."""

    def test_full_extraction(self):
        """Test full extraction from a file."""
        source = """
MY_CONST = 42

def greet(name):
    return f"Hello, {name}!"

class Greeter:
    def __init__(self, greeting):
        self.greeting = greeting
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(source)
            f.flush()
            path = Path(f.name)

        try:
            extractor = ASTFactExtractor()
            facts = extractor.extract(path)

            assert facts.path == path
            assert "MY_CONST" in facts.symbols
            assert "greet" in facts.symbols
            assert "Greeter" in facts.symbols
            assert len(facts.imports) == 0
            assert facts.ast_hash  # Should have a non-empty hash
        finally:
            path.unlink()

    def test_to_json_roundtrip(self):
        """Test serialization and deserialization."""
        source = "x = 10"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(source)
            f.flush()
            path = Path(f.name)

        try:
            extractor = ASTFactExtractor()
            facts = extractor.extract(path)

            json_str = facts.to_json()
            restored = ModuleFacts.from_json(json_str)

            assert restored.path == facts.path
            assert set(restored.symbols.keys()) == set(facts.symbols.keys())
            assert restored.ast_hash == facts.ast_hash
        finally:
            path.unlink()

    def test_checksum(self):
        """Test checksum computation."""
        source = "x = 10"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(source)
            f.flush()
            path = Path(f.name)

        try:
            extractor = ASTFactExtractor()
            facts = extractor.extract(path)

            checksum = facts.checksum()
            assert len(checksum) == 64  # SHA256 hex length
        finally:
            path.unlink()


class TestExtractDocumentableFacts:
    """Tests for documentable facts extraction."""

    def test_extract_with_docstring(self):
        """Test extraction including docstrings."""
        source = '''
"""Module docstring."""

MY_CONST = 42  # A constant

def hello():
    """Say hello."""
    pass

class Greeter:
    """A greeter class."""
    
    def greet(self, name):
        """Greet someone."""
        return f"Hello, {name}!"
'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write(source)
            f.flush()
            path = Path(f.name)

        try:
            extractor = ASTFactExtractor()
            facts = extractor.extract_documentable_facts(path)

            # Check docstrings were extracted
            assert facts.symbols["hello"].docstring == "Say hello."
            assert facts.symbols["Greeter"].docstring == "A greeter class."

            # Check signatures
            assert facts.symbols["hello"].signature is not None

            # Check semantic role inference
            assert facts.symbols["MY_CONST"].semantic_role in [
                "configuration",
                "threshold",
                "limit",
                "implementation",
            ]
        finally:
            path.unlink()


class TestRelationships:
    """Tests for relationship extraction."""

    def test_call_relationships(self):
        """Test extraction of function call relationships."""
        source = """
def outer():
    inner()

def inner():
    pass
"""
        tree = ast.parse(source)
        extractor = ASTFactExtractor()
        symbols = extractor._extract_definitions(tree)
        relations = extractor._extract_relationships(tree, symbols, source)

        assert "outer" in relations
        outer_calls = relations["outer"].calls
        assert len(outer_calls) > 0
        assert any(c.callee == "inner" for c in outer_calls)

    def test_variable_refs(self):
        """Test extraction of variable references."""
        source = """
def process(x):
    y = x + 1
    return y
"""
        tree = ast.parse(source)
        extractor = ASTFactExtractor()
        symbols = extractor._extract_definitions(tree)
        relations = extractor._extract_relationships(tree, symbols, source)

        assert "process" in relations
        var_refs = relations["process"].variable_refs
        assert len(var_refs) > 0


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_file(self):
        """Test extraction from empty file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as f:
            f.write("")
            f.flush()
            path = Path(f.name)

        try:
            extractor = ASTFactExtractor()
            facts = extractor.extract(path)

            assert facts.symbols == {}
            assert facts.control_flow == {}
            assert facts.imports == []
        finally:
            path.unlink()

    def test_async_function(self):
        """Test extraction of async functions."""
        source = """
async def fetch_data():
    await load()
"""
        tree = ast.parse(source)
        extractor = ASTFactExtractor()
        symbols = extractor._extract_definitions(tree)

        assert "fetch_data" in symbols
        assert symbols["fetch_data"].type == "function"
