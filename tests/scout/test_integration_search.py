"""Integration tests for search index using temporary SQLite database.

These tests exercise the full search flow with real SQLite FTS5 backend.
"""

import pytest
import tempfile
import os
from pathlib import Path

from scout.search import SearchIndex, create_index, query_index


@pytest.fixture
def temp_db_path():
    """Fixture providing a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_search.db")
        yield db_path


@pytest.fixture
def sample_documents():
    """Fixture providing sample documents for testing."""
    return [
        {"id": "1", "title": "Authentication Module", "content": "Handle user authentication and JWT tokens"},
        {"id": "2", "title": "User Model", "content": "User data model and validation with pydantic"},
        {"id": "3", "title": "Database Connection", "content": "PostgreSQL connection pooling and migrations"},
        {"id": "4", "title": "API Router", "content": "FastAPI router configuration and middleware"},
        {"id": "5", "title": "Cache Service", "content": "Redis caching layer for performance optimization"},
    ]


class TestSearchIndexIntegration:
    """Integration tests for SearchIndex with temporary SQLite."""

    def test_index_creation_and_search(self, temp_db_path, sample_documents):
        """Test creating index and performing search."""
        # Create index with documents
        index = SearchIndex(temp_db_path)
        count = index.build(sample_documents)

        assert count == 5
        assert index.count() == 5

        # Perform search
        results = index.search("authentication")

        assert len(results) > 0
        assert results[0]["id"] == "1"  # Should find "Authentication Module" first

    def test_search_with_limit(self, temp_db_path, sample_documents):
        """Test search respects limit parameter."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents)

        # Add more documents to make limit meaningful
        more_docs = [
            {"id": "6", "title": "Logging", "content": "Structured logging with JSON"},
            {"id": "7", "title": "Monitoring", "content": "Metrics and alerting"},
            {"id": "8", "title": "Testing", "content": "Unit and integration tests"},
        ]
        index.add_documents(more_docs)

        # Search with limit
        results = index.search("logging", limit=2)

        assert len(results) <= 2

    def test_search_with_confidence_filter(self, temp_db_path, sample_documents):
        """Test search respects min_confidence filter."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents)

        # Search with high confidence threshold
        results = index.search("authentication", min_confidence=50)

        # Results should have confidence scores
        for r in results:
            assert "confidence" in r
            assert r["confidence"] >= 50

    def test_add_documents_to_existing_index(self, temp_db_path, sample_documents):
        """Test adding documents to existing index."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents[:3])  # Build with first 3

        # Add more documents
        added = index.add_documents(sample_documents[3:])

        assert added == 2
        assert index.count() == 5

    def test_update_documents(self, temp_db_path, sample_documents):
        """Test updating existing documents."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents)

        # Update document
        updated = index.add_documents([
            {"id": "1", "title": "Authentication Module Updated", "content": "New content for auth"}
        ])

        assert updated == 1
        assert index.count() == 5  # Still 5 docs

        # Search should return updated content
        results = index.search("authentication")
        assert "Updated" in results[0].get("title", "")

    def test_delete_documents(self, temp_db_path, sample_documents):
        """Test deleting documents from index."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents)

        # Delete document
        deleted = index.delete(["1"])

        assert deleted == 1
        assert index.count() == 4

        # Search should not find deleted document
        results = index.search("authentication")
        assert not any(r["id"] == "1" for r in results)

    def test_clear_index(self, temp_db_path, sample_documents):
        """Test clearing all documents from index."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents)

        assert index.count() == 5

        index.clear()

        assert index.count() == 0

    def test_empty_index_search(self, temp_db_path):
        """Test searching in empty index."""
        index = SearchIndex(temp_db_path)
        # Don't build any documents

        results = index.search("test")

        assert results == []

    def test_search_with_field_weights(self, temp_db_path, sample_documents):
        """Test search with custom field weights."""
        config = {
            "field_weights": {
                "title": 10.0,  # Higher weight for title
                "content": 1.0,
            }
        }
        index = SearchIndex(temp_db_path, config=config)
        index.build(sample_documents)

        # Search for "database" which is in title of doc 3
        results = index.search("database")

        assert len(results) > 0

    def test_multiple_search_queries(self, temp_db_path, sample_documents):
        """Test multiple searches on same index."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents)

        # Multiple searches
        results1 = index.search("authentication")
        results2 = index.search("database")
        results3 = index.search("cache")

        assert len(results1) > 0
        assert len(results2) > 0
        assert len(results3) > 0

    def test_search_returns_snippets(self, temp_db_path, sample_documents):
        """Test that search returns snippets."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents)

        results = index.search("authentication")

        assert len(results) > 0
        assert "snippet" in results[0]


class TestSearchIndexEdgeCases:
    """Edge case tests for SearchIndex."""

    def test_search_with_special_characters(self, temp_db_path):
        """Test search handles special characters."""
        index = SearchIndex(temp_db_path)
        index.build([
            {"id": "1", "title": "Test with (parens)", "content": "Content with special chars: @#$%"},
        ])

        # Search should not crash
        results = index.search("test")
        assert isinstance(results, list)

    def test_search_with_empty_query(self, temp_db_path, sample_documents):
        """Test search with empty query."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents)

        results = index.search("")
        assert results == []

    def test_search_with_whitespace_only(self, temp_db_path, sample_documents):
        """Test search with whitespace-only query."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents)

        results = index.search("   ")
        assert results == []

    def test_very_long_query(self, temp_db_path, sample_documents):
        """Test search with very long query."""
        index = SearchIndex(temp_db_path)
        index.build(sample_documents)

        long_query = " ".join(["word"] * 1000)
        results = index.search(long_query)

        assert isinstance(results, list)

    def test_many_documents(self, temp_db_path):
        """Test index with many documents."""
        # Create 1000 documents
        docs = [
            {"id": str(i), "title": f"Document {i}", "content": f"Content for document {i}"}
            for i in range(1000)
        ]

        index = SearchIndex(temp_db_path)
        count = index.build(docs)

        assert count == 1000
        assert index.count() == 1000

        # Search should still work
        results = index.search("document 500")
        assert len(results) > 0


class TestSearchHelperFunctions:
    """Tests for search helper functions."""

    def test_create_index_function(self, temp_db_path, sample_documents):
        """Test create_index helper function."""
        index = create_index(temp_db_path, sample_documents)

        assert isinstance(index, SearchIndex)
        assert index.count() == 5

    def test_query_index_function(self, temp_db_path, sample_documents):
        """Test query_index helper function."""
        # First create index
        create_index(temp_db_path, sample_documents)

        # Then query
        results = query_index(temp_db_path, "authentication")

        assert len(results) > 0
        assert results[0]["id"] == "1"

    def test_query_index_with_limit(self, temp_db_path, sample_documents):
        """Test query_index with limit parameter."""
        create_index(temp_db_path, sample_documents)

        results = query_index(temp_db_path, "user", limit=2)

        assert len(results) <= 2


class TestSearchIndexPersistence:
    """Tests for search index persistence across instances."""

    def test_index_persists_across_instances(self, temp_db_path, sample_documents):
        """Test that index data persists when creating new SearchIndex instance."""
        # Create and build index
        index1 = SearchIndex(temp_db_path)
        index1.build(sample_documents)

        # Create new instance with same path
        index2 = SearchIndex(temp_db_path)

        # Should find documents
        results = index2.search("authentication")
        assert len(results) > 0
        assert index2.count() == 5
