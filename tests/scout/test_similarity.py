"""Tests for similarity module."""

import pytest
from scout.similarity import (
    compute_content_hash,
    compute_ngram_hash,
    jaccard_similarity,
    is_exact_duplicate,
    is_near_duplicate,
    find_duplicates,
    extract_key_phrases,
    DuplicateDetector,
)


class TestSimilarity:
    """Test similarity scoring functions."""

    def test_compute_content_hash(self):
        """Test content hash computation."""
        text = "hello world"
        hash1 = compute_content_hash(text)
        hash2 = compute_content_hash(text)
        assert hash1 == hash2
        # Different text should produce different hash
        assert compute_content_hash("different") != hash1

    def test_compute_ngram_hash(self):
        """Test n-gram hash computation."""
        text = "hello world hello"
        ngrams = compute_ngram_hash(text, n=2)
        assert isinstance(ngrams, set)
        assert len(ngrams) > 0

    def test_jaccard_similarity_identical(self):
        """Test jaccard similarity with identical sets."""
        set1 = {1, 2, 3, 4}
        set2 = {1, 2, 3, 4}
        assert jaccard_similarity(set1, set2) == 1.0

    def test_jaccard_similarity_disjoint(self):
        """Test jaccard similarity with disjoint sets."""
        set1 = {1, 2, 3, 4}
        set2 = {5, 6, 7, 8}
        assert jaccard_similarity(set1, set2) == 0.0

    def test_jaccard_similarity_partial(self):
        """Test jaccard similarity with partial overlap."""
        set1 = {1, 2, 3, 4}
        set2 = {3, 4, 5, 6}
        # Intersection = {3, 4}, Union = {1, 2, 3, 4, 5, 6}
        assert jaccard_similarity(set1, set2) == 2/6

    def test_is_exact_duplicate_true(self):
        """Test exact duplicate detection."""
        assert is_exact_duplicate("hello world", "hello world") is True

    def test_is_exact_duplicate_false(self):
        """Test exact duplicate detection with different text."""
        assert is_exact_duplicate("hello world", "hello there") is False

    def test_is_near_duplicate_high_similarity(self):
        """Test near duplicate detection with high similarity."""
        text1 = "This is a test document with some content"
        text2 = "This is a test document with some different content"
        assert is_near_duplicate(text1, text2, threshold=0.5) is True

    def test_is_near_duplicate_low_similarity(self):
        """Test near duplicate detection with low similarity."""
        text1 = "Completely different text here"
        text2 = "Another set of words that is nothing alike"
        assert is_near_duplicate(text1, text2, threshold=0.8) is False

    def test_find_duplicates(self):
        """Test finding duplicates in a list of texts."""
        texts = [
            {"id": "1", "plan": "hello world"},
            {"id": "2", "plan": "hello world"},  # duplicate
            {"id": "3", "plan": "different text"},
            {"id": "4", "plan": "hello world"},  # duplicate
            {"id": "5", "plan": "another one"},
        ]
        duplicates = find_duplicates(texts)
        assert len(duplicates) > 0

    def test_extract_key_phrases(self):
        """Test key phrase extraction."""
        text = "Python is a great programming language for data science"
        phrases = extract_key_phrases(text)
        assert isinstance(phrases, set)

    def test_duplicate_detector(self):
        """Test DuplicateDetector class."""
        detector = DuplicateDetector(threshold=0.8)
        
        item1 = {"id": "1", "plan": "This is a test document"}
        item2 = {"id": "2", "plan": "This is a test document"}
        item3 = {"id": "3", "plan": "Completely different"}
        
        # Add first item - should be unique
        is_dup1 = detector.add(item1)
        assert is_dup1 is False
        
        # Add duplicate item - should be detected as duplicate
        is_dup2 = detector.add(item2)
        assert is_dup2 is True
        
        # Add different item - should be unique
        is_dup3 = detector.add(item3)
        assert is_dup3 is False
