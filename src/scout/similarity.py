from __future__ import annotations
"""
Similarity detection for plan deduplication.

Provides hashing-based and embedding-based similarity algorithms.
"""

import hashlib


def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of normalized text content."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def compute_ngram_hash(text: str, n: int = 3) -> set[str]:
    """Compute set of n-gram hashes for fuzzy matching."""
    normalized = " ".join(text.lower().split())
    ngrams = [normalized[i : i + n] for i in range(len(normalized) - n + 1)]
    return {hashlib.sha256(ng.encode()).hexdigest()[:8] for ng in ngrams if ng}


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def is_exact_duplicate(text1: str, text2: str) -> bool:
    """Check if two texts are exact duplicates."""
    return compute_content_hash(text1) == compute_content_hash(text2)


def is_near_duplicate(text1: str, text2: str, threshold: float = 0.8) -> bool:
    """Check if two texts are near-duplicates using n-gram Jaccard."""
    hash1 = compute_ngram_hash(text1)
    hash2 = compute_ngram_hash(text2)
    return jaccard_similarity(hash1, hash2) >= threshold


def find_duplicates(
    texts: list[dict], threshold: float = 0.8, id_field: str = "id"
) -> list[list[str]]:
    """
    Find groups of duplicate/similar texts.

    Args:
        texts: List of dicts with 'id' and 'plan'/'text' fields
        threshold: Similarity threshold (0.0-1.0)
        id_field: Field name for unique identifier

    Returns:
        List of duplicate groups (each group is list of IDs)
    """
    ngram_hashes = {}
    for item in texts:
        item_id = item.get(id_field, "")
        text = item.get("plan") or item.get("text") or ""
        ngram_hashes[item_id] = compute_ngram_hash(text)

    duplicates = []
    processed = set()

    for item in texts:
        item_id = item.get(id_field, "")
        if item_id in processed:
            continue

        group = [item_id]
        item_hash = ngram_hashes.get(item_id, set())

        for other in texts:
            other_id = other.get(id_field, "")
            if other_id == item_id or other_id in processed:
                continue

            other_hash = ngram_hashes.get(other_id, set())
            if jaccard_similarity(item_hash, other_hash) >= threshold:
                group.append(other_id)
                processed.add(other_id)

        if len(group) > 1:
            duplicates.append(group)
            processed.add(item_id)

    return duplicates


def extract_key_phrases(text: str) -> set[str]:
    """Extract key phrases for semantic similarity."""
    words = text.lower().split()
    phrases = set()

    for i in range(len(words)):
        for length in [2, 3, 4]:
            if i + length <= len(words):
                phrase = " ".join(words[i : i + length])
                phrases.add(phrase)

    return phrases


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity using key phrase overlap.

    This is a lightweight alternative to embedding-based similarity.
    For more accurate results, consider using sentence-transformers.
    """
    phrases1 = extract_key_phrases(text1)
    phrases2 = extract_key_phrases(text2)

    return jaccard_similarity(phrases1, phrases2)


class DuplicateDetector:
    """Stateful duplicate detector for incremental checking."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self._stored: list[dict] = []
        self._hashes: dict[str, str] = {}
        self._ngram_hashes: dict[str, set[str]] = {}

    def add(self, item: dict) -> bool:
        """
        Add an item and check for duplicates.

        Returns True if item is a duplicate, False if unique.
        """
        item_id = item.get("id", "")
        text = item.get("plan") or item.get("text") or ""

        content_hash = compute_content_hash(text)
        if content_hash in self._hashes:
            return True

        ngram_hash = compute_ngram_hash(text)

        for stored_id, stored_ngram in self._ngram_hashes.items():
            if jaccard_similarity(ngram_hash, stored_ngram) >= self.threshold:
                return True

        self._stored.append(item)
        self._hashes[content_hash] = item_id
        self._ngram_hashes[item_id] = ngram_hash

        return False

    def get_similar(self, text: str, limit: int = 5) -> list[dict]:
        """Get most similar plans from stored items."""
        text_ngrams = compute_ngram_hash(text)

        similarities = []
        for item in self._stored:
            item_ngrams = self._ngram_hashes.get(item.get("id", ""), set())
            sim = jaccard_similarity(text_ngrams, item_ngrams)
            similarities.append((sim, item))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in similarities[:limit]]

    @property
    def count(self) -> int:
        return len(self._stored)
