"""
Scout Search - FTS5 + BM25 Search Implementation

Local, offline search using SQLite FTS5 with BM25 ranking.
Zero LLM calls, free and instant (<100ms).

Usage:
    from scout.search import SearchIndex

    index = SearchIndex("~/.scout/search.db")
    index.build([
        {"id": "1", "title": "Auth Module", "content": "Handle authentication tokens"},
        {"id": "2", "title": "User Model", "content": "User data model and validation"},
    ])
    results = index.search("auth token", limit=10)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from scout.config.defaults import (
    BM25_BASE,
    BM25_RANGE,
    BM25_CLARITY_MAX,
    BM25_EXACT_BOOST,
    BM25_CLASS_BOOST,
    SEARCH_CONFIDENCE_GAP_FACTOR,
    SEARCH_CONFIDENCE_GAP_BASE,
)


# Type alias for configuration
SearchConfig = Dict[str, Any]


class SearchIndex:
    """
    FTS5 + BM25 search index with configurable field weights.

    Supports generic document indexing with configurable field weights
    for BM25 ranking. Designed for offline, local search use cases.
    """

    DEFAULT_CONFIG: SearchConfig = {
        "index_path": "~/.scout/search.db",
        "field_weights": {
            "title": 5.0,
            "content": 3.0,
        },
        "tokenizer": "porter unicode61",
    }

    def __init__(
        self,
        path: Union[str, Path, None] = None,
        config: Optional[SearchConfig] = None,
    ):
        """
        Initialize search index.

        Args:
            path: Path to SQLite database. Defaults to ~/.scout/search.db
            config: Configuration dictionary with keys:
                - index_path: Path to database (overrides path arg)
                - field_weights: Dict of field name -> weight for BM25
                - tokenizer: FTS5 tokenizer options
        """
        # Merge configs (default < arg < config dict)
        self._config = dict(self.DEFAULT_CONFIG)
        
        if path:
            self._config["index_path"] = str(path)
        if config:
            self._config.update(config)

        # Resolve path
        self._db_path = Path(self._config["index_path"]).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._field_weights = self._config.get("field_weights", self.DEFAULT_CONFIG["field_weights"])
        self._tokenizer = self._config.get("tokenizer", self.DEFAULT_CONFIG["tokenizer"])
        
        # Ensure at least one searchable field
        if not self._field_weights:
            self._field_weights = {"content": 1.0}

    @property
    def path(self) -> Path:
        """Return the database path."""
        return self._db_path

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        return sqlite3.connect(str(self._db_path))

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """
        Create FTS5 virtual table for documents.
        
        Uses a simple approach: all searchable fields are FTS5 columns.
        ID is stored in a separate table for retrieval.
        """
        # Get searchable columns from field weights
        columns = list(self._field_weights.keys())
        
        # Ensure we have at least content
        if "content" not in columns:
            columns.append("content")
        
        conn.execute("DROP TABLE IF EXISTS documents")
        
        # Create FTS5 table with searchable columns
        col_list = ", ".join(columns)
        create_sql = f"""
            CREATE VIRTUAL TABLE documents USING fts5(
                {col_list},
                tokenize='{self._tokenizer}'
            )
        """
        conn.execute(create_sql)
        
        # Create metadata table for document IDs
        conn.execute("DROP TABLE IF EXISTS doc_metadata")
        conn.execute("""
            CREATE TABLE doc_metadata (
                doc_id INTEGER PRIMARY KEY,
                external_id TEXT UNIQUE,
                extra_data TEXT
            )
        """)

    def build(self, documents: List[Dict[str, Any]]) -> int:
        """
        Rebuild index from scratch.

        Args:
            documents: List of dicts with required 'id' and 'content' keys.
                      Optional 'title' and other fields supported based on config.

        Returns:
            Number of documents indexed
        """
        if not documents:
            return 0

        conn = self._get_connection()
        try:
            self._create_schema(conn)
            
            count = 0
            for doc in documents:
                doc_id = doc.get("id")
                if not doc_id:
                    continue
                
                # Build insert values based on configured fields
                fields = list(self._field_weights.keys())
                values = []
                
                for field in fields:
                    value = doc.get(field, "")
                    values.append(str(value) if value is not None else "")
                
                # Insert into FTS5
                placeholders = ", ".join(["?"] * len(fields))
                cursor = conn.execute(
                    f"INSERT INTO documents ({', '.join(fields)}) VALUES ({placeholders})",
                    values
                )
                
                # Get the FTS5 rowid
                fts_rowid = cursor.lastrowid
                
                # Store external ID in metadata (store rowid - 1 for 0-based indexing)
                extra = {k: v for k, v in doc.items() if k not in fields and k != "id"}
                import json
                extra_json = json.dumps(extra) if extra else None
                conn.execute(
                    "INSERT INTO doc_metadata (doc_id, external_id, extra_data) VALUES (?, ?, ?)",
                    (fts_rowid, doc_id, extra_json)
                )
                count += 1
            
            conn.commit()
            return count
        finally:
            conn.close()

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Insert or update documents.

        Args:
            documents: List of dicts with 'id' and 'content' keys

        Returns:
            Number of documents added/updated
        """
        if not documents:
            return 0

        conn = self._get_connection()
        try:
            # Ensure table exists
            try:
                conn.execute("SELECT 1 FROM documents LIMIT 1")
            except sqlite3.OperationalError:
                self._create_schema(conn)

            # Get current max doc_id for new inserts
            try:
                max_id = conn.execute("SELECT MAX(doc_id) FROM doc_metadata").fetchone()[0]
                if max_id is None:
                    max_id = -1
            except sqlite3.OperationalError:
                max_id = -1

            count = 0
            fields = list(self._field_weights.keys())
            
            for doc in documents:
                doc_id = doc.get("id")
                if not doc_id:
                    continue
                
                # Check if exists
                existing = conn.execute(
                    "SELECT doc_id FROM doc_metadata WHERE external_id = ?",
                    (doc_id,)
                ).fetchone()
                
                if existing:
                    # Update - delete old, insert new
                    conn.execute("DELETE FROM documents WHERE rowid = ?", (existing[0],))
                    conn.execute("DELETE FROM doc_metadata WHERE external_id = ?", (doc_id,))
                    # Don't increment max_id for updates
                
                values = [str(doc.get(f, "")) if doc.get(f) is not None else "" for f in fields]
                placeholders = ", ".join(["?"] * len(fields))
                
                cursor = conn.execute(
                    f"INSERT INTO documents ({', '.join(fields)}) VALUES ({placeholders})",
                    values
                )
                
                fts_rowid = cursor.lastrowid
                max_id = max(max_id + 1, fts_rowid)
                
                extra = {k: v for k, v in doc.items() if k not in fields and k != "id"}
                import json
                extra_json = json.dumps(extra) if extra else None
                conn.execute(
                    "INSERT INTO doc_metadata (doc_id, external_id, extra_data) VALUES (?, ?, ?)",
                    (fts_rowid, doc_id, extra_json)
                )
                count += 1
            
            conn.commit()
            return count
        finally:
            conn.close()

    def update(self, document_ids: List[str]) -> int:
        """
        Delete documents by ID.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        if not document_ids:
            return 0
        
        conn = self._get_connection()
        try:
            count = 0
            for doc_id in document_ids:
                row = conn.execute(
                    "SELECT doc_id FROM doc_metadata WHERE external_id = ?",
                    (doc_id,)
                ).fetchone()
                if row:
                    conn.execute("DELETE FROM documents WHERE rowid = ?", (row[0],))
                    conn.execute("DELETE FROM doc_metadata WHERE external_id = ?", (doc_id,))
                    count += 1
            conn.commit()
            return count
        except sqlite3.OperationalError:
            return 0
        finally:
            conn.close()

    def _normalize_scores(self, results: List[dict]) -> List[dict]:
        """
        Normalize BM25 scores to 0.0-1.0 range and calculate gap between top results.
        """
        if not results:
            return results

        scores = [-r.get("raw_score", 0) for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if len(scores) >= 2:
            sorted_scores = sorted(scores, reverse=True)
            gap_top2 = (sorted_scores[0] - sorted_scores[1]) / (
                max_score - min_score + 1e-10
            )
            gap_top2 = min(gap_top2, 1.0)
        else:
            gap_top2 = 1.0

        if max_score == min_score:
            for r in results:
                r["normalized"] = 0.5
                r["gap_top2"] = gap_top2
        else:
            for r, score in zip(results, scores):
                r["normalized"] = (score - min_score) / (max_score - min_score)
                r["gap_top2"] = gap_top2

        return results

    def _compute_confidence(
        self,
        normalized: float,
        gap_top2: float,
        is_exact_match: bool = False,
        kind: str = "",
    ) -> int:
        """
        Compute confidence (0-100) using normalized score and gap-based clarity.
        """
        base = BM25_BASE + (normalized * BM25_RANGE)
        clarity = max(0, (gap_top2 ** 0.5) * SEARCH_CONFIDENCE_GAP_FACTOR - SEARCH_CONFIDENCE_GAP_BASE)
        clarity = min(clarity, BM25_CLARITY_MAX)
        exact_boost = 1.03 if is_exact_match else 1.0
        kind_boost = 1.01 if kind == "class" else 1.0
        total = (base + clarity) * exact_boost * kind_boost
        return int(min(total, 100))

    def search(
        self,
        query: str,
        limit: int = 20,
        min_confidence: float = 0.0,
        exact_match_field: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the index using BM25 ranking.

        Args:
            query: Search query string
            limit: Maximum number of results
            min_confidence: Minimum confidence score (0-100) to return
            exact_match_field: If provided, boost exact matches in this field

        Returns:
            List of result dicts with keys:
            - id: Document ID
            - score: BM25 score (lower = better)
            - confidence: Computed confidence (0-100)
            - snippet: FTS5 snippet (if available)
            - Fields from the original document
        """
        if not query or not query.strip():
            return []

        conn = self._get_connection()
        try:
            # Check if table exists
            try:
                conn.execute("SELECT 1 FROM documents LIMIT 1")
            except sqlite3.OperationalError:
                return []

            # Build FTS5 query
            tokens = query.replace('"', " ").split()
            fts_query = " AND ".join(f'"{t}"' for t in tokens if t)
            
            if not fts_query:
                return []

            fields = list(self._field_weights.keys())
            if not fields:
                fields = ["content"]
            
            # Use BM25 with field weights
            weights = [self._field_weights.get(f, 1.0) for f in fields]
            weight_str = ", ".join(str(w) for w in weights)
            
            # Build snippet from first field
            first_field = fields[0]
            
            try:
                # Join with metadata to get external_id
                # Select: id, fields..., raw_score, snippet, extra_data
                field_cols = ", ".join([f"d.{f}" for f in fields])
                cursor = conn.execute(
                    f"""
                    SELECT 
                        dm.external_id as id,
                        {field_cols},
                        bm25(documents, {weight_str}) as raw_score,
                        snippet(documents, 0, '<mark>', '</mark>', '...', 32) as snippet,
                        dm.extra_data
                    FROM documents d
                    JOIN doc_metadata dm ON d.rowid = dm.doc_id
                    WHERE documents MATCH ?
                    ORDER BY raw_score
                    LIMIT ?
                    """,
                    (fts_query, limit * 2),
                )
                rows = cursor.fetchall()
            except sqlite3.OperationalError:
                # Fallback without weights
                try:
                    cursor = conn.execute(
                        f"""
                        SELECT 
                            dm.external_id as id,
                            {field_cols},
                            bm25(documents) as raw_score,
                            snippet(documents, 0, '<mark>', '</mark>', '...', 32) as snippet,
                            dm.extra_data
                        FROM documents d
                        JOIN doc_metadata dm ON d.rowid = dm.doc_id
                        WHERE documents MATCH ?
                        ORDER BY raw_score
                        LIMIT ?
                        """,
                        (fts_query, limit * 2),
                    )
                    rows = cursor.fetchall()
                except sqlite3.OperationalError:
                    return []

            if not rows:
                return []

            # Convert to result dicts - use numeric indices since no row_factory
            results: List[Dict[str, Any]] = []
            for row in rows:
                # row structure: [id, field1, field2, ..., raw_score, snippet, extra_data]
                result: Dict[str, Any] = {"id": row[0], "raw_score": row[len(fields) + 1]}
                
                # Add searchable fields (columns start at index 1)
                for i, field in enumerate(fields):
                    result[field] = row[i + 1]
                
                # Add snippet (after fields + raw_score)
                if len(row) > len(fields) + 2:
                    result["snippet"] = row[len(fields) + 2]
                else:
                    result["snippet"] = ""
                
                # Add extra data from metadata (last column)
                if len(row) > len(fields) + 3 and row[len(fields) + 3]:
                    import json
                    try:
                        result.update(json.loads(row[len(fields) + 3]))
                    except (json.JSONDecodeError, TypeError):
                        pass
                    except json.JSONDecodeError:
                        pass
                
                # Check for exact match
                if exact_match_field and exact_match_field in result:
                    is_exact = query.lower() == str(result[exact_match_field]).lower()
                    result["is_exact_match"] = is_exact
                else:
                    result["is_exact_match"] = False
                
                results.append(result)

            # Normalize and compute confidence
            results = self._normalize_scores(results)
            
            for r in results:
                r["confidence"] = self._compute_confidence(
                    normalized=r.get("normalized", 0.5),
                    gap_top2=r.get("gap_top2", 0),
                    is_exact_match=r.get("is_exact_match", False),
                    kind=r.get("kind", ""),
                )
                # Remove internal fields
                r.pop("raw_score", None)
                r.pop("normalized", None)
                r.pop("gap_top2", None)

            # Filter by confidence and limit
            results = [r for r in results if r["confidence"] >= min_confidence]
            results = results[:limit]

            return results

        except Exception:
            return []
        finally:
            conn.close()

    def count(self) -> int:
        """Return number of documents in index."""
        conn = self._get_connection()
        try:
            try:
                c = conn.execute("SELECT COUNT(*) FROM doc_metadata")
                return c.fetchone()[0]
            except sqlite3.OperationalError:
                return 0
        finally:
            conn.close()

    def clear(self) -> None:
        """Clear all documents from the index."""
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM doc_metadata")
            conn.commit()
        finally:
            conn.close()

    def delete(self, document_ids: List[str]) -> int:
        """
        Delete documents by ID.

        Args:
            document_ids: List of document IDs to delete

        Returns:
            Number of documents deleted
        """
        return self.update(document_ids)


# ============================================================================
# Helper functions
# ============================================================================

def _looks_like_identifier(token: str) -> bool:
    """True if token looks like CamelCase or snake_case identifier."""
    if not token or len(token) < 2:
        return False
    if token[0].isupper() and any(c.islower() for c in token):
        return True
    if "_" in token and token.replace("_", "").replace("-", "").isalnum():
        return True
    return False


def _normalize_scores(results: List[dict]) -> List[dict]:
    """Normalize BM25 scores to 0.0-1.0 range."""
    if not results:
        return results

    scores = [-r.get("raw_score", 0) for r in results]
    min_score = min(scores)
    max_score = max(scores)

    if len(scores) >= 2:
        sorted_scores = sorted(scores, reverse=True)
        gap_top2 = (sorted_scores[0] - sorted_scores[1]) / (
            max_score - min_score + 1e-10
        )
        gap_top2 = min(gap_top2, 1.0)
    else:
        gap_top2 = 1.0

    if max_score == min_score:
        for r in results:
            r["normalized"] = 0.5
            r["gap_top2"] = gap_top2
    else:
        for r, score in zip(results, scores):
            r["normalized"] = (score - min_score) / (max_score - min_score)
            r["gap_top2"] = gap_top2

    return results


def _compute_confidence(
    normalized: float,
    gap_top2: float,
    is_exact_match: bool = False,
    kind: str = "",
) -> int:
    """Compute confidence using normalized score and gap-based clarity."""
    base = 50 + (normalized * 30)
    clarity = max(0, (gap_top2 ** 0.5) * 40 - 8)
    clarity = min(clarity, 30)
    exact_boost = 1.03 if is_exact_match else 1.0
    kind_boost = 1.01 if kind == "class" else 1.0
    total = (base + clarity) * exact_boost * kind_boost
    return int(min(total, 100))


# ============================================================================
# Legacy compatibility functions
# ============================================================================

def create_index(
    db_path: Union[str, Path],
    documents: List[Dict[str, Any]],
    field_weights: Optional[Dict[str, float]] = None,
) -> SearchIndex:
    """Create a new search index with documents."""
    config = {}
    if field_weights:
        config["field_weights"] = field_weights
    
    index = SearchIndex(db_path, config)
    index.build(documents)
    return index


def query_index(
    db_path: Union[str, Path],
    query: str,
    limit: int = 20,
    min_confidence: float = 0.0,
) -> List[Dict[str, Any]]:
    """Query an existing search index."""
    index = SearchIndex(db_path)
    return index.search(query, limit, min_confidence)
