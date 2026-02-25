"""
TrustStore - Async SQLite CRUD for trust data.

Philosophy: DRY + Auditability + Async/Parallel + Right-Size Tooling
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import aiosqlite

from .models import TrustRecord
from .constants import TRUST_DB_FILENAME

logger = logging.getLogger(__name__)


class TrustStore:
    """
    Async SQLite CRUD for trust data.

    Async: All I/O operations are async with aiosqlite
    Resilient: Proper connection management
    """

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.db_path = repo_root / ".scout" / TRUST_DB_FILENAME
        self._lock = asyncio.Lock()
        self._connection: Optional[aiosqlite.Connection] = None

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create a shared connection."""
        if self._connection is None:
            self._connection = await aiosqlite.connect(str(self.db_path))
            self._connection.row_factory = aiosqlite.Row
        return self._connection

    async def initialize(self) -> None:
        """Create schema if not exists."""
        # Ensure .scout directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        async with self._lock:
            conn = await self._get_connection()
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trust (
                    source_path TEXT PRIMARY KEY,
                    doc_path TEXT NOT NULL,
                    trust_level TEXT NOT NULL,
                    embedded_checksum TEXT,
                    current_checksum TEXT,
                    stale_symbols TEXT,  -- JSON
                    fresh_symbols TEXT,  -- JSON
                    penalty REAL DEFAULT 0.0,
                    last_validated TEXT,
                    query_count INTEGER DEFAULT 0,
                    last_queried TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_trust_level ON trust(trust_level)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_queried ON trust(last_queried)"
            )
            await conn.commit()
            logger.info(f"Initialized trust store at {self.db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def upsert(self, record: TrustRecord) -> None:
        """Insert or update trust record (does NOT increment query_count)."""
        async with self._lock:
            conn = await self._get_connection()
            await conn.execute(
                """
                INSERT INTO trust (
                    source_path, doc_path, trust_level, embedded_checksum,
                    current_checksum, stale_symbols, fresh_symbols, penalty,
                    last_validated, query_count, last_queried,
                    success_count, failure_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                    doc_path = excluded.doc_path,
                    trust_level = excluded.trust_level,
                    embedded_checksum = excluded.embedded_checksum,
                    current_checksum = excluded.current_checksum,
                    stale_symbols = excluded.stale_symbols,
                    fresh_symbols = excluded.fresh_symbols,
                    penalty = excluded.penalty,
                    last_validated = excluded.last_validated,
                    query_count = trust.query_count,  -- preserve
                    last_queried = excluded.last_queried,
                    success_count = trust.success_count, -- preserve
                    failure_count = trust.failure_count   -- preserve
            """,
                (
                    record.source_path,
                    record.doc_path,
                    record.trust_level,
                    record.embedded_checksum,
                    record.current_checksum,
                    json.dumps(record.stale_symbols),
                    json.dumps(record.fresh_symbols),
                    record.penalty,
                    record.last_validated,
                    0,  # query_count not updated here
                    record.last_queried,
                    record.success_count,
                    record.failure_count,
                ),
            )
            await conn.commit()

    async def increment_query_count(self, source_path: str) -> None:
        """Increment query count for a source file."""
        async with self._lock:
            conn = await self._get_connection()
            await conn.execute(
                """
                UPDATE trust SET query_count = query_count + 1, last_queried = ?
                WHERE source_path = ?
            """,
                (datetime.utcnow().isoformat(), source_path),
            )
            await conn.commit()

    async def update_learning_counts(
        self, source_path: str, success: bool
    ) -> None:
        """Update success/failure counts for Bayesian learning."""
        col = "success_count" if success else "failure_count"
        async with self._lock:
            conn = await self._get_connection()
            await conn.execute(
                f"""
                UPDATE trust SET {col} = {col} + 1
                WHERE source_path = ?
            """,
                (source_path,),
            )
            await conn.commit()

    async def get(self, source_path: str) -> Optional[TrustRecord]:
        """Get trust record for a source file."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            "SELECT * FROM trust WHERE source_path = ?", (source_path,)
        )
        row = await cursor.fetchone()
        return self._row_to_record(row) if row else None

    async def get_by_level(self, trust_level: str) -> List[TrustRecord]:
        """Get all records with given trust level."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            "SELECT * FROM trust WHERE trust_level = ?", (trust_level,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(r) for r in rows]

    async def get_stale(self, limit: int = 100) -> List[TrustRecord]:
        """Get stale records, ordered by query count descending."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            """
            SELECT * FROM trust
            WHERE trust_level IN ('stale', 'no_checksum')
            ORDER BY query_count DESC
            LIMIT ?
        """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(r) for r in rows]

    async def get_frequently_queried(self, limit: int = 10) -> List[TrustRecord]:
        """Get most frequently queried records."""
        conn = await self._get_connection()
        cursor = await conn.execute(
            """
            SELECT * FROM trust
            ORDER BY query_count DESC
            LIMIT ?
        """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_record(r) for r in rows]

    async def update_penalty(self, source_path: str, penalty: float) -> None:
        """Update penalty for a record."""
        async with self._lock:
            conn = await self._get_connection()
            await conn.execute(
                "UPDATE trust SET penalty = ? WHERE source_path = ?",
                (penalty, source_path),
            )
            await conn.commit()

    def _row_to_record(self, row: aiosqlite.Row) -> TrustRecord:
        """Convert DB row to TrustRecord."""
        return TrustRecord(
            source_path=row["source_path"],
            doc_path=row["doc_path"],
            trust_level=row["trust_level"],
            embedded_checksum=row["embedded_checksum"],
            current_checksum=row["current_checksum"],
            stale_symbols=json.loads(row["stale_symbols"] or "[]"),
            fresh_symbols=json.loads(row["fresh_symbols"] or "[]"),
            penalty=row["penalty"],
            last_validated=row["last_validated"],
            query_count=row["query_count"],
            last_queried=row["last_queried"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
        )
