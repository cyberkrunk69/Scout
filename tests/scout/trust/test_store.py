"""Tests for TrustStore."""

import pytest
import tempfile
import asyncio
from pathlib import Path

from scout.trust.store import TrustStore
from scout.trust.models import TrustRecord
from scout.trust.constants import TRUST_DB_FILENAME


@pytest.fixture
def temp_repo():
    """Create a temporary repository root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def trust_store(temp_repo):
    """Create an initialized TrustStore."""
    store = TrustStore(temp_repo)
    await store.initialize()
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_store_initialization(temp_repo):
    """Test that store creates database and schema on init."""
    store = TrustStore(temp_repo)
    await store.initialize()

    # Check db file exists
    db_path = temp_repo / ".scout" / TRUST_DB_FILENAME
    assert db_path.exists()

    await store.close()


@pytest.mark.asyncio
async def test_store_upsert(trust_store, temp_repo):
    """Test inserting and updating trust records."""
    record = TrustRecord(
        source_path=str(temp_repo / "test.py"),
        doc_path=str(temp_repo / "docs" / "test.py.deep.md"),
        trust_level="trusted",
        embedded_checksum="abc123",
        current_checksum="abc123",
        stale_symbols=[],
        fresh_symbols=["foo", "bar"],
        penalty=0.0,
    )

    await trust_store.upsert(record)

    # Retrieve and verify
    retrieved = await trust_store.get(str(temp_repo / "test.py"))
    assert retrieved is not None
    assert retrieved.source_path == str(temp_repo / "test.py")
    assert retrieved.trust_level == "trusted"
    assert retrieved.embedded_checksum == "abc123"


@pytest.mark.asyncio
async def test_store_increment_query_count(trust_store, temp_repo):
    """Test incrementing query count."""
    source_path = str(temp_repo / "test.py")

    # Insert a record first
    record = TrustRecord(
        source_path=source_path,
        doc_path=str(temp_repo / "docs" / "test.py.deep.md"),
        trust_level="trusted",
        embedded_checksum="abc123",
        current_checksum="abc123",
    )
    await trust_store.upsert(record)

    # Increment query count
    await trust_store.increment_query_count(source_path)

    # Verify count
    retrieved = await trust_store.get(source_path)
    assert retrieved.query_count == 1


@pytest.mark.asyncio
async def test_store_get_by_level(trust_store, temp_repo):
    """Test retrieving records by trust level."""
    # Insert records with different trust levels
    for i in range(3):
        record = TrustRecord(
            source_path=str(temp_repo / f"trusted_{i}.py"),
            doc_path=str(temp_repo / "docs" / f"trusted_{i}.py.deep.md"),
            trust_level="trusted",
            embedded_checksum="abc",
            current_checksum="abc",
        )
        await trust_store.upsert(record)

    record = TrustRecord(
        source_path=str(temp_repo / "stale.py"),
        doc_path=str(temp_repo / "docs" / "stale.py.deep.md"),
        trust_level="stale",
        embedded_checksum="def",
        current_checksum="ghi",
    )
    await trust_store.upsert(record)

    # Get trusted records
    trusted = await trust_store.get_by_level("trusted")
    assert len(trusted) == 3

    # Get stale records
    stale = await trust_store.get_by_level("stale")
    assert len(stale) == 1


@pytest.mark.asyncio
async def test_store_get_stale(trust_store, temp_repo):
    """Test retrieving stale records."""
    # Insert stale record
    record = TrustRecord(
        source_path=str(temp_repo / "stale.py"),
        doc_path=str(temp_repo / "docs" / "stale.py.deep.md"),
        trust_level="stale",
        embedded_checksum="old",
        current_checksum="new",
        query_count=10,
    )
    await trust_store.upsert(record)

    # Get stale
    stale = await trust_store.get_stale(limit=10)
    assert len(stale) == 1
    assert stale[0].trust_level == "stale"


@pytest.mark.asyncio
async def test_store_update_learning_counts(trust_store, temp_repo):
    """Test updating success/failure counts for learning."""
    source_path = str(temp_repo / "test.py")

    record = TrustRecord(
        source_path=source_path,
        doc_path=str(temp_repo / "docs" / "test.py.deep.md"),
        trust_level="trusted",
        embedded_checksum="abc",
        current_checksum="abc",
    )
    await trust_store.upsert(record)

    # Update with success
    await trust_store.update_learning_counts(source_path, success=True)

    retrieved = await trust_store.get(source_path)
    assert retrieved.success_count == 1
    assert retrieved.failure_count == 0

    # Update with failure
    await trust_store.update_learning_counts(source_path, success=False)

    retrieved = await trust_store.get(source_path)
    assert retrieved.success_count == 1
    assert retrieved.failure_count == 1


@pytest.mark.asyncio
async def test_store_update_penalty(trust_store, temp_repo):
    """Test updating penalty for a record."""
    source_path = str(temp_repo / "test.py")

    record = TrustRecord(
        source_path=source_path,
        doc_path=str(temp_repo / "docs" / "test.py.deep.md"),
        trust_level="stale",
        embedded_checksum="abc",
        current_checksum="def",
        penalty=0.5,
    )
    await trust_store.upsert(record)

    # Update penalty
    await trust_store.update_penalty(source_path, 0.3)

    retrieved = await trust_store.get(source_path)
    assert retrieved.penalty == 0.3
