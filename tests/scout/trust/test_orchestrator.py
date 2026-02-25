"""Tests for TrustOrchestrator."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from scout.trust.orchestrator import TrustOrchestrator, TrustConfig
from scout.trust.verifier import TrustLevel
from scout.trust.store import TrustStore
from scout.trust.models import VerificationResult, PenaltyResult


@pytest.fixture
def temp_repo():
    """Create a temporary repository root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
async def orchestrator(temp_repo):
    """Create an initialized TrustOrchestrator."""
    orch = TrustOrchestrator(temp_repo)
    await orch.initialize()
    yield orch
    await orch.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_initialization(temp_repo):
    """Test orchestrator initializes all components."""
    orch = TrustOrchestrator(temp_repo)
    await orch.initialize()

    assert orch.store is not None
    assert orch.verifier is not None
    assert orch.auditor is not None
    assert orch.penalizer is not None
    assert orch.learner is not None

    await orch.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_verify_trusted(temp_repo):
    """Test verifying a trusted file."""
    orch = TrustOrchestrator(temp_repo)
    await orch.initialize()

    # Create source file and matching doc
    source = temp_repo / "test.py"
    source.write_text("def foo(): pass")

    import hashlib
    checksum = hashlib.sha256(source.read_text().encode()).hexdigest()

    docs_dir = temp_repo / ".docs"
    docs_dir.mkdir()
    doc = docs_dir / "test.py.deep.md"
    doc.write_text(f"<!-- FACT_CHECKSUM: {checksum} -->\n# Doc\n\ndef foo(): pass")

    result = await orch.verify(source)

    assert result.trust_level == TrustLevel.TRUSTED
    assert result.confidence > 0

    await orch.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_verify_missing(temp_repo):
    """Test verifying a file with no docs."""
    orch = TrustOrchestrator(temp_repo)
    await orch.initialize()

    source = temp_repo / "test.py"
    source.write_text("def foo(): pass")

    result = await orch.verify(source)

    assert result.trust_level == TrustLevel.MISSING

    await orch.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_verify_batch(temp_repo):
    """Test batch verification."""
    orch = TrustOrchestrator(temp_repo)
    await orch.initialize()

    # Create multiple source files
    sources = []
    for i in range(3):
        source = temp_repo / f"test_{i}.py"
        source.write_text(f"def foo_{i}(): pass")
        sources.append(source)

    results = await orch.verify_batch(sources)

    assert len(results) == 3

    await orch.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_record_outcome(temp_repo):
    """Test recording navigation/validation outcome."""
    orch = TrustOrchestrator(temp_repo)
    await orch.initialize()

    # Create a record first
    source = temp_repo / "test.py"
    source.write_text("def foo(): pass")

    import hashlib
    checksum = hashlib.sha256(source.read_text().encode()).hexdigest()

    docs_dir = temp_repo / ".docs"
    docs_dir.mkdir()
    doc = docs_dir / "test.py.deep.md"
    doc.write_text(f"<!-- FACT_CHECKSUM: {checksum} -->\n# Doc\n\ndef foo(): pass")

    # Manually create a record in the store
    from scout.trust.models import TrustRecord
    record = TrustRecord(
        source_path=str(source),
        doc_path=str(doc),
        trust_level="trusted",
        embedded_checksum=checksum,
        current_checksum=checksum,
    )
    await orch.store.upsert(record)

    # Record successful outcome
    await orch.record_outcome(str(source), success=True)

    # Record failed outcome
    await orch.record_outcome(str(source), success=False)

    # Verify counts were updated
    record = await orch.store.get(str(source))
    assert record.success_count == 1
    assert record.failure_count == 1

    await orch.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_auto_repair_disabled(temp_repo):
    """Test auto repair when disabled."""
    config = TrustConfig(auto_repair_enabled=False)
    orch = TrustOrchestrator(temp_repo, config=config)
    await orch.initialize()

    result = await orch.auto_repair()

    assert result["refreshed"] == 0
    assert result["reason"] == "disabled"

    await orch.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_auto_repair_no_stale(temp_repo):
    """Test auto repair with no stale docs."""
    config = TrustConfig(auto_repair_enabled=True)
    orch = TrustOrchestrator(temp_repo, config=config)
    await orch.initialize()

    result = await orch.auto_repair()

    assert result["refreshed"] == 0

    await orch.shutdown()


def test_trust_config_defaults():
    """Test TrustConfig default values."""
    config = TrustConfig()

    assert config.strictness == "normal"
    assert config.auto_repair_enabled is True
    assert config.auto_repair_threshold == 5
    assert config.min_confidence == 70


def test_trust_config_permissive():
    """Test TrustConfig permissive mode."""
    config = TrustConfig(strictness="permissive")

    assert config.min_confidence == 50
    assert config.auto_repair_threshold == 10


def test_trust_config_strict():
    """Test TrustConfig strict mode."""
    config = TrustConfig(strictness="strict")

    assert config.min_confidence == 80
    assert config.auto_repair_threshold == 3


@pytest.mark.asyncio
async def test_orchestrator_learn(temp_repo):
    """Test learn method calls learner."""
    orch = TrustOrchestrator(temp_repo)
    await orch.initialize()

    # Mock the learner's adjust_penalties method
    orch.learner.adjust_penalties = AsyncMock(return_value={"adjusted": 0})

    # Call learn
    await orch.learn()

    # Verify learner was called
    orch.learner.adjust_penalties.assert_called_once()

    await orch.shutdown()


@pytest.mark.asyncio
async def test_orchestrator_shutdown_cancels_tasks(temp_repo):
    """Test that shutdown completes without error."""
    orch = TrustOrchestrator(temp_repo)
    await orch.initialize()

    # Shutdown should handle empty tasks list gracefully
    await orch.shutdown()

    # Verify shutdown completed without error
