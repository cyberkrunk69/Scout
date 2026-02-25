"""Tests for TrustVerifier."""

import pytest
import tempfile
from pathlib import Path

from scout.trust.verifier import TrustVerifier, TrustLevel
from scout.trust.models import VerificationResult


@pytest.fixture
def temp_repo():
    """Create a temporary repository root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def verifier(temp_repo):
    """Create a TrustVerifier."""
    return TrustVerifier(temp_repo)


@pytest.mark.asyncio
async def test_verifier_missing_docs(verifier, temp_repo):
    """Test verification when no docs exist."""
    source = temp_repo / "test.py"
    source.write_text("def foo(): pass")

    result = await verifier.verify(source)

    assert result.trust_level == TrustLevel.MISSING
    assert result.gap_message is not None
    assert "No docs found" in result.gap_message


@pytest.mark.asyncio
async def test_verifier_no_checksum(verifier, temp_repo):
    """Test verification when doc exists but no checksum."""
    source = temp_repo / "test.py"
    source.write_text("def foo(): pass")

    # Create doc without checksum
    docs_dir = temp_repo / ".docs"
    docs_dir.mkdir()
    doc = docs_dir / "test.py.deep.md"
    doc.write_text("# Test documentation\n\nNo checksum here.")

    result = await verifier.verify(source)

    assert result.trust_level == TrustLevel.NO_CHECKSUM
    assert result.gap_message is not None
    assert "FACT_CHECKSUM" in result.gap_message


@pytest.mark.asyncio
async def test_verifier_stale(verifier, temp_repo):
    """Test verification when checksum doesn't match."""
    source = temp_repo / "test.py"
    source.write_text("def foo(): pass")

    # Create doc with different checksum (64-char hex string)
    docs_dir = temp_repo / ".docs"
    docs_dir.mkdir()
    doc = docs_dir / "test.py.deep.md"
    doc.write_text("<!-- FACT_CHECKSUM: 0000000000000000000000000000000000000000000000000000000000000000 -->\n# Documentation")

    result = await verifier.verify(source)

    assert result.trust_level == TrustLevel.STALE
    assert result.gap_message is not None
    assert "Stale" in result.gap_message


@pytest.mark.asyncio
async def test_verifier_trusted(verifier, temp_repo):
    """Test verification when checksum matches."""
    source_content = "def foo(): pass"
    source = temp_repo / "test.py"
    source.write_text(source_content)

    # Compute expected checksum
    import hashlib
    expected_checksum = hashlib.sha256(source_content.encode()).hexdigest()

    # Create doc with matching checksum
    docs_dir = temp_repo / ".docs"
    docs_dir.mkdir()
    doc = docs_dir / "test.py.deep.md"
    doc.write_text(f"<!-- FACT_CHECKSUM: {expected_checksum} -->\n# Documentation\n\ndef foo(): pass")

    result = await verifier.verify(source)

    assert result.trust_level == TrustLevel.TRUSTED
    assert result.gap_message is None
    assert result.embedded_checksum == expected_checksum
    assert result.current_checksum == expected_checksum


@pytest.mark.asyncio
async def test_verifier_batch(verifier, temp_repo):
    """Test batch verification."""
    # Create multiple source files
    sources = []
    for i in range(3):
        source = temp_repo / f"test_{i}.py"
        source.write_text(f"def foo_{i}(): pass")
        sources.append(source)

    results = await verifier.verify_batch(sources)

    assert len(results) == 3
    # All should be MISSING since no docs exist
    for result in results:
        assert result.trust_level == TrustLevel.MISSING


@pytest.mark.asyncio
async def test_find_doc_path_local(verifier, temp_repo):
    """Test finding doc in local .docs directory."""
    source = temp_repo / "test.py"
    source.write_text("def foo(): pass")

    docs_dir = temp_repo / ".docs"
    docs_dir.mkdir()
    doc = docs_dir / "test.py.deep.md"
    doc.write_text("<!-- FACT_CHECKSUM: abc -->\n# Doc")

    doc_path = await verifier._find_doc_path(source)

    assert doc_path is not None
    assert doc_path.exists()
    assert doc_path.name == "test.py.deep.md"


@pytest.mark.asyncio
async def test_find_doc_path_central(verifier, temp_repo):
    """Test finding doc in central docs/livingDoc directory."""
    source = temp_repo / "src" / "test.py"
    source.parent.mkdir(parents=True)
    source.write_text("def foo(): pass")

    docs_dir = temp_repo / "docs" / "livingDoc" / "src"
    docs_dir.mkdir(parents=True)
    doc = docs_dir / "test.py.deep.md"
    doc.write_text("<!-- FACT_CHECKSUM: abc -->\n# Doc")

    doc_path = await verifier._find_doc_path(source)

    assert doc_path is not None
    assert doc_path.exists()


def test_trust_level_constants():
    """Test TrustLevel constants are defined."""
    assert TrustLevel.TRUSTED == "trusted"
    assert TrustLevel.PARTIAL == "partial"
    assert TrustLevel.NO_CHECKSUM == "no_checksum"
    assert TrustLevel.STALE == "stale"
    assert TrustLevel.MISSING == "missing"
    assert TrustLevel.UNVERIFIED == "unverified"
