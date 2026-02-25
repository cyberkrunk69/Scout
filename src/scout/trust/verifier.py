"""
TrustVerifier - AST hash verification and staleness detection.

Philosophy: Right-Size Tooling (deterministic before LLM) + DRY
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import aiofiles

from .constants import TRUST_DB_FILENAME
from .models import VerificationResult

logger = logging.getLogger(__name__)


class TrustLevel:
    """Trust level constants."""
    TRUSTED = "trusted"
    PARTIAL = "partial"
    NO_CHECKSUM = "no_checksum"
    STALE = "stale"
    MISSING = "missing"
    UNVERIFIED = "unverified"


class TrustVerifier:
    """
    Verify trust using AST hash - deterministic, zero cost.

    Philosophy: Right-Size Tooling - no LLM needed for verification
    """

    def __init__(self, repo_root: Path, timeout: float = 5.0):
        self.repo_root = repo_root
        self.timeout = timeout

    async def verify(self, source_path: Path) -> VerificationResult:
        """
        Verify trust for a single source file.

        Args:
            source_path: Path to the source file to verify.

        Returns:
            VerificationResult with trust level and details.

        Raises:
            TimeoutError: If verification times out.
        """
        # Find doc path
        doc_path = await self._find_doc_path(source_path)

        if doc_path is None or not doc_path.exists():
            return VerificationResult(
                source_path=source_path,
                doc_path=source_path,
                trust_level=TrustLevel.MISSING,
                embedded_checksum=None,
                current_checksum=None,
                stale_symbols=[],
                fresh_symbols=[],
                gap_message=f"No docs found for {source_path}",
            )

        # Extract checksums
        embedded = await asyncio.wait_for(
            self._extract_checksum(doc_path), timeout=self.timeout
        )
        current = await asyncio.wait_for(
            self._compute_current_checksum(source_path), timeout=self.timeout
        )

        # Extract and compare symbols
        stale_symbols, fresh_symbols = await asyncio.gather(
            self._extract_stale_symbols(doc_path),
            self._extract_symbols_from_source(source_path),
        )

        # Determine trust level
        if embedded is None:
            trust_level = TrustLevel.NO_CHECKSUM
            gap = "[GAP] No FACT_CHECKSUM — doc may be stale"
        elif current is None:
            trust_level = TrustLevel.UNVERIFIED
            gap = "[GAP] Could not compute current checksum"
        elif embedded != current:
            trust_level = TrustLevel.STALE
            gap = f"[GAP] Stale — expected {embedded[:8]}..., got {current[:8]}..."
        else:
            # Checksum matches - check for stale symbols
            if stale_symbols:
                trust_level = TrustLevel.PARTIAL
                gap = f"[GAP] Checksum matches but {len(stale_symbols)} stale symbols"
            else:
                trust_level = TrustLevel.TRUSTED
                gap = None

        return VerificationResult(
            source_path=source_path,
            doc_path=doc_path,
            trust_level=trust_level,
            embedded_checksum=embedded,
            current_checksum=current,
            stale_symbols=stale_symbols,
            fresh_symbols=fresh_symbols,
            gap_message=gap,
        )

    async def verify_batch(
        self, source_paths: List[Path]
    ) -> List[VerificationResult]:
        """Verify multiple files in parallel."""
        tasks = [self.verify(p) for p in source_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed: List[VerificationResult] = []
        for path, result in zip(source_paths, results):
            if isinstance(result, Exception):
                logger.debug(f"Verification error for {path}: {result}")
                processed.append(
                    VerificationResult(
                        source_path=path,
                        doc_path=path,
                        trust_level=TrustLevel.UNVERIFIED,
                        embedded_checksum=None,
                        current_checksum=None,
                        stale_symbols=[],
                        fresh_symbols=[],
                        gap_message=f"Verification error: {result}",
                    )
                )
            else:
                processed.append(result)  # type: ignore[arg-type]

        return processed

    async def _find_doc_path(self, source_path: Path) -> Optional[Path]:
        """Find doc path for source file (checks .docs and livingDoc)."""
        # Try local .docs/
        local_docs = source_path.parent / ".docs"
        local_path = local_docs / f"{source_path.name}.deep.md"
        if local_path.exists():
            return local_path

        # Try docs/livingDoc/
        try:
            rel_path = source_path.relative_to(self.repo_root)
            central = self.repo_root / "docs" / "livingDoc" / rel_path.parent
            central_path = central / f"{source_path.name}.deep.md"
            if central_path.exists():
                return central_path
        except ValueError:
            pass

        return None

    async def _extract_checksum(self, doc_path: Path) -> Optional[str]:
        """Extract FACT_CHECKSUM from doc file."""
        try:
            async with aiofiles.open(doc_path, "r") as f:
                content = await f.read()

            match = re.search(r"<!-- FACT_CHECKSUM: ([a-f0-9]{64}) -->", content)
            return match.group(1) if match else None
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout reading {doc_path}")
        except Exception as e:
            logger.debug(f"Error extracting checksum from {doc_path}: {e}")
            return None

    async def _compute_current_checksum(self, source_path: Path) -> Optional[str]:
        """Compute current checksum from AST (simplified - uses file hash)."""
        try:
            async with aiofiles.open(source_path, "rb") as f:
                content = await f.read()
            return hashlib.sha256(content).hexdigest()
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout computing checksum for {source_path}")
        except Exception as e:
            logger.debug(f"Error computing checksum for {source_path}: {e}")
            return None

    async def _extract_stale_symbols(self, doc_path: Path) -> List[str]:
        """Extract stale symbols from doc (look for STALE markers)."""
        try:
            async with aiofiles.open(doc_path, "r") as f:
                content = await f.read()
            # Look for stale markers in doc
            stale_pattern = r"<!-- STALE_SYMBOLS: (\[[^\]]+\]) -->"
            match = re.search(stale_pattern, content)
            if match:
                import json

                return json.loads(match.group(1))
            return []
        except Exception:
            return []

    async def _extract_symbols_from_source(self, source_path: Path) -> List[str]:
        """Extract function/class names from source file."""
        try:
            async with aiofiles.open(source_path, "r") as f:
                content = await f.read()
            symbols = []
            # Simple regex extraction for functions and classes
            for match in re.finditer(r"^(?:def|class)\s+(\w+)", content, re.MULTILINE):
                symbols.append(match.group(1))
            return symbols
        except Exception:
            return []
