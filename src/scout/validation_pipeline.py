"""
Validation Pipeline for ToolOutput.

Provides a staged validation pipeline:
- DETERMINISTIC: Zero-cost validators (schema, content type, path existence)
- HEURISTIC: Cheap checks (length, confidence threshold, duplicates)
- LLM_BASED: Optional LLM scoring (defer to Phase 3)

Phase 2 of Unified Tool Framework.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional

from scout.tool_output import ToolOutput, ToolOutputRegistry


# Validation error codes
class ValidationErrorCode(str, Enum):
    """Standard validation error codes."""
    VALID = "VALID"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD"
    INVALID_CONTENT_TYPE = "INVALID_CONTENT_TYPE"
    JSON_PARSE_ERROR = "JSON_PARSE_ERROR"
    MISSING_PATH = "MISSING_PATH"
    HALLUCINATED_SYMBOL = "HALLUCINATED_SYMBOL"
    CONTENT_TOO_SHORT = "CONTENT_TOO_SHORT"
    CONTENT_TOO_LONG = "CONTENT_TOO_LONG"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    DUPLICATE_OUTPUT = "DUPLICATE_OUTPUT"


class ValidationStage(str, Enum):
    """Pipeline stages - ordered by cost."""
    DETERMINISTIC = "deterministic"
    HEURISTIC = "heuristic"
    LLM_BASED = "llm_based"


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationError:
    """Structured validation failure."""
    code: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    field: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "field": self.field,
            "suggestion": self.suggestion,
        }


@dataclass
class ValidationResult:
    """Aggregate result from validation pipeline."""
    is_valid: bool
    stage_reached: ValidationStage
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    duration_ms: float = 0.0
    validator_versions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "stage_reached": self.stage_reached.value,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "duration_ms": self.duration_ms,
            "validator_versions": self.validator_versions,
        }

    def to_error_strings(self) -> List[str]:
        """Convert errors to simple string list for ToolOutput.validation_errors."""
        return [f"[{e.code}] {e.message}" for e in self.errors]


class BaseValidator:
    """Abstract base class for validators."""

    stage: ValidationStage
    code: str

    def validate(self, output: ToolOutput, repo_root: Optional[Path] = None) -> Optional[ValidationError]:
        """Validate the output. Returns None if valid, ValidationError otherwise."""
        raise NotImplementedError

    @property
    def version(self) -> str:
        return "1.0.0"


# === Deterministic Validators ===

class OutputSchemaValidator(BaseValidator):
    """Validate ToolOutput has required fields."""

    stage = ValidationStage.DETERMINISTIC
    code = "INVALID_SCHEMA"

    def validate(self, output: ToolOutput, repo_root: Optional[Path] = None) -> Optional[ValidationError]:
        if not output.tool_name:
            return ValidationError(
                code=self.code,
                message="Missing required field: tool_name",
                field="tool_name",
                suggestion="Provide a non-empty tool_name",
            )
        if output.content is None:
            return ValidationError(
                code=self.code,
                message="Missing required field: content",
                field="content",
                suggestion="Provide content for the output",
            )
        if not output.output_id:
            return ValidationError(
                code=self.code,
                message="Missing required field: output_id",
                field="output_id",
                suggestion="Output should have a valid UUID",
            )
        return None


class ContentTypeValidator(BaseValidator):
    """Verify content matches expected type."""

    stage = ValidationStage.DETERMINISTIC
    code = "INVALID_CONTENT_TYPE"

    def validate(self, output: ToolOutput, repo_root: Optional[Path] = None) -> Optional[ValidationError]:
        content = output.content
        if content is None:
            return None  # Already caught by schema validator

        valid_types = (str, dict, list, int, float, bool)
        if not isinstance(content, valid_types):
            return ValidationError(
                code=self.code,
                message=f"Content must be str, dict, or list, got {type(content).__name__}",
                field="content",
                suggestion=f"Convert content to JSON-serializable type",
            )
        return None


class JsonParseValidator(BaseValidator):
    """If content claims to be JSON string, validate it's parseable."""

    stage = ValidationStage.DETERMINISTIC
    code = "JSON_PARSE_ERROR"

    def validate(self, output: ToolOutput, repo_root: Optional[Path] = None) -> Optional[ValidationError]:
        if not isinstance(output.content, str):
            return None

        content = output.content.strip()
        if content.startswith(("{", "[")) and not self._is_valid_json(content):
            return ValidationError(
                code=self.code,
                message="Content appears to be JSON but failed to parse",
                field="content",
                suggestion="Fix JSON syntax or wrap as plain string",
            )
        return None

    def _is_valid_json(self, text: str) -> bool:
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


class PathReferenceValidator(BaseValidator):
    """Resolve and verify file paths referenced in content."""

    stage = ValidationStage.DETERMINISTIC
    code = "MISSING_PATH"

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path.cwd()

    def validate(self, output: ToolOutput, repo_root: Optional[Path] = None) -> Optional[ValidationError]:
        repo = repo_root or self.repo_root
        paths = self._extract_paths(output.content)

        for path_str in paths:
            path = Path(path_str)
            exists, resolved, _ = self._path_exists_safe(path, repo)
            if not exists:
                return ValidationError(
                    code=self.code,
                    message=f"Referenced path does not exist: {path_str}",
                    field="content",
                    suggestion=f"Verify file exists or remove reference",
                )
        return None

    def _extract_paths(self, content: Any) -> List[str]:
        """Extract potential file paths from content."""
        paths = []
        if isinstance(content, str):
            # Match common path patterns
            path_pattern = re.compile(r'(?:[\w./\\-]+/)?[\w./\\-]+\.\w+')
            matches = path_pattern.findall(content)
            paths.extend([m for m in matches if self._looks_like_path(m)])
        elif isinstance(content, dict):
            for value in content.values():
                paths.extend(self._extract_paths(value))
        elif isinstance(content, list):
            for item in content:
                paths.extend(self._extract_paths(item))
        return paths

    def _looks_like_path(self, text: str) -> bool:
        """Heuristic: looks like a file path."""
        return "/" in text or "\\" in text or text.endswith((".py", ".md", ".txt", ".json", ".yaml", ".yml"))

    def _path_exists_safe(self, path: Path, repo_root: Path) -> tuple[bool, Optional[Path], bool]:
        """Check path existence with symlink loop detection."""
        try:
            resolved = (repo_root / path).resolve()
        except RuntimeError:
            return False, None, True
        visited: set[Path] = set()
        current = resolved
        while current.is_symlink():
            if current in visited:
                return False, None, True
            visited.add(current)
            try:
                current = current.resolve()
            except (OSError, RuntimeError):
                return False, None, False
        return current.exists(), (current if current.exists() else resolved), False


class SymbolReferenceValidator(BaseValidator):
    """Validate code symbols (functions/classes) referenced in content exist."""

    stage = ValidationStage.DETERMINISTIC
    code = "HALLUCINATED_SYMBOL"

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path.cwd()

    def validate(self, output: ToolOutput, repo_root: Optional[Path] = None) -> Optional[ValidationError]:
        repo = repo_root or self.repo_root
        symbols = self._extract_symbols(output.content)

        for symbol_info in symbols:
            file_path = symbol_info.get("file")
            symbol_name = symbol_info.get("symbol")

            if not file_path or not symbol_name:
                continue

            file_p = Path(file_path)
            if not file_p.exists():
                continue  # Path validator will catch this

            found_line, found_symbol, _ = self._grep_symbol(file_p, symbol_name)
            if found_line is None:
                return ValidationError(
                    code=self.code,
                    message=f"Symbol '{symbol_name}' not found in {file_path}",
                    field="content",
                    suggestion=f"Verify function/class exists or remove reference",
                )
        return None

    def _extract_symbols(self, content: Any) -> List[dict]:
        """Extract symbol references from content."""
        symbols = []
        if isinstance(content, dict):
            # Look for common patterns
            for key in ("symbol", "function", "class", "method"):
                if key in content:
                    symbol_entry = {"symbol": content[key]}
                    if "file" in content:
                        symbol_entry["file"] = content["file"]
                    symbols.append(symbol_entry)
            for value in content.values():
                symbols.extend(self._extract_symbols(value))
        elif isinstance(content, list):
            for item in content:
                symbols.extend(self._extract_symbols(item))
        return symbols

    def _grep_symbol(self, file_path: Path, symbol: str) -> tuple[Optional[int], Optional[str], Optional[str]]:
        """Grep for def or class in file."""
        pattern_def = re.compile(rf"^\s*def\s+{re.escape(symbol)}\s*\(")
        pattern_class = re.compile(rf"^\s*class\s+{re.escape(symbol)}\s*[:(]")
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None, None, None
        for i, line in enumerate(lines):
            if pattern_def.match(line) or pattern_class.match(line):
                snippet_lines = lines[i : i + 3]
                snippet = "\n".join(snippet_lines) if snippet_lines else None
                return i + 1, symbol, snippet
        return None, None, None


# === Heuristic Validators ===

class LengthValidator(BaseValidator):
    """Flag outputs that are suspiciously short or too long."""

    stage = ValidationStage.HEURISTIC
    code = "CONTENT_TOO_SHORT"

    def __init__(self, min_length: int = 10, max_length: int = 100_000):
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, output: ToolOutput, repo_root: Optional[Path] = None) -> Optional[ValidationError]:
        content_str = self._get_content_string(output.content)
        length = len(content_str)

        if length < self.min_length:
            return ValidationError(
                code=self.code,
                message=f"Content suspiciously short: {length} chars (min: {self.min_length})",
                field="content",
                suggestion="Output may be incomplete",
                severity=ValidationSeverity.WARNING,
            )

        if length > self.max_length:
            return ValidationError(
                code="CONTENT_TOO_LONG",
                message=f"Content exceeds maximum length: {length} chars (max: {self.max_length})",
                field="content",
                suggestion="Consider splitting output",
                severity=ValidationSeverity.WARNING,
            )

        return None

    def _get_content_string(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            return json.dumps(content)
        return str(content)


class ConfidenceThresholdValidator(BaseValidator):
    """Fail if confidence is below threshold."""

    stage = ValidationStage.HEURISTIC
    code = "LOW_CONFIDENCE"

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def validate(self, output: ToolOutput, repo_root: Optional[Path] = None) -> Optional[ValidationError]:
        if output.confidence is None:
            return None  # No confidence = skip

        if output.confidence < self.min_confidence:
            return ValidationError(
                code=self.code,
                message=f"Confidence too low: {output.confidence:.2f} (min: {self.min_confidence})",
                field="confidence",
                suggestion="Regenerate with higher confidence or adjust threshold",
                severity=ValidationSeverity.WARNING,
            )
        return None


class DuplicateOutputValidator(BaseValidator):
    """Check content hash against recent outputs."""

    stage = ValidationStage.HEURISTIC
    code = "DUPLICATE_OUTPUT"

    def __init__(self, registry: Optional[ToolOutputRegistry] = None, lookback: int = 10):
        self.registry = registry
        self.lookback = lookback

    def validate(self, output: ToolOutput, repo_root: Optional[Path] = None) -> Optional[ValidationError]:
        reg = self.registry or ToolOutputRegistry()
        content_hash = self._hash_content(output.content)

        # Get recent outputs
        recent_ids = reg.list()[-self.lookback:] if reg.list() else []

        for output_id in recent_ids:
            if output_id == output.output_id:
                continue
            prev = reg.load(output_id)
            if prev and self._hash_content(prev.content) == content_hash:
                return ValidationError(
                    code=self.code,
                    message=f"Duplicate content detected (matches {output_id})",
                    field="content",
                    suggestion="Ensure output is unique",
                    severity=ValidationSeverity.WARNING,
                )
        return None

    def _hash_content(self, content: Any) -> str:
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]


# === Pipeline ===

class ValidationPipeline:
    """Executes validators in stages."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self._validators: dict[ValidationStage, List[BaseValidator]] = {
            ValidationStage.DETERMINISTIC: [],
            ValidationStage.HEURISTIC: [],
            ValidationStage.LLM_BASED: [],
        }
        self._setup_default_validators()

    def _setup_default_validators(self) -> None:
        """Add default validators based on config."""
        # Deterministic validators
        self._validators[ValidationStage.DETERMINISTIC] = [
            OutputSchemaValidator(),
            ContentTypeValidator(),
            JsonParseValidator(),
        ]

        # Add path/symbol validators if repo_root available
        # These will be added dynamically in run() if repo_root provided

        # Heuristic validators
        min_conf = self.config.get("min_confidence", 0.5)
        max_len = self.config.get("max_content_length", 100_000)
        min_len = self.config.get("min_content_length", 10)

        self._validators[ValidationStage.HEURISTIC] = [
            LengthValidator(min_length=min_len, max_length=max_len),
            ConfidenceThresholdValidator(min_confidence=min_conf),
        ]

    def add_validator(self, stage: ValidationStage, validator: BaseValidator) -> None:
        """Add a validator to a stage."""
        self._validators[stage].append(validator)

    def run(
        self,
        output: ToolOutput,
        repo_root: Optional[Path] = None,
        max_stage: Optional[ValidationStage] = None,
        fail_fast: bool = True,
    ) -> ValidationResult:
        """Run validation pipeline on output."""
        start_time = time.perf_counter()
        stage_order = [ValidationStage.DETERMINISTIC, ValidationStage.HEURISTIC, ValidationStage.LLM_BASED]

        # Dynamically add path validators if repo_root provided
        if repo_root:
            self._validators[ValidationStage.DETERMINISTIC].append(
                PathReferenceValidator(repo_root)
            )
            self._validators[ValidationStage.DETERMINISTIC].append(
                SymbolReferenceValidator(repo_root)
            )

        # Add duplicate validator with registry
        self._validators[ValidationStage.HEURISTIC].append(DuplicateOutputValidator())

        all_errors: List[ValidationError] = []
        all_warnings: List[ValidationError] = []
        reached_stage = ValidationStage.DETERMINISTIC
        validator_versions = {}

        for stage in stage_order:
            if max_stage and stage.value > max_stage.value:
                break

            validators = self._validators.get(stage, [])
            for validator in validators:
                validator_versions[validator.__class__.__name__] = validator.version
                error = validator.validate(output, repo_root)

                if error:
                    if error.severity == ValidationSeverity.ERROR:
                        all_errors.append(error)
                        if fail_fast:
                            break
                    else:
                        all_warnings.append(error)

            if fail_fast and all_errors:
                break

            reached_stage = stage

        duration_ms = (time.perf_counter() - start_time) * 1000

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            stage_reached=reached_stage,
            errors=all_errors,
            warnings=all_warnings,
            duration_ms=duration_ms,
            validator_versions=validator_versions,
        )


# === Convenience Functions ===

_default_pipeline: Optional[ValidationPipeline] = None


def get_pipeline(config: Optional[dict] = None) -> ValidationPipeline:
    """Get default validation pipeline."""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = ValidationPipeline(config)
    return _default_pipeline


def validate_output(
    output: ToolOutput,
    repo_root: Optional[Path] = None,
    config: Optional[dict] = None,
) -> ValidationResult:
    """Convenience function to validate a ToolOutput."""
    pipeline = get_pipeline(config)
    return pipeline.run(output, repo_root)
