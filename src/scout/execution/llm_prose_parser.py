"""LLM Prose Parser — Convert high-level plans into structured executable steps.

This module provides the LlmProseParser class that uses an LLM to parse
prose plans into StructuredStep objects conforming to the schema.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from .actions import ActionType, StructuredStep
from scout.config.defaults import LLM_PARSER_DEFAULT_MAX_RETRIES

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================


class ParsingError(Exception):
    """Raised when LLM response fails schema validation."""

    def __init__(self, message: str, raw_response: Optional[str] = None):
        super().__init__(message)
        self.raw_response = raw_response


class LLMError(Exception):
    """Raised when LLM call fails (network, timeout, etc.)."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.cause = cause


# =============================================================================
# Protocol for LLM Client (to support various implementations)
# =============================================================================


class LLMClientProtocol(Protocol):
    """Protocol defining the expected LLM client interface.

    The client should implement either:
    - sync: complete(prompt: str, system: str = None) -> str
    - async: acomplete(prompt: str, system: str = None) -> str

    For token tracking, the client should return metadata including:
    - input_tokens: int
    - output_tokens: int
    - cost_usd: float (optional)
    """

    def complete(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous completion method.

        Args:
            prompt: User prompt text
            system: Optional system prompt

        Returns:
            Dict with keys: content (str), input_tokens (int),
            output_tokens (int), cost_usd (float)
        """
        ...


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are an expert at converting high-level plans into concrete executable steps. Given a prose plan, output a JSON array of steps. Each step must have the following fields:
- action_type: one of ["CREATE_FILE", "MODIFY_FILE", "DELETE_FILE", "RUN_COMMAND", "READ_FILE", "UNKNOWN"]
- description: a short human-readable description of the step
- file_path: (optional) path to a file, required for file actions
- content: (optional) content for create/modify
- command: (optional) shell command for RUN_COMMAND
- args: list of strings (optional arguments)
- working_directory: (optional)
- rollback_info: (optional dict)

Examples:
Input: "Create a new file called hello.py that prints hello"
Output: [{{"action_type": "CREATE_FILE", "description": "Create hello.py", "file_path": "hello.py", "content": "print('hello')", "args": []}}]

Input: "Run tests and then update the readme"
Output: [
  {{"action_type": "RUN_COMMAND", "description": "Run tests", "command": "pytest", "args": []}},
  {{"action_type": "MODIFY_FILE", "description": "Update readme", "file_path": "README.md", "content": "Updated content", "args": []}}
]

Input: "Read the config.yaml file to check settings"
Output: [{{"action_type": "READ_FILE", "description": "Read config file", "file_path": "config.yaml", "args": []}}]

Now convert the following plan: {plan_text}

Respond ONLY with valid JSON array. No additional text."""


# =============================================================================
# CostTracker Protocol (for type hints)
# =============================================================================


class CostTrackerProtocol:
    """Protocol for cost tracking."""

    def record_task(self, result: Any) -> None:
        """Record a task's token/cost metrics."""
        ...

    def get_metrics(self) -> Any:
        """Get aggregate execution metrics."""
        ...


# =============================================================================
# Main Parser Class
# =============================================================================


class LlmProseParser:
    """Parse prose plans into structured steps using an LLM.

    This class wraps an LLM client and provides:
    - Prompt building with few-shot examples
    - Retry logic with exponential backoff
    - Schema validation for StructuredStep
    - Token/cost tracking via CostTracker
    - Comprehensive error handling

    Example:
        parser = LlmProseParser(llm_client=my_client, cost_tracker=tracker)
        steps = parser.parse("Create a new file hello.py with print('hello')")
    """

    def __init__(
        self,
        llm_client: LLMClientProtocol,
        cost_tracker: Optional[CostTrackerProtocol] = None,
        max_retries: int = LLM_PARSER_DEFAULT_MAX_RETRIES,
        model: str = "llama-3.1-8b-instant",
    ):
        """Initialize the parser.

        Args:
            llm_client: LLM client implementing complete() or acomplete()
            cost_tracker: Optional cost tracker for token accounting
            max_retries: Maximum retry attempts (default 3)
            model: Model identifier for the LLM (default llama-3.1-8b-instant)
        """
        self.llm_client = llm_client
        self.cost_tracker = cost_tracker
        self.max_retries = max_retries
        self.model = model
        self.logger = logging.getLogger(__name__)

    def parse(self, plan_text: str) -> List[StructuredStep]:
        """Parse a prose plan into structured steps.

        Args:
            plan_text: The prose plan to parse

        Returns:
            List of StructuredStep objects

        Raises:
            ParsingError: If LLM response fails schema validation
            LLMError: If all LLM attempts fail
            BudgetExceededError: If cost tracker exceeds budget
        """
        prompt = self._build_prompt(plan_text)

        for attempt in range(self.max_retries):
            try:
                response = self._call_llm(prompt, attempt)
                steps_data = self._validate_and_parse(response)
                return [self._dict_to_step(step) for step in steps_data]

            except ParsingError:
                # Don't retry on validation errors - the response won't change
                raise

            except (asyncio.TimeoutError, TimeoutError, LLMError) as e:
                self.logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, ... (doubles each attempt)
                    # Formula: 2^attempt gives delay in seconds
                    sleep_time = 2 ** attempt
                    self.logger.info(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    raise LLMError(f"All {self.max_retries} LLM attempts failed") from e

            except BudgetExceededError:
                # Don't retry on budget errors
                raise

            except Exception as e:
                self.logger.error(f"Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 1s, 2s, 4s, ... (doubles each attempt)
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                else:
                    raise LLMError("All LLM attempts failed", cause=e) from e

        raise ParsingError("Failed to parse plan after all retries")

    def _build_prompt(self, plan_text: str) -> str:
        """Build the prompt for the LLM.

        Args:
            plan_text: The prose plan text

        Returns:
            Formatted prompt string
        """
        return SYSTEM_PROMPT.format(plan_text=plan_text)

    def _call_llm(self, prompt: str, attempt: int) -> Dict[str, Any]:
        """Call the LLM client with the given prompt.

        Args:
            prompt: The formatted prompt
            attempt: Current attempt number (for logging)

        Returns:
            Dict with response content and metadata

        Raises:
            LLMError: On network/timeout errors
            BudgetExceededError: If budget exceeded
        """
        self.logger.debug(f"Calling LLM (attempt {attempt + 1})")

        # Try sync first, then async
        try:
            if hasattr(self.llm_client, "complete"):
                result = self.llm_client.complete(prompt)
            elif hasattr(self.llm_client, "acomplete"):
                # Run async in sync context
                try:
                    import asyncio
                    result = asyncio.run(self.llm_client.acomplete(prompt))
                except RuntimeError:
                    # Already in async context
                    result = None
            else:
                raise LLMError(
                    f"LLM client must implement 'complete' or 'acomplete'. "
                    f"Got: {type(self.llm_client)}"
                )
        except (TimeoutError, asyncio.TimeoutError) as e:
            self.logger.warning(f"LLM call timed out (attempt {attempt + 1})")
            raise LLMError("LLM call timed out", cause=e) from e
        except Exception as e:
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                raise LLMError("LLM call timed out", cause=e) from e
            self.logger.error(f"LLM call failed: {type(e).__name__}: {e}")
            raise LLMError(f"LLM call failed: {e}", cause=e) from e

        # Extract metadata and track costs
        content = result.get("content", "") if isinstance(result, dict) else str(result)
        input_tokens = result.get("input_tokens", 0) if isinstance(result, dict) else 0
        output_tokens = result.get("output_tokens", 0) if isinstance(result, dict) else 0
        cost_usd = result.get("cost_usd", 0.0) if isinstance(result, dict) else 0.0

        self.logger.info(
            f"LLM response: {len(content)} chars, "
            f"{input_tokens + output_tokens} tokens, ${cost_usd:.6f}"
        )

        # Track costs if cost tracker is available
        if self.cost_tracker:
            self._track_cost(input_tokens, output_tokens, cost_usd)

        return {"content": content, "input_tokens": input_tokens, "output_tokens": output_tokens}

    def _track_cost(self, input_tokens: int, output_tokens: int, cost_usd: float) -> None:
        """Track token usage via cost tracker.

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cost_usd: Cost in USD

        Raises:
            BudgetExceededError: If budget is exceeded
        """
        if not self.cost_tracker:
            return

        try:
            # Check if budget would be exceeded
            if hasattr(self.cost_tracker, "check_budget"):
                self.cost_tracker.check_budget(cost_usd)

            # Record the cost
            if hasattr(self.cost_tracker, "record_cost"):
                self.cost_tracker.record_cost(
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                )
            elif hasattr(self.cost_tracker, "record_task"):
                # Fallback: create a TaskResult-like object
                class FakeResult:
                    tokens = input_tokens + output_tokens
                    cost = cost_usd
                    error = None

                self.cost_tracker.record_task(FakeResult())

        except BudgetExceededError:
            raise
        except Exception as e:
            self.logger.warning(f"Failed to track cost: {e}")

    def _validate_and_parse(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate and parse the LLM response.

        Args:
            response: Response dict with 'content' key

        Returns:
            List of step dicts

        Raises:
            ParsingError: If JSON is invalid or schema doesn't match
        """
        content = response.get("content", "")

        # Try to extract JSON from response (handle markdown code blocks)
        json_str = self._extract_json(content)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in LLM response: {e}")
            self.logger.debug(f"Raw response: {content[:500]}...")
            raise ParsingError(f"Invalid JSON: {e}", raw_response=content) from e

        # Ensure it's a list
        if not isinstance(data, list):
            raise ParsingError(
                f"Expected JSON array, got {type(data).__name__}", raw_response=content
            )

        # Validate each step
        for i, step in enumerate(data):
            self._validate_step(step, i, content)

        return data

    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response (handles markdown code blocks).

        Args:
            content: Raw LLM response

        Returns:
            Extracted JSON string
        """
        content = content.strip()

        # Handle markdown code blocks
        if content.startswith("```"):
            # Find the JSON block
            lines = content.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```"):
                    in_json = not in_json
                    continue
                if in_json:
                    json_lines.append(line)
            content = "\n".join(json_lines)

        return content.strip()

    def _validate_step(self, step: Dict[str, Any], index: int, raw_response: str) -> None:
        """Validate a single step against the schema.

        Args:
            step: Step dict to validate
            index: Step index (for error messages)
            raw_response: Raw response for error context

        Raises:
            ParsingError: If step is invalid
        """
        required_fields = ["action_type", "description"]

        for field in required_fields:
            if field not in step:
                raise ParsingError(
                    f"Step {index} missing required field: {field}", raw_response=raw_response
                )

        # Validate action_type
        valid_types = [t.value for t in ActionType]
        action_type = step.get("action_type", "").upper()

        # Handle case variations
        if action_type not in valid_types:
            # Try with underscore conversion
            action_type_normalized = action_type.replace("-", "_").lower()
            type_mapping = {
                "CREATE_FILE": "create_file",
                "MODIFY_FILE": "modify_file",
                "DELETE_FILE": "delete_file",
                "RUN_COMMAND": "run_command",
                "READ_FILE": "read_file",
                "UNKNOWN": "unknown",
            }
            if action_type_normalized in type_mapping.values():
                # Use the normalized value to find the correct key
                step["action_type"] = action_type_normalized
            else:
                self.logger.warning(
                    f"Step {index} has invalid action_type: {step.get('action_type')}, "
                    f"defaulting to UNKNOWN"
                )
                step["action_type"] = "unknown"

        # Validate file-related actions have file_path
        file_actions = ["CREATE_FILE", "MODIFY_FILE", "DELETE_FILE"]
        if step.get("action_type") in file_actions and not step.get("file_path"):
            self.logger.warning(
                f"Step {index} is file action but missing file_path, setting to UNKNOWN"
            )
            step["action_type"] = "unknown"

    def _dict_to_step(self, step_dict: Dict[str, Any]) -> StructuredStep:
        """Convert a dict to a StructuredStep dataclass.

        Args:
            step_dict: Validated step dictionary

        Returns:
            StructuredStep instance
        """
        # Normalize action_type to ActionType enum
        action_type_str = step_dict.get("action_type", "unknown")
        try:
            action_type = ActionType(action_type_str)
        except ValueError:
            action_type = ActionType.UNKNOWN

        return StructuredStep(
            action_type=action_type,
            description=step_dict.get("description", ""),
            file_path=step_dict.get("file_path"),
            content=step_dict.get("content"),
            command=step_dict.get("command"),
            args=step_dict.get("args", []),
            working_directory=step_dict.get("working_directory"),
            rollback_info=step_dict.get("rollback_info"),
        )


# =============================================================================
# BudgetExceededError (re-export from doc_generation for compatibility)
# =============================================================================

# Import from doc_generation if available, otherwise define locally
try:
    from scout.doc_generation import BudgetExceededError
except ImportError:

    class BudgetExceededError(RuntimeError):
        """Raised when token cost exceeds the configured budget."""

        def __init__(self, current_cost: float, budget: float):
            self.current_cost = current_cost
            self.budget = budget
            super().__init__(f"Cost ${current_cost:.4f} exceeds budget ${budget:.4f}")
