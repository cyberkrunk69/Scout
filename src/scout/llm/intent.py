"""
Intent classification module for Scout MCP server.

Classifies user requests into intent types and extracts metadata
for routing to appropriate tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from scout.llm import LLMResponse, call_groq_async


class IntentType(Enum):
    """Supported intent types for Scout operations."""

    IMPLEMENT_FEATURE = "implement_feature"
    FIX_BUG = "fix_bug"
    REFACTOR = "refactor"
    QUERY_CODE = "query_code"
    OPTIMIZE = "optimize"
    DOCUMENT = "document"
    TEST = "test"
    DEPLOY = "deploy"
    UNKNOWN = "unknown"


QUICK_PATTERNS: dict[re.Pattern[str], IntentType] = {
    re.compile(
        r"what (does|is|do) .* (do|mean|function|class)", re.I
    ): IntentType.QUERY_CODE,
    re.compile(r"fix .*(bug|error|issue|problem)", re.I): IntentType.FIX_BUG,
    re.compile(
        r"add .*(feature|support|capability)", re.I
    ): IntentType.IMPLEMENT_FEATURE,
    re.compile(r"refactor|restructure|reorganize", re.I): IntentType.REFACTOR,
    re.compile(r"optimize|perf|improve|speed", re.I): IntentType.OPTIMIZE,
    re.compile(
        r"(write|add|create|generate) (tests?|docs?|documentation)", re.I
    ): IntentType.DOCUMENT,
    re.compile(r"(write|add|create) tests?", re.I): IntentType.TEST,
    re.compile(
        r"deploy|release|push to (prod|production|staging)", re.I
    ): IntentType.DEPLOY,
}

QUICK_TARGET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(auth|login|oauth|jwt)", re.I), "auth"),
    (re.compile(r"(api|endpoint|route)", re.I), "api"),
    (re.compile(r"(database|db|sql|model)", re.I), "database"),
    (re.compile(r"(test|spec)", re.I), "tests"),
    (re.compile(r"(doc|readme)", re.I), "docs"),
]


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent_type: IntentType
    target: Optional[str] = None
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    clarifying_questions: list[str] = field(default_factory=list)


SYSTEM_PROMPT = """You are an intent classifier for Scout, a code operations MCP server.
Given a user request, classify it into one of these intent types:
- implement_feature: Adding new functionality
- fix_bug: Fixing errors or bugs
- refactor: Restructuring existing code
- query_code: Understanding code or asking questions
- optimize: Performance improvements
- document: Documentation tasks
- test: Test-related tasks
- deploy: Deployment tasks
- unknown: Cannot determine intent

Extract:
1. intent_type: The classification
2. target: The file, module, or function being referenced (if any)
3. confidence: How confident you are (0.0-1.0)
4. metadata: Any additional relevant information
5. clarifying_questions: Questions to ask if confidence < 0.7

Respond in this JSON format:
{"intent_type": "...", "target": "...", "confidence": 0.0, "metadata": {}
    , "clarifying_questions": []}"""

CLASSIFICATION_PROMPT = """Classify this request: "{request}"

Consider:
- Look for specific file/module names mentioned
- Identify action verbs (fix, add, refactor, etc.)
- Determine scope if mentioned (e.g., "function", "module", "entire")

Respond with valid JSON only."""


class IntentClassifier:
    """
    Classifies user requests into intent types using quick patterns
    and LLM for complex cases.

    Example usage:

    ```python
    classifier = IntentClassifier()

    # Quick pattern match
    result = await classifier.classify("fix the auth bug")
    # -> IntentResult(intent_type=FIX_BUG, target="auth", confidence=0.9, ...)

    # Complex classification via LLM
    result = await classifier.classify("optimize the database queries in user.py")
    # -> IntentResult(intent_type=OPTIMIZE, target="user.py", confidence=0.88, ...)

    # With custom LLM callable
    classifier = IntentClassifier(llm_call=my_custom_llm)
    result = await classifier.classify("add oauth support")
    ```
    """

    def __init__(
        self,
        model: str = "llama-3.1-8b-instant",
        max_tokens: int = 200,
        min_confidence_threshold: float = 0.7,
        llm_call: Optional[Callable] = None,
    ):
        """
        Initialize the classifier.

        Args:
            model: LLM model to use for classification.
            max_tokens: Max tokens for LLM response.
            min_confidence_threshold: Minimum confidence to accept LLM result
                                      without asking clarifying questions.
            llm_call: Optional callable for LLM calls. Defaults to call_groq_async.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.min_confidence_threshold = min_confidence_threshold
        self.llm_call = llm_call or call_groq_async

    async def classify(self, request: str) -> IntentResult:
        """
        Classify user request into intent type + extract metadata.

        Args:
            request: The user's natural language request.

        Returns:
            IntentResult with classified intent and metadata.
        """
        request = request.strip()
        if not request:
            return IntentResult(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                clarifying_questions=["Please describe what you'd like to do."],
            )

        quick_result = self._try_quick_match(request)
        if quick_result:
            return quick_result

        return await self._classify_with_llm(request)

    def _try_quick_match(self, request: str) -> Optional[IntentResult]:
        """Try to match using quick patterns for obvious cases."""
        for pattern, intent_type in QUICK_PATTERNS.items():
            if pattern.search(request):
                target = self._extract_target(request)
                return IntentResult(
                    intent_type=intent_type,
                    target=target,
                    confidence=0.9,
                    metadata={"matched_pattern": pattern.pattern},
                )
        return None

    def _extract_target(self, request: str) -> Optional[str]:
        """Extract potential target from request using patterns."""
        request_lower = request.lower()
        for pattern, target in QUICK_TARGET_PATTERNS:
            if pattern.search(request_lower):
                return target

        words = request.split()
        for word in words:
            if word.endswith((".py", ".ts", ".js", ".md")):
                return word
            if "/" in word and not word.startswith("-"):
                return word.split("/")[-1].replace(".py", "")

        return None

    async def _classify_with_llm(self, request: str) -> IntentResult:
        """Use LLM for complex classification."""
        try:
            prompt = CLASSIFICATION_PROMPT.format(request=request)
            response: LLMResponse = await self.llm_call(
                prompt=prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.0,
                system=SYSTEM_PROMPT,
            )

            result = self._parse_llm_response(response.content)
            result.metadata["llm_cost_usd"] = response.cost_usd

            if result.confidence < self.min_confidence_threshold:
                result.clarifying_questions = self._generate_clarifying_questions(
                    request, result
                )

            return result

        except Exception as e:
            return IntentResult(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                metadata={"error": str(e)},
                clarifying_questions=[
                    "I couldn't understand that request. Could you rephrase it?"
                ],
            )

    def _parse_llm_response(self, response: str) -> IntentResult:
        """Parse JSON response from LLM."""
        import json

        try:
            data = json.loads(response.strip())

            intent_str = data.get("intent_type", "unknown")
            try:
                intent_type = IntentType(intent_str)
            except ValueError:
                intent_type = IntentType.UNKNOWN

            return IntentResult(
                intent_type=intent_type,
                target=data.get("target"),
                confidence=float(data.get("confidence", 0.5)),
                metadata=data.get("metadata", {}),
                clarifying_questions=data.get("clarifying_questions", []),
            )

        except (json.JSONDecodeError, KeyError, ValueError):
            return IntentResult(
                intent_type=IntentType.UNKNOWN,
                confidence=0.0,
                metadata={"raw_response": response},
            )

    def _generate_clarifying_questions(
        self, request: str, result: IntentResult
    ) -> list[str]:
        """Generate clarifying questions based on ambiguous requests."""
        questions = []

        if not result.target:
            questions.append("Which file or module are you referring to?")

        if result.intent_type == IntentType.UNKNOWN:
            questions.append("What action would you like me to take?")
            questions.append(
                "Are you trying to fix something, add something, or understand code?"
            )

        return questions


async def classify_request(request: str) -> IntentResult:
    """
    Convenience function for one-shot intent classification.

    Args:
        request: User request to classify.

    Returns:
        IntentResult with classification.

    Example:
        >>> result = await classify_request("fix the auth bug")
        >>> print(result.intent_type, result.target)
        FIX_BUG auth
    """
    classifier = IntentClassifier()
    return await classifier.classify(request)
