import pytest
from unittest.mock import AsyncMock, MagicMock

from scout.llm.intent import IntentClassifier, IntentType, IntentResult


@pytest.mark.asyncio
async def test_intent_classifier_quick_match():
    """Test quick pattern matching for obvious intents."""
    classifier = IntentClassifier()
    result = await classifier.classify("fix the bug in auth")
    assert result.intent_type == IntentType.FIX_BUG
    assert result.target == "auth"
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_intent_classifier_feature():
    """Test feature implementation intent."""
    classifier = IntentClassifier()
    result = await classifier.classify("add oauth support")
    assert result.intent_type == IntentType.IMPLEMENT_FEATURE
    assert result.target == "auth"


@pytest.mark.asyncio
async def test_intent_classifier_query():
    """Test code query intent."""
    classifier = IntentClassifier()
    result = await classifier.classify("what does this function do")
    assert result.intent_type == IntentType.QUERY_CODE
    assert result.confidence == 0.9


@pytest.mark.asyncio
async def test_intent_classifier_empty_request():
    """Test handling of empty request."""
    classifier = IntentClassifier()
    result = await classifier.classify("")
    assert result.intent_type == IntentType.UNKNOWN
    assert result.confidence == 0.0
    assert len(result.clarifying_questions) > 0


@pytest.mark.asyncio
async def test_intent_classifier_llm_fallback():
    """Test LLM classification when quick patterns don't match."""
    mock_response = MagicMock()
    mock_response.content = '{"intent_type": "optimize", "target": "user.py", "confidence": 0.88, "metadata": {}, "clarifying_questions": []}'
    mock_response.cost_usd = 0.001

    mock_llm = AsyncMock(return_value=mock_response)
    classifier = IntentClassifier(llm_call=mock_llm)

    result = await classifier.classify("make the database queries faster")

    assert result.intent_type == IntentType.OPTIMIZE
    assert result.target == "user.py"
    assert result.confidence == 0.88
    mock_llm.assert_called_once()


@pytest.mark.asyncio
async def test_intent_classifier_custom_llm():
    """Test classifier with custom LLM callable."""
    mock_response = MagicMock()
    mock_response.content = '{"intent_type": "test", "target": null, "confidence": 0.95, "metadata": {}, "clarifying_questions": []}'
    mock_response.cost_usd = 0.002

    custom_llm = AsyncMock(return_value=mock_response)
    classifier = IntentClassifier(llm_call=custom_llm)

    result = await classifier.classify("write unit tests for auth module")

    assert result.intent_type == IntentType.TEST
    assert result.confidence == 0.95
    custom_llm.assert_called_once()


@pytest.mark.asyncio
async def test_intent_classifier_unknown_response():
    """Test handling of unknown LLM response."""
    mock_response = MagicMock()
    mock_response.content = "not valid json"
    mock_response.cost_usd = 0.0

    mock_llm = AsyncMock(return_value=mock_response)
    classifier = IntentClassifier(llm_call=mock_llm)

    result = await classifier.classify("do something weird")

    assert result.intent_type == IntentType.UNKNOWN
    assert "raw_response" in result.metadata


def test_intent_type_enum():
    """Test IntentType enum values."""
    assert IntentType.FIX_BUG.value == "fix_bug"
    assert IntentType.IMPLEMENT_FEATURE.value == "implement_feature"
    assert IntentType.QUERY_CODE.value == "query_code"
