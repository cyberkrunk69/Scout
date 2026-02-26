"""
Scout Big Brain — MiniMax API for natural language interpretation and synthesis.

Used for: query interpretation, PR descriptions, commit messages, module-level analysis.
Supports MiniMax (MiniMax-M2.5, MiniMax-M2.1-highspeed). Requires MINIMAX_API_KEY.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from scout.audit import AuditLog
from scout.middle_manager import GateDecision, MiddleManagerGate

logger = logging.getLogger(__name__)

# Primary big brain model: reasoning, analysis, synthesis
BIG_BRAIN_MODEL = "MiniMax-M2.5"
# Fallback when rate-limited or unavailable
BIG_BRAIN_FALLBACK = "MiniMax-M2.1-highspeed"

# TICKET-19: Gate-approved briefs → M2.1 (cheap); escalate → M2.5 (expensive)
MINIMAX_MODEL_FLASH = "MiniMax-M2.1-highspeed"
MINIMAX_MODEL_PRO = "MiniMax-M2.5"


@dataclass
class BigBrainResponse:
    """Response from big brain API."""

    content: str
    cost_usd: float
    model: str
    input_tokens: int
    output_tokens: int


def _estimate_minimax_cost(
    model_id: str, input_tokens: int, output_tokens: int
) -> float:
    """Use real pricing from llm.pricing."""
    from scout.llm.pricing import estimate_cost_usd

    return estimate_cost_usd(model_id, input_tokens, output_tokens)


async def call_big_brain_async(
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: int = 2048,
    model: Optional[str] = None,
    task_type: str = "general",
    big_brain_client: Optional[Callable] = None,
) -> BigBrainResponse:
    """
    Call MiniMax API for big brain tasks (query, PR synthesis, commit, analysis).

    Uses MINIMAX_API_KEY. Logs to audit as "big_brain_{task_type}".
    """
    if big_brain_client:
        return await big_brain_client(
            prompt,
            system=system,
            max_tokens=max_tokens,
            model=model,
            task_type=task_type,
        )

    from scout.llm.minimax import call_minimax_async_detailed

    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MINIMAX_API_KEY missing. Set it in .env or environment for big brain tasks."
        )

    model_used = model or BIG_BRAIN_MODEL

    try:
        result = await call_minimax_async_detailed(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            model=model_used,
        )
        return BigBrainResponse(
            content=result.response_text,
            cost_usd=result.cost_usd,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            model=result.model,
        )
    except Exception as e:
        if "MiniMax-M2.5" in model_used:
            logger.warning("Big brain %s failed, trying fallback: %s", model_used, e)
            model_used = BIG_BRAIN_FALLBACK
            try:
                result = await call_minimax_async_detailed(
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tokens,
                    model=model_used,
                )
                return BigBrainResponse(
                    content=result.response_text,
                    cost_usd=result.cost_usd,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    model=result.model,
                )
            except Exception as e2:
                raise RuntimeError(f"Big brain failed: {e2}") from e2
        else:
            raise

    cost = _estimate_gemini_cost(model_used, input_t, output_t)
    if cost == 0.0 and text:
        cost = 1e-7

    audit = AuditLog()
    audit.log(
        f"big_brain_{task_type}",
        cost=cost,
        model=model_used,
        input_t=input_t,
        output_t=output_t,
    )

    return BigBrainResponse(
        content=text,
        cost_usd=cost,
        model=model_used,
        input_tokens=input_t,
        output_tokens=output_t,
    )


async def call_big_brain_gated_async(
    question: str,
    facts: Optional[Any] = None,
    *,
    raw_tldr_context: Optional[str] = None,
    deps_graph: Optional[Any] = None,
    query_symbols: Optional[list] = None,
    task_type: str = "synthesis",
    model: Optional[str] = None,
    model_escalate: Optional[str] = None,
    big_brain_client: Optional[Callable] = None,
    gate: Optional[MiddleManagerGate] = None,
    on_decision: Optional[Callable[["GateDecision"], None]] = None,
    on_decision_async: Optional[Callable[["GateDecision"], Awaitable[None]]] = None,
) -> BigBrainResponse:
    """
    Gated path: compress context via 70B, then call Gemini.

    TICKET-43: Prefer facts (ModuleFacts). raw_tldr_context for synthesis path.
    TICKET-19: Gate-approved briefs → Flash; escalate → Pro.
    """
    gate = gate or MiddleManagerGate(confidence_threshold=0.75)
    decision = await gate.validate_and_compress(
        question=question,
        facts=facts,
        raw_tldr_context=raw_tldr_context,
        deps_graph=deps_graph,
        query_symbols=query_symbols,
    )

    if on_decision:
        on_decision(decision)
    if on_decision_async:
        await on_decision_async(decision)

    # decision.content is compressed (pass) or raw (escalate)
    context_for_prompt = decision.content
    audit = getattr(gate, "_audit", None) or AuditLog()

    # TICKET-48d: Preserve gate-declared gaps in synthesis prompt
    gap_context = ""
    if decision.gaps:
        gap_context = "\n\n[GAP MARKERS FROM GATE]\n" + "\n".join(
            f"- {g}" for g in decision.gaps
        )

    prompt = f"""Context:
{context_for_prompt}
{gap_context}

---
Question: {question}

Answer from context. If gaps exist, acknowledge uncertainty where relevant."""

    # TICKET-19: ROUTING: M2.1 for pass, M2.5 for escalate
    if decision.decision == "pass":
        model_used = model or MINIMAX_MODEL_FLASH
        audit.log(
            "gate_synthesis",
            model="M2.1",
            confidence=int((decision.confidence or 0) * 100),
        )
    else:
        model_used = model_escalate or MINIMAX_MODEL_PRO
        audit.log(
            "gate_synthesis",
            model="M2.5",
            reason="escalate",
        )

    return await call_big_brain_async(
        prompt,
        system="You answer concisely based on the provided context.",
        max_tokens=1024,
        model=model_used,
        task_type=task_type,
        big_brain_client=big_brain_client,
    )


async def interpret_query_async(natural_language: str) -> dict[str, Any]:
    """
    Use big brain to interpret natural language into a scout query spec.

    Returns dict with: tool, scope, include_deep, copy_to_clipboard, task (for nav).
    Pattern-based routing for symbol lookups (nav) needs no API key.
    LLM fallback requires MINIMAX_API_KEY.
    """
    # Fast path: pattern match needs no API
    pattern_spec = _route_query_by_patterns(natural_language)
    if pattern_spec is not None:
        return pattern_spec
    if not os.environ.get("MINIMAX_API_KEY"):
        raise EnvironmentError(
            "MINIMAX_API_KEY required for query interpretation. Set it in .env"
        )
    return await _interpret_query_via_big_brain(natural_language)


def _route_query_by_patterns(natural_language: str) -> dict[str, Any] | None:
    """
    Fast pattern-based routing for symbol lookups vs doc queries.
    Returns spec dict or None if no pattern matches (fall through to LLM).
    """
    import re

    text = natural_language.strip()

    # Nav patterns: symbol lookup → scout-index (instant, free)
    nav_patterns = [
        (r"where\s+is\s+(.+?)\s+defined", r"\1"),
        (r"find\s+(?:the\s+)?(?:class|function)\s+(.+?)(?:\s|$)", r"\1"),
        (r"find\s+(.+?)\s+(?:class|function)", r"\1"),
        (r"show\s+me\s+(?:the\s+)?(.+?)(?:\s|$)", r"\1"),
        (r"locate\s+(?:the\s+)?(.+?)(?:\s|$)", r"\1"),
        (r"where\s+(?:do\s+i\s+)?(?:find|look\s+for)\s+(.+?)(?:\s|$)", r"\1"),
    ]
    for pat, repl in nav_patterns:
        m = re.search(pat, text, re.I)
        if m:
            task = m.expand(repl).strip()
            if task and len(task) > 1:
                return {
                    "tool": "nav",
                    "task": task.strip(),
                    "scope": "vivarium/scout",
                    "include_deep": False,
                    "copy_to_clipboard": True,
                }

    # Query patterns: documentation/explanations
    query_patterns = [
        r"explain\s+\w",
        r"tell\s+me\s+about\s+\w",
        r"what\s+is\s+\w",
        r"how\s+does\s+\w",
        r"documentation\s+for",
        r"docs?\s+for",
    ]
    for pat in query_patterns:
        if re.search(pat, text, re.I):
            return {
                "tool": "query",
                "scope": "vivarium/scout",
                "include_deep": True,
                "copy_to_clipboard": True,
            }

    return None


async def _interpret_query_via_big_brain(natural_language: str) -> dict[str, Any]:
    """Call big brain to interpret query. Raises on failure."""
    prompt = f"""Interpret this natural language request into a structured scout query.

User request: "{natural_language}"

Symbol lookups ("where is X", "find class Y") → tool="nav" (scout-index).
Docs/explanations ("explain X", "tell me about X") → tool="query".

Respond with ONLY valid JSON, no markdown or explanation:
{{
  "tool": "nav" or "query",
  "scope": "vivarium/scout",
  "include_deep": true,
  "copy_to_clipboard": true,
  "task": "extracted symbol or topic (required when tool is nav)"
}}

- tool: "nav" for symbol lookups, "query" for docs
- scope: package path (e.g. vivarium/scout)
- include_deep: tldr vs deep (query only)
- copy_to_clipboard: copy results
- task: for nav, symbol to find (e.g. "TriggerRouter")"""

    response = await call_big_brain_async(
        prompt,
        system="Output only valid JSON. Symbol lookups → nav. Docs → query.",
        max_tokens=256,
        task_type="query_interpret",
    )
    raw = response.content.strip()
    # Strip markdown code blocks if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        spec = json.loads(raw)
        spec.setdefault("tool", "query")
        return spec
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Big brain returned invalid JSON: {raw[:200]!r}") from e


def _extract_json_from_content(raw: str) -> dict[str, Any] | None:
    """Extract first valid JSON object from content. Returns None if none found."""
    raw = raw.strip()
    if not raw:
        return None
    # Strip markdown code blocks
    if raw.startswith("```"):
        parts = raw.split("```", 2)
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
    # Find first { and matching }
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i, c in enumerate(raw[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _flatten_spec(obj: dict) -> dict:
    """Flatten nested params into top-level spec for runners."""
    spec = dict(obj)
    params = spec.pop("params", None)
    if isinstance(params, (dict)):
        for k, v in params.items():
            spec.setdefault(k, v)
    return spec


def parse_chat_response(raw: str) -> tuple[str, Any]:
    """
    Parse big brain response. Returns ("tool", spec) or ("message", text).
    Flexible: extracts JSON from content; if no valid tool, treats as message.
    Never returns raw JSON/partial output to user.
    """
    import re

    from scout.tools import get_valid_tool_names

    valid_tools = get_valid_tool_names()
    obj = _extract_json_from_content(raw)
    if obj is not None and isinstance(obj, (dict)):
        tool = obj.get("tool")
        if tool in valid_tools:
            return ("tool", _flatten_spec(obj))
        msg = obj.get("message")
        if msg is not None and isinstance(msg, (str)):
            return ("message", msg)
    # Salvage truncated/malformed JSON: extract tool name and scope if present
    text = raw.strip()
    if text and "tool" in text.lower():
        tm = re.search(r'"tool"\s*:\s*"(\w+)"', text, re.I)
        if tm and tm.group(1) in valid_tools:
            spec = {"tool": tm.group(1)}
            sm = re.search(r'"scope"\s*:\s*"([^"]+)"', text)
            if sm:
                spec["scope"] = sm.group(1)
            im = re.search(r'"include_deep"\s*:\s*(true|false)', text, re.I)
            if im:
                spec["include_deep"] = im.group(1).lower() == "true"
            return ("tool", spec)
    if (
        text
        and ("{" in text or "```json" in text.lower())
        and ("tool" in text or "params" in text)
    ):
        return ("message", None)
    return ("message", text if text else None)


def _has_tool_output(messages: list[dict]) -> bool:
    """True if conversation includes tool output (synthesis needed)."""
    for m in messages:
        if "[Tool " in str(m.get("content", "")):
            return True
    return False


def _extract_last_tool_output(messages: list[dict]) -> str:
    """TICKET-38: Extract content of most recent tool result."""
    import re

    for m in reversed(messages):
        content = m.get("content", "")
        if "[Tool " in content and "result]" in content:
            match = re.search(r"\[Tool \w+ result\]:\s*\n(.*)", content, re.DOTALL)
            if match and match.group(1).strip():
                return match.group(1).strip()
    return ""


def _extract_last_user_message(messages: list[dict]) -> str:
    """Last user message content."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _truncate_string_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate string to fit token cap (~4 chars/token)."""
    if not text:
        return ""
    if len(text) // 4 <= max_tokens:
        return text
    return text[: max_tokens * 4]


async def chat_turn_async(
    messages: list[dict[str, str]],
    cwd_scope: str = "vivarium",
    repo_state: dict[str, Any] | None = None,
    caveman: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> str:
    """
    Chat turn. Uses Groq for easy work (tool selection, routing). Uses big brain
    (Gemini) only for cognitive tasks (synthesizing answer from tool output).
    """
    from scout.tools import get_tools_minimal

    tools_minimal = get_tools_minimal()
    tools_lines = [f"- {t['name']}: {t['desc']}" for t in tools_minimal]
    tools_block = "\n".join(tools_lines)
    state = repo_state or {}
    state.setdefault("cwd_scope", cwd_scope)
    state_json = json.dumps(state, indent=2)

    conv_lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        label = "User" if role == "user" else "Scout"
        conv_lines.append(f"{label}: {content}")
    conv_block = "\n\n".join(conv_lines) if conv_lines else "(no prior messages)"

    # Routing: Groq has strict context limits. Chat history can be 1MB+.
    # For routing, use only the last user message so we stay under limits.
    needs_synthesis = _has_tool_output(messages)
    if not needs_synthesis:
        last_user = next(
            (
                m.get("content", "")
                for m in reversed(messages)
                if m.get("role") == "user"
            ),
            "",
        )
        conv_block = f"User: {last_user}" if last_user else "(no message)"

    caveman_rule = " Use eliv. Small words. " if caveman else ""
    prompt = f"""You are Scout. You help with code, docs, and dev workflow.

Context: {state_json}

Tools (pick one):
{tools_block}

Conversation:
{conv_block}

Rules: Pick a tool when user asks for info, docs, tldr, or how it works.
- Symbol lookups (where is X, find class Y) → nav (scout-index, free).
- Docs (explain X, tell me about X) → query.
Use message only for greetings. Never greet when user asked a question.

Respond: {{"tool": "name", "scope": "...", "include_deep": bool, "task": "...", ...}}
Or {{"message": "..."}} for greetings only."""

    system = (
        "You are Scout. Use tools. Symbol lookups → nav. Docs → query. "
        "tldr/deep for query; nav uses index first." + caveman_rule
    )

    # Synthesis: TICKET-38 — Gate synthesis path. Bounded 4K context.
    if needs_synthesis:
        if progress_cb:
            progress_cb("Gating synthesis...")
        try:
            last_tool = _extract_last_tool_output(messages)
            last_user = _extract_last_user_message(messages)
            synthesis_context = (
                f"User: {last_user}\n\nTool output:\n{last_tool}"
                if last_user or last_tool
                else last_tool or "(no context)"
            )
            synthesis_context = _truncate_string_to_tokens(synthesis_context, 4000)

            synthesis_query = (
                "Summarize the tool output above for the user. Be concise. "
                "Acknowledge [GAP] markers when present."
            )

            on_decision_async = None
            try:
                from scout.config import ScoutConfig
                from scout.ui.whimsy import (
                    decision_to_whimsy_params,
                    generate_gate_whimsy,
                )

                if ScoutConfig().whimsy_mode:

                    async def _print_synthesis_whimsy(d: GateDecision) -> None:
                        cost = getattr(d, "cost_usd", 0) or (
                            0.05 if d.decision == "pass" else 0.50
                        )
                        params = decision_to_whimsy_params(d, cost)
                        line = await generate_gate_whimsy(**params)
                        print(line, file=sys.stderr)

                    on_decision_async = _print_synthesis_whimsy
            except ImportError:
                pass

            response = await call_big_brain_gated_async(
                question=synthesis_query,
                raw_tldr_context=synthesis_context,
                task_type="synthesis",
                on_decision_async=on_decision_async,
            )
            return response.content.strip()
        except Exception as e:
            logger.debug("Synthesis failed: %s", e)
            msg = str(e).split("\n")[0] if str(e) else "unknown"
            raise RuntimeError(f"Groq synthesis failed: {msg}") from e

    # Research intent detection: check if query needs autonomous research
    # Extract last user message for intent detection
    last_user_for_research = next(
        (
            m.get("content", "")
            for m in reversed(messages)
            if m.get("role") == "user"
        ),
        "",
    )

    if last_user_for_research:
        from scout.research import detect_research_intent, should_use_llm_fallback, llm_decide_research
        from scout.config import ScoutConfig

        is_research, confidence = detect_research_intent(last_user_for_research)
        config = ScoutConfig()
        research_config = config._raw.get("research", {})
        intent_threshold = research_config.get("intent_threshold", 0.7)
        llm_fallback_threshold = research_config.get("llm_fallback_threshold", 0.5)

        if is_research and confidence >= intent_threshold:
            # Route to research tool (high confidence)
            if progress_cb:
                progress_cb(f"Routing to research (confidence: {confidence:.2f})...")

            # Import and execute research
            from scout.research.command import execute_research

            try:
                research_result = await execute_research(last_user_for_research)
                audit = AuditLog()
                audit.log(
                    "research_intent_detected",
                    query=last_user_for_research,
                    confidence=confidence,
                    routed_to="research",
                    method="keyword",
                )
                return research_result
            except Exception as e:
                logger.debug("Research execution failed: %s", e)
                # Fall through to normal routing on research failure

        elif is_research and should_use_llm_fallback(confidence, llm_fallback_threshold):
            # LLM fallback for ambiguous queries
            if progress_cb:
                progress_cb(f"Using LLM fallback (confidence: {confidence:.2f})...")

            should_research = await llm_decide_research(last_user_for_research)

            if should_research:
                # Route to research
                if progress_cb:
                    progress_cb("LLM confirmed research needed, routing...")

                from scout.research.command import execute_research

                try:
                    research_result = await execute_research(last_user_for_research)
                    audit = AuditLog()
                    audit.log(
                        "research_intent_detected",
                        query=last_user_for_research,
                        confidence=confidence,
                        routed_to="research",
                        method="llm_fallback",
                    )
                    return research_result
                except Exception as e:
                    logger.debug("Research execution failed: %s", e)

    # Improvement intent detection: check if query needs code improvement
    if last_user_for_research:
        from scout.research import (
            detect_improvement_intent,
            llm_decide_improvement,
        )

        is_improvement, imp_confidence = detect_improvement_intent(
            last_user_for_research
        )

        if is_improvement:
            config = ScoutConfig()
            improvement_config = config._raw.get("improvement", {})
            imp_intent_threshold = improvement_config.get("intent_threshold", 0.7)
            imp_llm_fallback_threshold = improvement_config.get(
                "llm_fallback_threshold", 0.5
            )

            if imp_confidence >= imp_intent_threshold:
                # Route to code improvement (high confidence)
                if progress_cb:
                    progress_cb(
                        f"Routing to code improvement (confidence: {imp_confidence:.2f})..."
                    )

                from scout.research.command import execute_code_analysis

                try:
                    improvement_result = await execute_code_analysis(
                        last_user_for_research, messages
                    )
                    audit = AuditLog()
                    audit.log(
                        "improvement_intent_detected",
                        query=last_user_for_research,
                        confidence=imp_confidence,
                        routed_to="improvement",
                        method="keyword",
                    )
                    return improvement_result
                except Exception as e:
                    logger.debug("Code improvement execution failed: %s", e)

            elif 0.3 <= imp_confidence < imp_llm_fallback_threshold:
                # LLM fallback for ambiguous improvement queries
                if progress_cb:
                    progress_cb(
                        f"Using LLM fallback for improvement (confidence: {imp_confidence:.2f})..."
                    )

                should_improve = await llm_decide_improvement(last_user_for_research)

                if should_improve:
                    if progress_cb:
                        progress_cb(
                            "LLM confirmed improvement needed, routing..."
                        )

                    from scout.research.command import execute_code_analysis

                    try:
                        improvement_result = await execute_code_analysis(
                            last_user_for_research, messages
                        )
                        audit = AuditLog()
                        audit.log(
                            "improvement_intent_detected",
                            query=last_user_for_research,
                            confidence=imp_confidence,
                            routed_to="improvement",
                            method="llm_fallback",
                        )
                        return improvement_result
                    except Exception as e:
                        logger.debug("Code improvement execution failed: %s", e)

    # Routing (no tool output yet): Groq only
    if progress_cb:
        progress_cb("Calling LLM to pick tool...")
    from scout.llm.router import call_llm
    try:
        result = await call_llm(
            prompt,
            task_type="simple",
            model="llama-3.1-8b-instant",
            system=system,
            max_tokens=1024,
        )
        audit = AuditLog()
        audit.log(
            "chat_llm",
            cost=result.cost_usd,
            model=result.model,
            provider=result.provider,
            input_t=result.input_tokens,
            output_t=result.output_tokens,
        )
        return result.content.strip()
    except Exception as e:
        logger.debug("Groq routing failed: %s", e)
        msg = str(e).split("\n")[0] if str(e) else "unknown"
        raise RuntimeError(f"Groq routing failed: {msg}") from e



async def interpret_command_async(
    natural_language: str,
    cwd_scope: str = "vivarium",
) -> dict[str, Any]:
    """
    Interpret natural language into a scout tool call.
    Big brain picks cheapest tool (index=free, query=read-only, nav=index-or-LLM).
    Requires MINIMAX_API_KEY. Raises when big brain unavailable.
    """
    if not os.environ.get("MINIMAX_API_KEY"):
        raise EnvironmentError("MINIMAX_API_KEY required. Set it in .env")
    return await _interpret_command_via_big_brain(natural_language, cwd_scope)


async def _interpret_command_via_big_brain(
    natural_language: str,
    cwd_scope: str,
) -> dict[str, Any]:
    """Call big brain to pick the right tool. Tools passed as data."""
    from scout.tools import get_tools, get_valid_tool_names

    tools_json = json.dumps(get_tools(), indent=2)
    valid_names = sorted(get_valid_tool_names())
    prompt = f"""Pick the best scout tool. Use cheapest/fastest that satisfies user.

User request: "{natural_language}"
Context: scope "{cwd_scope}"

Symbol lookups (where is X, find class Y) → nav (scout-index, free).
Docs (explain X, tell me about X) → query.

Available tools (JSON):
{tools_json}

Respond with ONLY valid JSON. Set "tool" to one of: {valid_names}.
Include params: scope, include_deep, copy_to_clipboard, changed_only, task, query.
Use null when inapplicable. Prefer cheaper/faster tools."""

    response = await call_big_brain_async(
        prompt,
        system="Output valid JSON. Symbol lookups → nav. Docs → query. Pick cheapest.",
        max_tokens=256,
        task_type="command_interpret",
    )
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        spec = json.loads(raw)
        valid = get_valid_tool_names()
        tool = spec.get("tool")
        if tool is None or tool not in valid:
            raise RuntimeError(
                f"Big brain must return valid tool. Got {tool!r}. "
                f"Valid: {sorted(valid)}"
            )
        spec.setdefault("scope", cwd_scope)
        spec.setdefault("include_deep", False)
        spec.setdefault("copy_to_clipboard", True)
        spec.setdefault("changed_only", False)
        return spec
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Big brain returned invalid JSON: {raw[:200]!r}") from e


async def answer_help_async(
    repo_state: dict[str, Any],
    *,
    use_gate: bool = True,
    query: str = "",
) -> str:
    """
    User asked "what can you do". Big brain lists capabilities from tools data,
    suggests one based on repo state, you be the judge.

    When use_gate=True (default), context is compressed via MiddleManagerGate
    before Gemini. Escalation uses raw TLDRs with Pro model.
    """
    from scout.tools import get_tools

    tools_json = json.dumps(get_tools(), indent=2)
    state_json = json.dumps(repo_state, indent=2)
    caveman = repo_state.get("caveman_mode", False)
    caveman_rule = (
        " CAVEMAN MODE: eliv only. Small words. Cave man understand. "
        if caveman
        else " Use eliv when you talk to user. "
    )
    caveman_examples = (
        """
Caveman style examples:
- User: "you no hardcode string! bad robot. you use big brain for that."
- User: "make so cave man can use. small word, easy meaning."
- Scout: "me do: find code, read doc, plan, status. 3 file stage. me say try sync."
"""
        if caveman
        else ""
    )

    raw_context = f"""Available tools (JSON, each has desc and eliv):
{tools_json}

Repo state:
{state_json}
{caveman_rule}
{caveman_examples}"""

    question = "What can Scout do? List capabilities and suggest one from repo state."

    if use_gate:
        # TICKET-27/29: 8B gate whimsy when SCOUT_WHIMSY=1; else legacy phrase banks
        on_decision = None
        on_decision_async = None
        try:
            from scout.config import ScoutConfig
            from scout.ui.whimsy import (
                WhimsyFormatter,
                generate_gate_whimsy,
                decision_to_whimsy_params,
            )

            if ScoutConfig().whimsy_mode:
                # TICKET-27: Fresh 8B whimsy per gate decision
                async def _print_whimsy_8b(d: GateDecision) -> None:
                    cost = getattr(d, "cost_usd", 0) or (
                        0.05 if d.decision == "pass" else 0.50
                    )
                    params = decision_to_whimsy_params(d, cost)
                    line = await generate_gate_whimsy(**params)
                    print(line, file=sys.stderr)

                on_decision_async = _print_whimsy_8b
            else:

                def _print_whimsy_legacy(d: GateDecision) -> None:
                    print(
                        WhimsyFormatter.format_gate_decision(d, query=query),
                        file=sys.stderr,
                    )

                on_decision = _print_whimsy_legacy
        except ImportError:
            pass

        response = await call_big_brain_gated_async(
            question=question,
            raw_tldr_context=raw_context,
            task_type="help",
            model_escalate=BIG_BRAIN_MODEL,
            on_decision=on_decision,
            on_decision_async=on_decision_async,
        )
    else:
        prompt = f"""The user asked what scout can do.

{raw_context}
List what you do. Suggest one from repo state. You be the judge. Plain text."""
        system = (
            "You are scout. Caveman mode: small words only."
            if caveman
            else "You are scout. Use eliv. Suggest from repo state. Empower user."
        )
        response = await call_big_brain_async(
            prompt,
            system=system,
            max_tokens=1024,
            task_type="help",
        )
    return response.content.strip()


# =============================================================================
# Autonomous Planning - Generate execution plans with reasoning
# =============================================================================


async def generate_autonomous_plan(
    request: str,
    context: Optional[dict[str, Any]] = None,
    tools: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Generate an autonomous execution plan for a complex task.

    This function takes a natural language request and uses the big brain
    to determine which tools to call, in what order, with what arguments.
    Each step includes a 'reasoning' field explaining why that tool was chosen.

    Args:
        request: The natural language task description
        context: Optional context about the repo state (cwd, files, etc.)
        tools: Optional list of available tools. If not provided, will fetch from tools registry.

    Returns:
        dict with keys:
        - plan: list of steps, each with command, args, depends_on, and reasoning
        - cost_usd: estimated cost
        - model: model used
        - reasoning: overall explanation of the plan
    """
    from scout.tools import get_tools

    if tools is None:
        tools = get_tools()

    tools_json = json.dumps(tools, indent=2)

    context_info = ""
    if context:
        context_info = f"""
Current context:
- Working directory: {context.get('cwd', 'unknown')}
- Files in scope: {', '.join(context.get('files', [])[:10])}
- Repo root: {context.get('repo_root', 'unknown')}
"""

    prompt = f"""You are Scout's autonomous planning engine. Given a user request,
determine the optimal sequence of tool calls to accomplish the task.

Available tools:
{tools_json}
{context_info}

User request: "{request}"

Generate a JSON plan with the following format:
{{
  "plan": [
    {{
      "command": "tool_name",
      "args": {{"param1": "value1"}},
      "depends_on": [],
      "reasoning": "Why this tool was chosen and what it accomplishes"
    }}
  ],
  "reasoning": "Overall explanation of the plan strategy"
}}

CRITICAL INSTRUCTIONS:
1. If the user asks to "execute" or "run it", generate a single-step plan:
   {{"command": "execute", "args": {{"plan_path": ".scout/plans/last_plan.md"}}, "reasoning": "Executing the most recently generated plan as requested."}}

2. If the user asks to "write a doc", "explain for my mom", "explain for mom", or similar,
   prioritize calling the scout_generate_docs tool with a "simplified" or "high-level" persona argument.

Rules:
1. Each step must have: command, args, depends_on (array of indices), reasoning
2. Use depends_on to specify execution order - step 0 runs first, step 1 can depend on step 0
3. reasoning should explain WHY this tool was chosen
4. Use scout_batch-compatible commands: nav, query, doc_sync, lint, run, plan, git_status, etc.
5. Start with exploratory tools (nav, query) before making changes (edit, create)
6. Keep the plan focused - maximum 5 steps

Respond with ONLY valid JSON, no markdown or explanation."""

    response = await call_big_brain_async(
        prompt,
        system="You are Scout's autonomous planning engine. Output valid JSON only.",
        max_tokens=1024,
        task_type="autonomous_plan",
    )

    raw = response.content.strip()
    # Strip markdown code blocks if present
    if raw.startswith("```"):
        parts = raw.split("```", 2)
        if len(parts) >= 2:
            raw = parts[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

    try:
        plan_data = json.loads(raw)
        # Ensure required fields
        plan_data.setdefault("plan", [])
        plan_data.setdefault("reasoning", "")
        for step in plan_data["plan"]:
            step.setdefault("depends_on", [])
            step.setdefault("reasoning", "")
        return {
            "plan": plan_data["plan"],
            "reasoning": plan_data["reasoning"],
            "cost_usd": response.cost_usd,
            "model": response.model,
            # Include JSON block for machine execution (V12 Whimsy Engine)
            "json_block": f"\n```json\n{json.dumps(plan_data['plan'], indent=2)}\n```\n",
        }
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Big brain returned invalid JSON: {raw[:200]!r}") from e
