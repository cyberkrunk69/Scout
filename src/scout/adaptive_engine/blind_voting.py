"""
Blind Voting Engine for Browser Element Selection.

Extends BallotGate pattern from scout/adaptive_engine/gates.py.
Uses parallel 8B LLM calls for element disambiguation.

Cost Analysis (per Scout philosophy - cost is feature):
- 3 voters × 1000 input tokens × $0.00005 = $0.00015
- 3 voters × 200 output tokens × $0.00008 = $0.000048
- Total per iteration: ~$0.00020
- Max 3 iterations: ~$0.00060
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from scout.adaptive_engine.gates import (
    BallotGate,
    BallotVote,
    GateDecision,
    GateStage,
)
from scout.audit import AuditLog
from scout.llm.retry import call_with_retries

logger = logging.getLogger(__name__)


class VotingResult(str, Enum):
    """Voting outcome types."""

    MAJORITY = "majority"
    PLURALITY = "plurality"
    NO_CONSENSUS = "no_consensus"
    SKIPPED = "skipped"


@dataclass
class VotingCandidate:
    """Element candidate for voting."""

    element_id: str
    selector: str
    text: str
    tag: str
    score: float
    bbox: Optional[Dict[str, int]] = None


@dataclass
class VotingOutcome:
    """Result of voting process."""

    result: VotingResult
    winner: Optional[VotingCandidate]
    votes: List[BallotVote]
    iterations: int
    total_cost_usd: float
    confidence: float


class BlindVotingEngine:
    """
    Element selection via blind voting with 8B model.

    Architecture:
    1. Receive multiple candidates from BM25F/embedding
    2. Spawn M parallel 8B calls with same context
    3. Aggregate votes via majority/plurality
    4. If no consensus, iterate with refined context
    5. Escalate to larger model after max iterations

    Extends BallotGate from gates.py:
    - Reuses BallotVote, GateDecision types
    - Adds voting-specific parameters

    Execution Traces:
    1. Happy Path: First round majority achieved
    2. Failure Path: No consensus, iterate max times
    3. Edge Case: Single candidate, skip voting
    """

    # 8B model pricing (Groq)
    MODEL_COST_PER_1K_INPUT = 0.00005
    MODEL_COST_PER_1K_OUTPUT = 0.00008
    MODEL_NAME = "llama-3.1-8b-instant"

    def __init__(
        self,
        audit: AuditLog,
        voters: int = 3,
        threshold: float = 0.66,
        max_iterations: int = 3,
        max_cost_per_action: float = 0.01
    ):
        self.audit = audit
        self.voters = voters
        self.threshold = threshold  # 2/3 majority
        self.max_iterations = max_iterations
        self.max_cost_per_action = max_cost_per_action

        # Extend BallotGate for vote aggregation
        self.ballot_gate = BallotGate(
            approval_threshold=threshold,
            min_votes=voters,
            stage=GateStage.INTEGRATION
        )

        self.total_cost = 0.0

    async def select_element(
        self,
        goal: str,
        candidates: List[VotingCandidate],
        context: Optional[Dict[str, Any]] = None
    ) -> VotingOutcome:
        """
        Run blind voting to select best element for goal.

        Args:
            goal: Natural language goal
            candidates: List of candidate elements
            context: Additional context (page title, url, etc.)

        Returns:
            VotingOutcome with selected element and vote details

        Execution Traces:
        - Happy: Majority achieved in first round
        - Failure: No consensus, escalate after max iterations
        - Edge: Single candidate, skip voting
        """
        # Edge case: single candidate
        if len(candidates) == 1:
            self.audit.log("voting_skip", reason="single_candidate")
            return VotingOutcome(
                result=VotingResult.SKIPPED,
                winner=candidates[0],
                votes=[],
                iterations=0,
                total_cost_usd=0.0,
                confidence=1.0
            )

        # Edge case: no candidates
        if len(candidates) == 0:
            self.audit.log("voting_skip", reason="no_candidates")
            raise ValueError("No candidates to vote on")

        # Check budget
        if self.total_cost >= self.max_cost_per_action:
            self.audit.log(
                "voting_skip",
                reason="budget_exhausted",
                current_cost=self.total_cost
            )
            return VotingOutcome(
                result=VotingResult.SKIPPED,
                winner=candidates[0],
                votes=[],
                iterations=0,
                total_cost_usd=self.total_cost,
                confidence=0.5
            )

        # Run voting iterations
        for iteration in range(1, self.max_iterations + 1):
            self.audit.log("voting_iteration", iteration=iteration)

            outcome = await self._run_voting_round(
                goal, candidates, context, iteration
            )

            if outcome.result == VotingResult.MAJORITY:
                return outcome

        # No consensus after max iterations - use plurality or first
        self.audit.log(
            "voting_no_consensus",
            iterations=self.max_iterations,
            total_cost=self.total_cost
        )

        # Use first candidate as fallback
        return VotingOutcome(
            result=VotingResult.NO_CONSENSUS,
            winner=candidates[0],
            votes=[],
            iterations=self.max_iterations,
            total_cost_usd=self.total_cost,
            confidence=0.3
        )

    async def _run_voting_round(
        self,
        goal: str,
        candidates: List[VotingCandidate],
        context: Optional[Dict[str, Any]],
        iteration: int
    ) -> VotingOutcome:
        """Execute one round of blind voting."""

        # Build prompt
        prompt = self._build_prompt(goal, candidates, context, iteration)

        # Spawn parallel LLM calls
        tasks = [
            self._call_voter(i, prompt)
            for i in range(self.voters)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate votes
        votes = []
        vote_counts: Dict[str, int] = {}

        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.warning(f"Voter {i} failed: {response}")
                continue

            selected_id = response.get("selected_id")
            confidence = response.get("confidence", 50)
            reasoning = response.get("reasoning", "")

            votes.append(BallotVote(
                voter=f"8b_voter_{i}",
                decision=selected_id,
                reason=reasoning
            ))

            if selected_id:
                vote_counts[selected_id] = vote_counts.get(selected_id, 0) + 1

        # Determine result
        winner_id = None
        max_votes = 0

        for element_id, count in vote_counts.items():
            if count > max_votes:
                max_votes = count
                winner_id = element_id

        # Check for majority
        if len(votes) > 0:
            majority_ratio = max_votes / len(votes)
        else:
            majority_ratio = 0

        if majority_ratio >= self.threshold:
            result = VotingResult.MAJORITY
        elif max_votes > 1:
            result = VotingResult.PLURALITY
        else:
            result = VotingResult.NO_CONSENSUS

        # Find winner candidate
        winner = next(
            (c for c in candidates if c.element_id == winner_id),
            candidates[0] if candidates else None
        )

        # Calculate confidence
        confidence = majority_ratio if result == VotingResult.MAJORITY else max_votes / max(len(votes), 1)

        # Audit
        self.audit.log(
            "voting_round",
            iteration=iteration,
            result=result.value,
            winner_id=winner_id,
            vote_counts=vote_counts,
            cost_usd=self.total_cost
        )

        return VotingOutcome(
            result=result,
            winner=winner,
            votes=votes,
            iterations=iteration,
            total_cost_usd=self.total_cost,
            confidence=confidence
        )

    async def _call_voter(
        self,
        voter_id: int,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Single voter LLM call with retry resilience.

        Execution Traces:
        - Happy: LLM returns valid JSON with selected_id
        - Failure: JSON parse error or LLM timeout → returns fallback
        - Edge: All retries exhausted → returns fallback with error reasoning
        """
        from scout.llm.router import call_llm

        try:
            # Wrap with retries for transient failures
            response = await call_with_retries(
                call_llm,
                prompt=prompt,
                model=self.MODEL_NAME,
                temperature=0.3 + (voter_id * 0.1),
                max_tokens=200,
                max_retries=2,
                context=None
            )

            # Track cost using direct attribute access (LLMResult is dataclass)
            self.total_cost += response.cost_usd

            # Parse JSON response - handle markdown code blocks
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())

        except Exception as e:
            logger.warning(f"Voter {voter_id} error: {e}")
            return {
                "selected_id": None,
                "confidence": 0,
                "reasoning": f"Error: {e}"
            }

    def _build_prompt(
        self,
        goal: str,
        candidates: List[VotingCandidate],
        context: Optional[Dict[str, Any]],
        iteration: int
    ) -> str:
        """Build voting prompt."""

        candidate_text = "\n".join([
            f"[{c.element_id}] <{c.tag}> \"{c.text[:50]}\" (BM25F: {c.score:.2f})"
            for c in candidates[:10]
        ])

        context_text = ""
        if context:
            if context.get("page_title"):
                context_text += f"\nPage: {context['page_title']}"
            if context.get("url"):
                context_text += f"\nURL: {context['url']}"

        iteration_hint = ""
        if iteration > 1:
            iteration_hint = "\nNote: Previous voting round was inconclusive. Be more decisive."

        return f"""You are selecting the best element for a user's goal.

GOAL: {goal}
{context_text}
{iteration_hint}

CANDIDATES:
{candidate_text}

Respond in JSON format ONLY:
{{"selected_id": "<element_id>", "confidence": <0-100>, "reasoning": "<brief explanation>"}}

Select the SINGLE best element. Be decisive. Do not add any text outside the JSON."""
