from __future__ import annotations
#!/usr/bin/env python
"""
Scout Plan CLI – Generate implementation plans using MiniMax.

Uses LLM-native approach: always provide relevant context and let the LLM
determine what files/functions are relevant, rather than hardcoded keyword matching.

Supports recursive sub-planning with parallel execution for complex requests.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

from scout.audit import AuditLog
from scout.llm.minimax import call_minimax_async
from scout.llm.dispatch import call_llm_async, get_provider_info
from scout.utils.summarize import slugify

# Import Scout tools for free verification
from scout.cli.index import query_for_nav

# Setup audit logging - use existing AuditLog infrastructure
from scout.audit import AuditLog

# Import Adaptive Engine components
from scout.adaptive_engine import (
    PlanningContext,
    TriggerRegistry,
    create_default_registry,
    log_pivot_outcome,
    compute_optimal_threshold,
    PIVOT_TRIGGERS,  # Keep for backward compatibility
    DEFAULT_PIVOT_THRESHOLD,
)

# Create default trigger registry for backward compatibility
_trigger_registry = create_default_registry()

# Create module-level audit instance for convenience
_audit = AuditLog()

# === TOOL ORCHESTRATION LAYER ===
# The planner doesn't DO everything - it DELEGATES to other tools

import subprocess


class ToolOrchestrator:
    """
    Orchestrates Scout tools to gather context efficiently.

    Instead of the planner doing everything, it delegates to specialized tools:
    - scout_query: Semantic search (free)
    - scout_grep: Pattern search (free)
    - scout_nav: Code navigation (free)
    - scout_index: Index queries (free)

    This is "tools together, strong" - composing capabilities.
    """

    def __init__(self, repo_root: Path, progress=None):
        self.repo_root = repo_root
        self.progress = progress

    def run_scout_tool(self, tool_name: str, args: list) -> str:
        """
        Run a Scout CLI tool and return its output.

        These tools are FREE - they use local index or grep, not LLM.
        """
        # Build command: python -m scout.cli.<tool> <args>
        cmd = ["python3", "-m", f"scout.cli.{tool_name}"] + args

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=30,  # 30s timeout per tool
            )
            if result.returncode == 0:
                return result.stdout[:2000]  # Limit output
            else:
                return f"[{tool_name} error]: {result.stderr[:500]}"
        except subprocess.TimeoutExpired:
            return f"[{tool_name} timeout]"
        except Exception as e:
            return f"[{tool_name} exception]: {str(e)[:200]}"

    def research_codebase(self, query: str) -> dict:
        """
        Research the codebase using multiple tools in parallel.

        This replaces expensive LLM context gathering with FREE tool calls.
        """
        results = {"semantic": "", "patterns": "", "navigation": "", "index": []}

        if self.progress:
            self.progress.spin(f"Researching: {query[:30]}...")

        # 1. Semantic search - understand what code is relevant
        results["semantic"] = self.run_scout_tool("query", [query])

        # 2. Pattern search - find specific patterns
        # Extract key terms from query
        keywords = query.replace("?", " ").replace("'", " ").split()
        significant = [w for w in keywords if len(w) > 4][:3]
        if significant:
            pattern_results = []
            for kw in significant[:2]:
                out = self.run_scout_tool("grep", [kw, "--json", "-l"])
                if out:
                    pattern_results.append(out[:500])
            results["patterns"] = "\n".join(pattern_results)

        # 3. Navigation - get file/function context
        results["navigation"] = self.run_scout_tool("nav", [query])

        # 4. Index - find related code
        results["index"] = query_for_nav(self.repo_root, query, limit=5)

        return results

    def verify_code_elements(self, plan_text: str) -> dict:
        """
        Verify code elements mentioned in plan using FREE tools.

        Replaces expensive LLM verification with fast tool calls.
        """
        import re

        verification = {"verified": [], "missing": [], "uncertain": []}

        # Extract function names from plan
        func_pattern = re.findall(r"`(\w+)\(`|def (\w+)\(", plan_text)
        functions = list(set([f[0] or f[1] for f in func_pattern if f[0] or f[1]]))[:5]

        for func in functions:
            if self.progress:
                self.progress.spin(f"Verifying: {func}...")

            # Use function_info - this is FREE
            info = self.run_scout_tool("function_info", [func])

            if info and "not found" not in info.lower():
                verification["verified"].append(f"✓ {func}")
            elif info:
                verification["uncertain"].append(f"? {func}: {info[:100]}")
            else:
                verification["missing"].append(f"✗ {func}")

        return verification


# Helper function to use orchestrator
def orchestrate_research(query: str, repo_root: Path, progress=None) -> str:
    """
    Orchestrate multiple Scout tools to research a query.

    This is FREE - uses local tools, no LLM cost.
    """
    orchestrator = ToolOrchestrator(repo_root, progress)
    results = orchestrator.research_codebase(query)

    # Combine results into context
    context = f"""
## Scout Tool Research Results for: {query}

### Semantic Search (scout_query)
{results.get('semantic', '')[:500]}

### Pattern Matches (scout_grep)
{results.get('patterns', '')[:500]}

### Navigation (scout_nav)
{results.get('navigation', '')[:500]}

### Index Results
"""
    for idx, r in enumerate((results.get("index") or [])[:3], 1):
        target = f"{r.get('target_file')}::{r.get('target_function')}"
        context += f"{idx}. {target} ({r.get('confidence')})"

    return context


# === COMPOSITE TOOL PATTERNS ===
# Pre-built pipelines that combine tools for common workflows

COMPOSITE_PATTERNS = {
    """
    scout-review: Code review pipeline
    Pipeline: git diff → roast → query → summarize

    Usage: Review changes in PR with automated review
    Tools: scout_git_diff, scout_roast, scout_query
    Cost: ~$0.001 (roast uses LLM, rest is free)
    """,
    """
    scout-refactor: Multi-file refactoring
    Pipeline: grep (find patterns) → nav (understand context) → edit

    Usage: Find and replace patterns across codebase
    Tools: scout_grep, scout_nav, scout_edit
    Cost: FREE (no LLM needed)
    """,
    """
    scout-test-gen: Test generation
    Pipeline: nav (understand code) → query (find related) → generate

    Usage: Generate tests for a function or module
    Tools: scout_nav, scout_query, LLM for generation
    Cost: ~$0.005 (generation needs LLM)
    """,
    """
    scout-explain: Comprehensive code explanation
    Pipeline: function_info → nav → query → summarize

    Usage: Understand what a piece of code does
    Tools: scout_function_info, scout_nav, scout_query
    Cost: FREE (all local tools)
    """,
    """
    scout-audit: Security/best-practice audit
    Pipeline: grep (find patterns) → audit (analyze) → query (explain)

    Usage: Audit codebase for security issues or best practices
    Tools: scout_grep, scout_audit, scout_query
    Cost: ~$0.01 (audit may use LLM)
    """,
    """
    scout-debug: Debugging workflow
    Pipeline: git_log → nav (find error) → query (explain) → plan (fix)

    Usage: Debug an error or issue
    Tools: scout_git_log, scout_nav, scout_query, scout_plan
    Cost: ~$0.01 (plan uses LLM)
    """,
    """
    scout-migrate: Migration assistance
Pipeline: grep (find old patterns) → nav (understand usage) → generate(new patterns)

    Usage: Migrate from one library/pattern to another
    Tools: scout_grep, scout_nav, LLM for generation
    Cost: ~$0.02 (needs LLM for generation)
    """,
    """
    scout-improve: Autonomous code improvement pipeline
    Pipeline: hotspot → analyze → plan → apply

    Usage: Identify hotspots, analyze issues, generate improvement plan, optionally apply safe changes
    Tools: scout_hotspots (stub), scout_roast, scout_plan, scout_edit
    Cost: ~$0.05 (uses LLM for analysis and planning)
    """,
}


def run_composite_pattern(pattern_name: str, args: dict, repo_root: Path) -> str:
    """
    Run a pre-built composite tool pattern.

    This demonstrates "tools together, strong" - combining capabilities
    for powerful workflows.
    """
    orchestrator = ToolOrchestrator(repo_root)

    if pattern_name == "explain":
        # Explain code - completely FREE
        target = args.get("target", "")
        info = orchestrator.run_scout_tool("function_info", [target])
        nav = orchestrator.run_scout_tool("nav", [target])
        query = orchestrator.run_scout_tool("query", [f"explain {target}"])

        return f"""
## Code Explanation for: {target}

### Function Signature
{info[:500]}

### Navigation Context
{nav[:500]}

### Semantic Understanding
{query[:500]}
"""

    elif pattern_name == "audit":
        # Security audit - mostly free
        patterns = args.get("patterns", ["TODO", "FIXME", "password", "api_key"])
        audit_results = []
        for p in patterns:
            out = orchestrator.run_scout_tool("grep", [p, "--json"])
            audit_results.append(f"Pattern '{p}': {out[:200]}")

        return f"""
## Security Audit Results

{chr(10).join(audit_results)}

### Recommendations
Use scout_plan to generate fixes for issues found.
"""

    elif pattern_name == "research":
        # Research - FREE
        query = args.get("query", "")
        return orchestrate_research(query, repo_root)

    elif pattern_name == "improve":
        # Autonomous code improvement pipeline
        # Import here to avoid circular imports
        from scout.improve import run_improve_pipeline

        goal = args.get("goal", "improve code quality")
        target = args.get("target")
        apply_fixes = args.get("apply", False)

        # Run the async pipeline
        import asyncio
        result = asyncio.run(
            run_improve_pipeline(
                goal=goal,
                target=target,
                apply_fixes=apply_fixes,
                repo_root=repo_root,
            )
        )

        # Format output
        output_lines = [
            f"## Scout Improve Results",
            f"Status: {result.status}",
            f"Estimated cost: ${result.cost:.4f}",
            "",
        ]

        if result.hotspots:
            output_lines.append(f"### Hotspots ({len(result.hotspots)})")
            for h in result.hotspots:
                output_lines.append(f"  - {h.get('file')}: {h.get('reason')}")
            output_lines.append("")

        if result.plans:
            output_lines.append(f"### Plans ({len(result.plans)})")
            for p in result.plans:
                output_lines.append(f"  - {p.get('file')}: {p.get('status')}")
            output_lines.append("")

        if result.errors:
            output_lines.append("### Errors")
            for e in result.errors:
                output_lines.append(f"  - {e}")
            output_lines.append("")

        return "\n".join(output_lines)

    return f"Unknown pattern: {pattern_name}"


# Load environment variables
load_dotenv()


# === Live Terminal Feedback ===


class PlanProgress:
    """Live terminal feedback for plan generation."""

    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    COST_PER_1K_TOKENS = 0.35  # Approximate cost per 1K tokens for MiniMax

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.frame = 0
        self.total_cost = 0.0
        self.total_tokens = 0
        self.start_time = None
        self.task_name = ""
        self._task_count = 0
        self._completed = 0
        self.current_model = "unknown"  # Track which model is being used

    def start(self, task_name: str):
        """Start a new task with live feedback."""
        self.task_name = task_name
        self._task_count += 1
        self._completed = 0
        if not self.quiet:
            import time

            self.start_time = time.time()
            print(f"▶ {task_name}...")

    def spin(self, message: str = ""):
        """Show spinner with message."""
        if self.quiet:
            return
        frame = self.SPINNER_FRAMES[self.frame % len(self.SPINNER_FRAMES)]
        cost_str = f"${self.total_cost:.4f}" if self.total_cost > 0 else ""
        elapsed = ""
        if self.start_time:
            import time

            elapsed_s = time.time() - self.start_time
            elapsed = f" [{elapsed_s:.1f}s]"
        msg = f"  {frame} {message} {cost_str}{elapsed}"
        # Clear line and overwrite
        sys.stdout.write("\r" + " " * 80 + "\r" + msg)
        sys.stdout.flush()
        self.frame += 1

    def progress(self, current: int, total: int, message: str = ""):
        """Show progress for multiple items."""
        if self.quiet:
            return
        pct = (current / total * 100) if total > 0 else 0
        cost_str = f"${self.total_cost:.4f}" if self.total_cost > 0 else ""
        bar_len = 20
        filled = int(bar_len * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_len - filled)
        msg = f"  [{bar}] {pct:.0f}% {message} {cost_str}"
        sys.stdout.write("\r" + " " * 80 + "\r" + msg)
        sys.stdout.flush()

    def complete(self, message: str = "", tokens: int = 0, cost: float = 0):
        """Mark task complete with final stats."""
        self._completed += 1
        self.total_cost += cost
        self.total_tokens += tokens
        if not self.quiet:
            elapsed = ""
            if self.start_time:
                import time

                elapsed_s = time.time() - self.start_time
                elapsed = f" in {elapsed_s:.1f}s"
            cost_str = (
                f" (${cost:.4f}, {tokens} tokens)" if cost > 0 or tokens > 0 else ""
            )
            sys.stdout.write("\r" + " " * 80 + "\r")
            print(f"  ✓ {message}{cost_str}{elapsed}")

    def sub_complete(self, index: int, total: int, title: str):
        """Mark a sub-task complete."""
        if not self.quiet:
            print(f"    [{index}/{total}] ✓ {title[:50]}")

    def info(self, message: str):
        """Print info message."""
        if not self.quiet:
            print(f"  ℹ {message}")

    def warning(self, message: str):
        """Print warning message."""
        if not self.quiet:
            print(f"  ⚠ {message}")

    def set_model(self, model: str):
        """Set the current model being used and display it."""
        self.current_model = model
        if not self.quiet:
            print(f"  🤖 Using model: {model}")

    def error(self, message: str):
        """Print error message."""
        if not self.quiet:
            print(f"  ✗ {message}", file=sys.stderr)

    def summary(self):
        """Print final summary."""
        if not self.quiet:
            import time

            elapsed = ""
            if self.start_time:
                elapsed_s = time.time() - self.start_time
                elapsed = f" in {elapsed_s:.1f}s"
            print(f"\n{'─' * 50}")
            print(
                f"  📊 Total: ${self.total_cost:.4f} | {self.total_tokens} tokens"
                f"{elapsed}"
            )
            print(f"{'─' * 50}")


# === Planning Context for Dynamic Re-Synthesis ===

import os
import re
from typing import Optional

# Configuration for depth limits
DEFAULT_MAX_DEPTH = 3
GLOBAL_MAX_DEPTH = int(os.getenv("SCOUT_PLAN_MAX_GLOBAL_DEPTH", "10"))

# Configurable via env vars
CONFIG_MAX_DEPTH = int(os.getenv("SCOUT_PLAN_MAX_DEPTH", str(DEFAULT_MAX_DEPTH)))


def parse_depth_override(request: str) -> tuple[str, Optional[int]]:
    """Extract @depth=N from request, return (cleaned_request, depth_override).
    
    Example: "build auth @depth=5" -> ("build auth", 5)
    """
    match = re.search(r'@depth=(\d+)', request, re.IGNORECASE)
    if match:
        depth = int(match.group(1))
        cleaned = request[:match.start()] + request[match.end():]
        return cleaned.strip(), depth
    return request, None


def get_max_depth(request: str = None) -> int:
    """Get max depth with proper precedence:
    1. Per-request override: request @depth=N
    2. Environment variable: SCOUT_PLAN_MAX_DEPTH
    3. Default: 3
    """
    if request:
        _, depth_override = parse_depth_override(request)
        if depth_override is not None:
            return min(depth_override, GLOBAL_MAX_DEPTH)
    return CONFIG_MAX_DEPTH


PIVOT_TRIGGERS = {
    # Priority 1: Critical (always trigger, even at max depth)
    "security_finding": {"priority": 1, "weight": 10, "heuristic_kw": ["security", "vulnerability", "exploit"]},
    "impossible_step": {"priority": 1, "weight": 10, "heuristic_kw": ["impossible", "cannot", "can't do"]},

    # Priority 2: High
    "dependency_conflict": {"priority": 2, "weight": 7, "heuristic_kw": ["conflict", "contradict", "incompatible"]},
    "new_critical_path": {"priority": 2, "weight": 7, "heuristic_kw": ["must have", "required", "essential"]},

    # Priority 3: Medium
    "scope_change": {"priority": 3, "weight": 5, "heuristic_kw": ["also need", "additionally", "extended"]},
    "performance_constraint": {"priority": 3, "weight": 5, "heuristic_kw": ["slow", "performance", "latency", "timeout"]},

    # Priority 4: Low (heuristics only)
    "resource_limit": {"priority": 4, "weight": 3, "heuristic_kw": ["memory", "cpu", "disk", "quota"]},
    "api_change": {"priority": 4, "weight": 3, "heuristic_kw": ["api changed", "deprecated", "breaking"]},

    # Priority 5: Subtle (LLM detection recommended)
    "user_feedback": {"priority": 5, "weight": 2, "heuristic_kw": []},
    "missing_context": {"priority": 5, "weight": 2, "heuristic_kw": ["need more", "unclear", "what about"]},
}

# Pivot threshold configuration
DEFAULT_PIVOT_THRESHOLD = 0.3
PIVOT_FEEDBACK_FILE = Path(".scout/pivot_feedback.jsonl")


def log_pivot_outcome(trigger_type: str, confirmed: bool, plan_id: str = None) -> None:
    """Log pivot trigger outcome for threshold adaptation.

    Records whether an LLM-detected pivot was confirmed or rejected,
    enabling adaptive threshold tuning based on historical precision.

    Args:
        trigger_type: Type of pivot trigger (e.g., 'security_finding')
        confirmed: True if pivot was confirmed by human/action, False if rejected
        plan_id: Optional plan ID for tracking
    """
    import json
    from datetime import datetime, timezone

    PIVOT_FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "trigger_type": trigger_type,
        "confirmed": confirmed,
        "plan_id": plan_id,
    }

    with open(PIVOT_FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def compute_optimal_threshold() -> float:
    """Compute pivot threshold that maximizes precision using logged feedback.

    Loads historical feedback from .scout/pivot_feedback.jsonl and calculates
    the threshold that maximizes precision (confirmed/total).

    Returns:
        Optimized threshold (0.0-1.0), or DEFAULT_PIVOT_THRESHOLD if insufficient data
    """
    import json
    import os

    # Check for manual override
    manual = os.environ.get("SCOUT_PLAN_PIVOT_THRESHOLD")
    if manual:
        try:
            return float(manual)
        except (ValueError, TypeError):
            pass  # Invalid value, use computed/default

    # Load feedback data
    if not PIVOT_FEEDBACK_FILE.exists():
        return DEFAULT_PIVOT_THRESHOLD

    try:
        feedback_entries = []
        with open(PIVOT_FEEDBACK_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    feedback_entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # Need minimum data points for meaningful threshold computation
        min_samples = 10
        if len(feedback_entries) < min_samples:
            return DEFAULT_PIVOT_THRESHOLD

        # Count confirmed vs total for threshold analysis
        # For now, compute simple precision: confirmed / total
        confirmed = sum(1 for e in feedback_entries if e.get("confirmed"))
        total = len(feedback_entries)

        if total > 0:
            precision = confirmed / total
            # Adjust threshold based on precision:
            # - High precision (>0.7): lower threshold to catch more
            # - Low precision (<0.3): raise threshold to reduce false positives
            # - Medium: keep default
            if precision > 0.7:
                return 0.2  # More sensitive
            elif precision < 0.3:
                return 0.5  # Less sensitive
            else:
                return DEFAULT_PIVOT_THRESHOLD

    except Exception as e:
        logging.getLogger(__name__).warning("threshold_computation_failed", error=str(e))

    return DEFAULT_PIVOT_THRESHOLD


# PlanningContext is now imported from adaptive_engine
# Keeping this comment for reference to original location


class StepResult:
    """
    Enriched step execution result with discovery reporting.
    """

    def __init__(
        self,
        step_id: int,
        success: bool,
        result: Any = None,
        findings: list[str] = None,
        dependencies_found: list[str] = None,
        files_discovered: list[str] = None,
        can_skip_subsequent: list[int] = None,
    ):
        self.step_id = step_id
        self.success = success
        self.result = result
        self.findings = findings or []
        self.dependencies_found = dependencies_found or []
        self.files_discovered = files_discovered or []
        self.can_skip_subsequent = can_skip_subsequent or []

    def to_discovery(self) -> dict:
        """Convert to discovery format for upward reporting."""
        return {
            "type": "step_result",
            "step_id": self.step_id,
            "success": self.success,
            "findings": self.findings,
            "dependencies_found": self.dependencies_found,
            "files_discovered": self.files_discovered,
            "can_skip_subsequent": self.can_skip_subsequent,
        }


def _should_pivot(context: PlanningContext) -> bool:
    """Check if discoveries warrant pivot.
    
    Uses the TriggerRegistry for evaluation.
    """
    return _trigger_registry.should_pivot(context.discoveries)


def _determine_pivot_reason(context: PlanningContext) -> str:
    """Return highest-priority (lowest number) trigger type.
    
    Uses the TriggerRegistry for evaluation.
    """
    return _trigger_registry.determine_pivot_reason(context.discoveries)


def _should_escalate_depth(context: PlanningContext, global_max: int = None) -> tuple[bool, str]:
    """Check if depth should be increased or re-synthesis triggered.
    
    Returns (should_modify, reason).
    - If at global max, still returns (True, "critical_at_max") for critical discoveries
      to trigger re-synthesis WITHOUT increasing depth
    """
    if global_max is None:
        global_max = GLOBAL_MAX_DEPTH
    
    # Always allow re-synthesis for critical discoveries, even at max depth
    if context.discoveries:
        for d in context.discoveries:
            if d.get("type") in ["security_finding", "impossible_step"]:
                # Critical: re-synthesize even at max depth
                if context.depth >= global_max:
                    return True, f"critical_at_max_depth:{d.get('type')}"
                # Below max: escalate depth
                return True, f"critical_discovery:{d.get('type')}"
    
    # Non-critical: only escalate if below global max
    if context.depth >= global_max:
        return False, "global_max_reached_non_critical"
    
    return False, "no_escalation_needed"


def _compute_heuristic_pivot_score(sub_plan_results: list[dict]) -> float:
    """Compute heuristic score for pivot likelihood (0.0 to 1.0).

    Uses keyword matching from TriggerRegistry.
    Returns low score if subtle triggers may be present (triggers LLM check).

    Threshold is adaptive based on historical precision (see compute_optimal_threshold).
    Can be overridden via SCOUT_PLAN_PIVOT_THRESHOLD env var.
    """
    # Use the registry for scoring
    return _trigger_registry.compute_heuristic_score(sub_plan_results)


def _prune_sub_plans(context: PlanningContext, sub_plans: list) -> list:
    """Remove sub-plans that are no longer needed after pivot."""
    if not context.pivots_needed and not context.discoveries:
        return sub_plans

    # Collect step IDs that can be skipped
    skip_ids = set()
    prune_reasons = {}

    for discovery in context.discoveries:
        if discovery.get("can_skip_subsequent"):
            for skip_id in discovery["can_skip_subsequent"]:
                skip_ids.add(skip_id)
                prune_reasons[skip_id] = discovery.get("reason", "discovery triggered skip")

    # Mark steps as pruned
    pruned_plans = []
    for sp in sub_plans:
        sp_id = sp.get("id")
        if sp_id in skip_ids:
            sp_copy = sp.copy()
            sp_copy["_pruned"] = True
            sp_copy["_pruned_reason"] = prune_reasons.get(sp_id, "triggered by discovery")
            pruned_plans.append(sp_copy)
        else:
            pruned_plans.append(sp)

    return pruned_plans


def _build_sub_plan_context(parent_context: PlanningContext, sub_task: dict) -> PlanningContext:
    """Build context for sub-task from parent context."""
    sub_context = PlanningContext(
        request=sub_task.get("description", sub_task.get("title", "")),
        depth=parent_context.depth + 1,
        max_depth=parent_context.max_depth,
        summary=parent_context.summary or parent_context.request,
        parent_goals=[parent_context.request] + parent_context.parent_goals,
        constraints=parent_context.constraints,
    )
    return sub_context


# === Context Gathering Functions ===

def _query_index_for_context(request: str, repo_root: Path) -> list:
    """Query Scout index for relevant files. Returns nav-style results."""
    try:
        from scout.cli.index import query_for_nav

        # Pass FULL request - let LLM decide what's relevant, not keyword filtering
        results = query_for_nav(repo_root, request, limit=10)
        return results or []
    except Exception:
        return []


def _load_living_docs(file_paths: list[str], repo_root: Path) -> str:
    """Load .deep.md docs for found files."""
    docs = []
    for fp in file_paths:
        if not fp:
            continue
        # Convert to living doc path:
        # scout/foo.py -> docs/livingDoc/scout/foo.py.deep.md
        doc_path = repo_root / "docs" / "livingDoc" / f"{fp}.deep.md"
        if doc_path.exists():
            content = doc_path.read_text(encoding="utf-8", errors="replace")
            # Truncate to first 2000 chars to save tokens
            docs.append(f"## {fp}\n{content[:2000]}\n")
    return "\n\n".join(docs)


def _load_source_code(
    file_paths: list[str], repo_root: Path, max_chars: int = 5000
) -> str:
    """
    Load actual source code from index hits.
    This is the KEY DIFFERENCE for quality - we read the REAL code, not just names.
    """
    sources = []
    for fp in file_paths:
        if not fp:
            continue
        # Skip test files
        if "test_" in fp or "/test/" in fp:
            continue

        # Direct path
        full_path = Path(fp)
        if not full_path.is_absolute():
            full_path = repo_root / fp

        # Try original path
        if not full_path.exists():
            # Try without scope/ prefix (e.g., scout/)
            if "/" in fp:
                parts = fp.split("/", 1)
                alt_fp = parts[1]  # Remove first part of path
                full_path = repo_root / alt_fp

        if full_path.exists() and full_path.is_file():
            try:
                content = full_path.read_text(encoding="utf-8", errors="replace")
                # Extract first chunk (function definitions usually at start)
                sources.append(f"## {fp}\n```python\n{content[:max_chars]}\n```\n")
            except Exception as e:
                print(f"Warning: Could not read {fp}: {e}", file=sys.stderr)

    return "\n\n".join(sources)


def _verify_plan_files(plan_text: str, repo_root: Path) -> list[dict]:
    """
    Programmatically verify plan files using Scout tools (free, no LLM needed).

    Returns list of verification results:
    - file_exists: bool
    - function_exists: bool
    - warnings: list[str]
    """
    import re

    results = []

    # Extract file paths mentioned in the plan
    # Match patterns like "scout_mcp_server.py", "scout/foo.py"
    file_patterns = re.findall(r"[\w/]+\.py", plan_text)

    # Deduplicate
    files = list(
        dict.fromkeys([f for f in file_patterns if "/" in f or f.startswith("scout_")])
    )

    for fp in files[:10]:  # Limit to 10 verifications
        result = {"file": fp, "exists": False, "suggestion": None}

        # Try various paths (original, and with common scope prefixes)
        possible_paths = [
            repo_root / fp,
            repo_root / "scout" / fp,
        ]

        for pp in possible_paths:
            if pp.exists() and pp.is_file():
                result["exists"] = True
                break

        if not result["exists"]:
            # Check if it might be a proposed new file
            if "new" in plan_text.lower() and "create" in plan_text.lower():
                result["suggestion"] = "New file - will be created"
            else:
                result["suggestion"] = "⚠️ File may not exist - verify manually"

        results.append(result)

    return results


async def _verify_plan_with_tools(
    plan_text: str, repo_root: Path, progress: PlanProgress = None
) -> str:
    """
    Verify plan using existing Scout tools (programmatically, no LLM cost).

    This is the "free" verification pass that catches obvious issues.
    """
    import re

    if progress:
        progress.spin("Verifying plan programmatically...")

    # 1. File existence check (free - just filesystem)
    file_results = _verify_plan_files(plan_text, repo_root)

    issues = []
    verified = []

    for r in file_results:
        if r["exists"]:
            verified.append(r["file"])
        else:
            issues.append(f"⚠️ {r['file']}: {r['suggestion']}")

    # 2. Use scout_grep to verify function mentions exist (free)
    # Extract function names from plan and verify they exist in mentioned files
    func_pattern = re.findall(r"`(\w+)`|def (\w+)|function (\w+)", plan_text)
    funcs = [f[0] or f[1] or f[2] for f in func_pattern]
    funcs = list(dict.fromkeys([f for f in funcs if len(f) > 3]))[:5]  # Limit

    # 3. Check for common issues in plan text
    plan_lower = plan_text.lower()

    # Check for vague steps
    vague_terms = ["somewhere", "some file", "appropriate", "as needed", "if necessary"]
    for vt in vague_terms:
        if vt in plan_lower:
            issues.append(f"💡 Consider being more specific: '{vt}'")

    # Check for missing test coverage mention
    if "test" not in plan_lower and "Test" not in plan_text:
        issues.append("💡 No test coverage mentioned - consider adding tests")

    # Build verification report - use list + join for efficiency
    verification_parts = []

    if verified:
        verification_parts.append("\n### Verified Files (exist in codebase)\n")
        for f in verified[:5]:
            verification_parts.append(f"- ✅ {f}\n")

    if issues:
        verification_parts.append("\n### Plan Issues Found\n")
        for issue in issues[:5]:  # Limit to top 5
            verification_parts.append(f"{issue}\n")

    verification_report = "".join(verification_parts)

    if progress and issues:
        progress.info(f"Found {len(issues)} potential issues")

    return verification_report


# === Quality Assessment & Budget Management ===

DEFAULT_MAX_BUDGET = 0.10  # 10 cents max for heavy refactors
MIN_BUDGET = 0.005  # Minimum budget for a single plan

# Quality thresholds (0-100)
MIN_QUALITY_SCORE = 70  # "Good enough" threshold
GOOD_QUALITY_SCORE = 85  # "Great" threshold


def _assess_plan_quality_programmatic(plan_text: str) -> dict:
    """
    Programmatically assess plan quality (free, no LLM).

    Returns dict with:
    - score: 0-100 quality score
    - issues: list of quality issues found
    - strengths: list of quality strengths
    """
    score = 50  # Start neutral
    issues = []
    strengths = []

    plan_lower = plan_text.lower()
    lines = plan_text.split("\n")

    # 1. Has overview? (+10)
    if "# overview" in plan_lower or "## overview" in plan_lower:
        score += 10
        strengths.append("Has overview section")
    else:
        issues.append("Missing overview section")

    # 2. Has numbered steps? (+10)
    has_steps = any(line.strip()[0].isdigit() for line in lines if line.strip())
    if has_steps:
        score += 10
        strengths.append("Has numbered steps")
    else:
        issues.append("Missing numbered steps")

    # 3. Has files affected? (+10)
    if "file" in plan_lower and ("## " in plan_text or "### " in plan_text):
        score += 10
        strengths.append("Lists affected files")
    else:
        issues.append("Missing files affected section")

    # 4. Has dependencies? (+5)
    if "depend" in plan_lower:
        score += 5
        strengths.append("Mentions dependencies")

    # 5. Has risk assessment? (+5)
    if "risk" in plan_lower:
        score += 5
        strengths.append("Has risk assessment")

    # 6. Has implementation phases? (+10)
    if "phase" in plan_lower or "step" in plan_lower:
        score += 10
        strengths.append("Has phased approach")

    # Penalties
    # 7. Vague language (-5 per occurrence)
    vague_terms = ["somewhere", "some file", "as needed", "if necessary", "tbd"]
    for vt in vague_terms:
        if vt in plan_lower:
            score -= 5
            issues.append(f"Contains vague term: '{vt}'")

    # 8. Too short (-10)
    if len(plan_text) < 500:
        score -= 10
        issues.append("Plan is too short (< 500 chars)")

    # 9. No test coverage (-10)
    if "test" not in plan_lower:
        score -= 10
        issues.append("No test coverage mentioned")

    # Clamp score
    score = max(0, min(100, score))

    return {
        "score": score,
        "issues": issues[:5],  # Limit to top 5
        "strengths": strengths[:5],
    }


async def _llm_quality_check(
    plan_text: str, request: str, progress: PlanProgress = None
) -> dict:
    """
    Use LLM to do a quick quality check (cheap, ~300 tokens).

    This is optional and only runs if budget allows.
    """
    if progress:
        progress.spin("Checking plan quality...")

    system_prompt = (
        "You are a code review assistant. Evaluate this implementation plan.\n"
        "Rate quality 0-100. Return JSON with:\n"
        "- score: quality score\n"
        "- missing: what's missing\n"
        "- suggestions: how to improve"
    )

    user_prompt = f"""Evaluate this plan for: {request}

{plan_text[:1500]}

Provide JSON with score (0-100), missing items, and suggestions. Be brief."""

    try:
        response, cost = await call_minimax_async(
            prompt=user_prompt, system=system_prompt, max_tokens=300
        )

        # Try to parse JSON from response
        import re
        import json

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            result = json.loads(json_match.group())
            result["cost"] = cost
            return result
    except Exception:
        pass

    return {"score": 0, "missing": [], "suggestions": [], "cost": 0}


# =============================================================================
# QualityGate Class (Phase 2)
# =============================================================================

class QualityGate:
    """
    Quality assessment for different task types with self-critique integration.
    """
    
    # Quality thresholds per task type
    THRESHOLDS = {
        "simple": {"min_score": 50, "min_length": 100},
        "plan": {"min_score": 70, "min_length": 500},
        "synthesis": {"min_score": 60, "min_length": 500},
        "verification": {"min_score": 70, "min_length": 200},
    }
    
    # Required sections for plans
    REQUIRED_SECTIONS = ["overview", "steps"]
    
    @staticmethod
    def assess_plan_quality(plan_text: str, previous_feedback: str = "") -> dict:
        """
        Assess plan quality with programmatic checks.
        
        Args:
            plan_text: The plan text to assess
            previous_feedback: Previous feedback from self-critique
            
        Returns:
            dict with score, issues, suggestions, passed
        """
        # Use existing programmatic assessment
        result = _assess_plan_quality_programmatic(plan_text)
        
        # Add additional checks for plan-specific requirements
        plan_lower = plan_text.lower()
        
        # Check for required sections
        for section in QualityGate.REQUIRED_SECTIONS:
            if f"## {section}" not in plan_lower and f"## {section}" not in plan_text:
                result["issues"].append(f"Missing required section: {section}")
                result["score"] = max(0, result["score"] - 10)
        
        # Check for placeholders (bad)
        import re
        placeholders = re.findall(r"_\d+_", plan_text)
        if placeholders:
            result["issues"].append(f"Contains {len(placeholders)} placeholders")
            result["score"] = max(0, result["score"] - 15)
        
        # Check for excessive ellipsis
        if "..." in plan_text and plan_text.count("...") > 3:
            result["issues"].append("Excessive ellipsis (...) found")
            result["score"] = max(0, result["score"] - 5)
        
        # Add previous feedback as issue if present
        if previous_feedback:
            result["previous_feedback"] = previous_feedback
        
        # Determine if passed
        threshold = QualityGate.THRESHOLDS["plan"]
        result["passed"] = (
            result["score"] >= threshold["min_score"] and
            len(plan_text) >= threshold["min_length"]
        )
        
        return result
    
    @staticmethod
    def assess_simple_quality(result: str) -> dict:
        """
        Assess simple task quality.
        
        Args:
            result: The result text to assess
            
        Returns:
            dict with score, issues, passed
        """
        threshold = QualityGate.THRESHOLDS["simple"]
        issues = []
        score = 50  # Start neutral
        
        # Check length
        if len(result) < threshold["min_length"]:
            issues.append(f"Result too short ({len(result)} < {threshold['min_length']} chars)")
            score -= 20
        
        # Check for placeholders
        import re
        if re.search(r"_\d+_", result):
            issues.append("Contains placeholders")
            score -= 15
        
        # Check for empty/incomplete responses
        if not result or result.strip() == "":
            issues.append("Empty response")
            score = 0
        
        score = max(0, min(100, score))
        
        return {
            "score": score,
            "issues": issues,
            "passed": score >= threshold["min_score"] and len(result) >= threshold["min_length"]
        }
    
    @staticmethod
    def assess_synthesis_quality(result: str) -> dict:
        """
        Assess synthesis quality (structural coherence).
        
        Args:
            result: The synthesized result to assess
            
        Returns:
            dict with score, issues, passed
        """
        threshold = QualityGate.THRESHOLDS["synthesis"]
        issues = []
        score = 50
        
        # Check minimum length
        if len(result) < threshold["min_length"]:
            issues.append(f"Result too short ({len(result)} < {threshold['min_length']} chars)")
            score -= 15
        
        # Check ( for structural markersheadings, lists)
        has_structure = "##" in result or "###" in result or "- " in result or "* " in result
        if has_structure:
            score += 20
        else:
            issues.append("No structural markers found")
        
        # Check for contradictions or disconnects (simple heuristic)
        lines = result.split("\n")
        if len(lines) < 3:
            issues.append("Too few lines for coherent synthesis")
            score -= 10
        
        score = max(0, min(100, score))
        
        return {
            "score": score,
            "issues": issues,
            "passed": score >= threshold["min_score"] and len(result) >= threshold["min_length"]
        }
    
    @staticmethod
    async def self_critique(
        plan_text: str,
        task_type: str = "plan",
        llm_call: callable = None
    ) -> dict:
        """
        Have the model critique its own output.
        
        Args:
            plan_text: The text to critique
            task_type: Type of task (plan, simple, synthesis)
            llm_call: Optional LLM call function (for testing)
            
        Returns:
            dict with issues, suggestions, score
        """
        critique_prompt = f"""Critique this {task_type} for clarity, completeness, and actionability.

{plan_text}

Provide your critique as JSON:
{{
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["suggestion 1", "suggestion 2"],
    "score": 0-100
}}
"""
        try:
            if llm_call:
                response = await llm_call(
                    prompt=critique_prompt,
                    system="You are a critical reviewer. Provide concise, actionable feedback.",
                    max_tokens=500
                )
                # Parse JSON from response
                import re
                import json
                json_match = re.search(r"\{[\s\S]*\}", response)
                if json_match:
                    result = json.loads(json_match.group())
                    return result
        except Exception:
            pass
        
        # Fallback to programmatic check if LLM fails
        return {"issues": ["Self-critique unavailable"], "suggestions": [], "score": 50}


# =============================================================================
# Generate with Quality Loop (Phase 3)
# =============================================================================

async def generate_with_quality_loop(
    prompt: str,
    system: str,
    task_type: str,
    max_tokens: int = 2048,
    context: list = None,
    previous_attempts: list = None,
    max_budget: float = 0.10,
    progress: "PlanProgress" = None,
    temperature: float = 0.0,
) -> tuple:
    """
    Generate content with iterative quality refinement.
    
    This is the core of the EIN (Escalate If Necessary) pattern:
    - Start with fast/cheap model
    - Run up to max_iterations with quality checks
    - Escalate only if quality remains insufficient
    
    Args:
        prompt: The user prompt
        system: System prompt
        task_type: Type of task (simple, plan, synthesis, verification)
        max_tokens: Max tokens to generate
        context: Optional context to include
        previous_attempts: Previous attempts with feedback (for escalation)
        max_budget: Maximum budget for this generation
        progress: Optional progress tracker
        temperature: Temperature for LLM generation (default 0.0)
        
    Returns:
        Tuple of (result, metadata, attempts_history)
    """
    from scout.llm.router import (
        select_model,
        escalation_judge,
        TASK_CONFIGS,
        get_tier_for_task,
        get_next_tier,
    )
    from scout.llm import call_llm
    
    # Initialize tracking
    task_config = TASK_CONFIGS.get(task_type, TASK_CONFIGS["simple"])
    current_tier = task_config.get("tier", "fast")
    iteration = 0
    cumulative_cost = 0.0
    quality_trend = []  # Track quality scores over time
    attempts = []  # Full history for auditability
    escalations = []  # Track escalations
    best_result = None
    best_score = 0
    over_budget = False
    
    # Build initial prompt with context
    def build_prompt(feedback: str = ""):
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{chr(10).join(context)}\n\n{full_prompt}"
        if feedback:
            full_prompt += f"\n\nPrevious attempts had these issues:\n{feedback}"
        return full_prompt
    
    while True:
        # Pre-iteration budget check (Phase 3.2)
        # Estimate cost based on tier
        estimated_cost_per_call = {
            "fast": 0.001,   # ~$0.001 for 8b model
            "medium": 0.0001, # ~$0.0001 for free tier
            "large": 0.30,    # ~$0.30 for MiniMax
        }
        estimated_cost = cumulative_cost + estimated_cost_per_call.get(current_tier, 0.001)
        
        if estimated_cost > max_budget and cumulative_cost > 0:
            over_budget = True
            break
        
        # Get model for this iteration - use router's MODEL_TIERS
        from scout.llm.router import MODEL_TIERS
        
        # Get first model from each tier (router handles provider selection)
        tier_models = {
            "fast": list(MODEL_TIERS.get("fast", {}).keys())[0] if MODEL_TIERS.get("fast") else "llama-3.1-8b-instant",
            "medium": list(MODEL_TIERS.get("medium", {}).keys())[0] if MODEL_TIERS.get("medium") else "deepseek-chat",
            "large": list(MODEL_TIERS.get("large", {}).keys())[0] if MODEL_TIERS.get("large") else "deepseek-chat",
        }
        model = tier_models.get(current_tier, "deepseek-chat")
        
        # Build feedback from previous attempts
        feedback = ""
        if previous_attempts:
            recent = previous_attempts[-3:]  # Limit to last 3
            feedback_parts = []
            for attempt in recent:
                if "issues" in attempt:
                    feedback_parts.append(f"- Iteration {attempt.get('iteration', '?')}: {', '.join(attempt['issues'])}")
            if feedback_parts:
                feedback = "\n".join(feedback_parts)
        
        if progress:
            progress.spin(f"Generating (tier={current_tier}, iter={iteration + 1})...")
        
        try:
            # Show which model is being used
            if progress:
                progress.set_model(model)
            
            # Call LLM using unified dispatcher (routes to correct provider based on model)
            result_text, cost = await call_llm_async(
                model=model,
                prompt=build_prompt(feedback),
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            cumulative_cost += cost
            
            # Assess quality
            if task_type == "plan":
                quality = QualityGate.assess_plan_quality(result_text, feedback)
            elif task_type == "simple":
                quality = QualityGate.assess_simple_quality(result_text)
            elif task_type == "synthesis":
                quality = QualityGate.assess_synthesis_quality(result_text)
            else:
                quality = {"score": 70, "passed": True, "issues": []}  # Default pass
            
            quality_trend.append(quality.get("score", 0))
            
            # Store attempt
            attempts.append({
                "iteration": iteration,
                "tier": current_tier,
                "model": model,
                "result": result_text[:200],  # Truncate for storage
                "score": quality.get("score", 0),
                "cost": cost,
                "issues": quality.get("issues", []),
                "passed": quality.get("passed", False),
            })
            
            # Check if passed
            if quality.get("passed", False):
                # Return success
                metadata = {
                    "success": True,
                    "cost": cumulative_cost,
                    "tokens_in": len(build_prompt(feedback)) // 4,
                    "tokens_out": len(result_text) // 4,
                    "model": model,
                    "provider": get_provider_info(model).get("provider", "unknown"),
                    "task_type": task_type,
                    "iteration": iteration,
                    "escalated": len(escalations) > 0,
                    "over_budget": False,
                    "escalations": escalations,
                }
                return result_text, metadata, attempts
            
            # Not passed - store as best so far if better
            if quality.get("score", 0) > best_score:
                best_result = result_text
                best_score = quality.get("score", 0)
            
            # Prepare feedback for next iteration
            feedback = quality.get("issues", [])
            feedback_str = "\n".join([f"- {issue}" for issue in feedback])
            previous_attempts = previous_attempts or []
            previous_attempts.append({
                "iteration": iteration,
                "issues": feedback,
                "suggestions": quality.get("suggestions", []),
            })
            
        except Exception as e:
            # Log error and continue - use safe fallback for progress error
            error_msg = f"Generation failed: {e}"
            if progress:
                if hasattr(progress, 'error'):
                    progress.error(error_msg)
                elif hasattr(progress, 'info'):
                    progress.info(f"ERROR: {error_msg}")
                else:
                    print(error_msg, file=sys.stderr)
            attempts.append({
                "iteration": iteration,
                "tier": current_tier,
                "error": str(e),
                "passed": False,
            })
        
        iteration += 1
        
        # Check if we should escalate
        should_escalate, reason = escalation_judge(
            current_tier=current_tier,
            iteration=iteration,
            cumulative_cost=cumulative_cost,
            quality_trend=quality_trend,
            task_criticality="normal",  # Could be enhanced with detection
            task_config=task_config,
        )
        
        if should_escalate:
            next_tier = get_next_tier(current_tier)
            if next_tier:
                escalations.append({
                    "from_tier": current_tier,
                    "to_tier": next_tier,
                    "reason": reason,
                    "cost_at_escalation": cumulative_cost,
                })
                current_tier = next_tier
                iteration = 0  # Reset iteration for new tier
                if progress:
                    progress.info(f"Escalating to {next_tier}: {reason}")
    
    # Return best result or over-budget result
    result_to_return = best_result or ""
    
    # Extract all models attempted
    models_attempted = list(set(
        t.get("model", "unknown") for t in attempts if t.get("model")
    ))
    final_model = attempts[-1].get("model", "unknown") if attempts else "unknown"
    
    metadata = {
        "success": len(escalations) > 0 or best_score > 50,
        "cost": cumulative_cost,
        "tokens_in": len(prompt) // 4,
        "tokens_out": len(result_to_return) // 4,
        "model": final_model,  # Final model used
        "models_attempted": models_attempted,  # All models tried
        "provider": "mixed",
        "task_type": task_type,
        "iteration": iteration,
        "escalated": len(escalations) > 0,
        "over_budget": over_budget,
        "escalations": escalations,
    }
    
    return result_to_return, metadata, attempts


def _should_improve(
    quality: dict, current_cost: float, max_budget: float
) -> tuple[bool, str]:
    """
    Determine if we should iterate/improve the plan.

    Returns: (should_continue, reason)
    """
    score = quality.get("score", 0)
    issues_count = len(quality.get("issues", []))

    # Check budget
    remaining_budget = max_budget - current_cost
    if remaining_budget < MIN_BUDGET:
        return False, f"Budget exhausted (${current_cost:.3f}/${max_budget})"

    # Check quality
    if score >= GOOD_QUALITY_SCORE:
        return False, f"Quality score {score} is excellent"

    if score >= MIN_QUALITY_SCORE and issues_count <= 2:
        return False, f"Good enough (score: {score}, issues: {issues_count})"

    if remaining_budget < 0.005:
        return False, "Minimum budget reached"

    # Should improve
    reason = f"Score {score} needs improvement ({issues_count} issues)"
    if score < MIN_QUALITY_SCORE:
        reason += " [BELOW THRESHOLD]"

    return True, reason


async def _improve_plan(
    plan_text: str, request: str, quality_feedback: str, progress: PlanProgress = None
) -> tuple[str, float]:
    """
    Improve the plan by first asking what context is needed, then using that context.

    This is the "right" way - don't just blindly improve, get the context needed first.
    """
    if progress:
        progress.spin("Analyzing what's missing...")

    # Step 1: Ask what context would help improve this plan
    system_prompt = """You are a code review assistant. Based on the quality feedback,
identify what specific context would help improve this plan.

Respond with a JSON array of file paths or module names you need to understand better.
Example: ["scout_mcp_server.py", "scout/router.py"]"""

    user_prompt = f"""For this plan about: {request}

Current issues: {quality_feedback}

What specific files or code would help you improve this plan? List as JSON array."""

    # Get what context is needed
    import re
    import json

    context_needed = []
    try:
        response, _ = await call_minimax_async(
            prompt=user_prompt, system=system_prompt, max_tokens=300
        )
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            context_needed = json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    if progress and context_needed:
        progress.info(f"Need context: {context_needed[:3]}")

    # Step 2: Get that context (reuse our smart gathering)
    extra_context = ""
    if context_needed:
        repo_root = Path.cwd()
        source_context = _load_source_code(
            context_needed[:5], repo_root, max_chars=3000
        )
        if source_context:
            extra_context = f"\n\nAdditional context you requested:\n{source_context}"

    # Step 3: Now improve with context
    if progress:
        progress.spin("Improving plan with context...")

    system_prompt = (
        "You are a senior software architect. "
        "Improve this implementation plan based on feedback.\n"
        "\n"
        "Keep what's good. Fix what's missing. "
        "Use the provided context to make improvements."
    )

    user_prompt = f"""Improve this plan for: {request}

Quality feedback:
{quality_feedback}

Current plan:
{plan_text}
{extra_context}

Provide improved plan. Make sure to address all the quality issues."""

    response, cost = await call_minimax_async(
        prompt=user_prompt, system=system_prompt, max_tokens=2500
    )

    return response, cost


def _load_cli_context(repo_root: Path) -> str:
    """Load Scout CLI tools context."""
    # Look for CLI directory - try scout/cli first, then src/scout/cli
    cli_dir = repo_root / "scout" / "cli"
    if not cli_dir.exists():
        cli_dir = repo_root / "src" / "scout" / "cli"
    if not cli_dir.exists():
        return ""

    # Get list of CLI modules
    py_files = sorted(
        [f.stem for f in cli_dir.glob("*.py") if f.stem not in ("__init__", "__main__")]
    )
    return f"## Scout CLI modules\n{', '.join(py_files)}"


def _load_policy_context(repo_root: Path) -> str:
    """Load relevant policy.yaml sections."""
    policy_path = repo_root / "policy.yaml"
    if not policy_path.exists():
        return ""

    content = policy_path.read_text(encoding="utf-8", errors="replace")
    # Return first 1500 chars (capabilities section most relevant)
    return f"## policy.yaml (capabilities section)\n{content[:1500]}"


async def _smart_gather_context(
    request: str, repo_root: Path, progress: PlanProgress = None
) -> str:
    """
    LLM-native context gathering: ask the LLM what it needs, then get that.
    """
    if progress:
        progress.spin("Asking LLM what context it needs...")

    # First, ask the LLM what files would help
    system_prompt = (
        "You are a senior software architect planning a feature implementation.\n"
        "Given a feature request, list the specific files, modules,\n"
        "or functions you need to understand to write a good implementation plan.\n"
        "\n"
        "Respond with a JSON array of file paths. "
        "Be specific - include exact file paths.\n"
        'Example: ["scout_mcp_server.py", "scout/router.py"]'
    )

    user_prompt = (
        f"What files do you need to create an implementation plan for:\n\n"
        f"{request}\n\nList the top 8 files as JSON array."
    )

    response, _ = await call_minimax_async(
        prompt=user_prompt, system=system_prompt, max_tokens=500
    )

    # Parse the response
    import re
    import json

    file_list = []
    try:
        json_match = re.search(r"\[[\s\S]*\]", response)
        if json_match:
            file_list = json.loads(json_match.group())
    except (json.JSONDecodeError, ValueError, AttributeError):
        pass

    if not file_list:
        for line in response.split("\n"):
            if ".py" in line:
                file_list.extend(re.findall(r"[\w/]+\.py", line))

    file_list = list(dict.fromkeys(file_list))[:10]

    if progress and file_list:
        progress.info(
            f"LLM requested: {', '.join([f.split('/')[-1] for f in file_list[:4]])}"
        )

    # Gather requested context
    context_parts = []

    if file_list:
        source_context = _load_source_code(file_list, repo_root, max_chars=4000)
        if source_context:
            context_parts.append(f"### Requested Source Code\n{source_context}")

    # Also get index hits
    nav_results = _query_index_for_context(request, repo_root)
    if nav_results:
        nav_summary = "\n".join(
            [
                f"- {r.get('target_file')}::{r.get('target_function')} "
                f"({r.get('confidence')}%)"
                for r in nav_results[:5]
            ]
        )
        context_parts.append(f"## Additional Index\n{nav_summary}")

    cli_context = _load_cli_context(repo_root)
    if cli_context:
        context_parts.append(cli_context)

    policy_context = _load_policy_context(repo_root)
    if policy_context:
        context_parts.append(policy_context)

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# Web Task Detection & Prompt Suffix
# =============================================================================

WEB_TASK_KEYWORDS = {
    "navigate", "visit", "go to", "open website", "open url",
    "click", "type in", "enter text", "fill in", "submit",
    "search on", "search for", "find on",
    "log in", "login", "log out", "logout", "sign in", "sign up", "register",
    "book", "reserve", "order", "purchase", "buy",
    "download", "upload", "extract", "scrape",
    "website", "webpage", "site", "url", "http", ".com", ".org", ".net",
    "github", "google", "amazon", "facebook", "twitter",
}

WEB_TASK_URL_PATTERNS = [
    r"https?://", r"www\.", r"\.(com|org|net|io|co)",
]


def _is_web_task(request: str) -> bool:
    """Detect if a request involves web automation tasks.
    
    Execution Traces:
    - Happy: Request contains web-related keywords or URLs
    - Failure: Request is purely code-related
    - Edge: Ambiguous request defaults to False
    
    Args:
        request: The user's request string
        
    Returns:
        True if request appears to be a web automation task
    """
    request_lower = request.lower()
    
    # Check for URL patterns
    import re
    for pattern in WEB_TASK_URL_PATTERNS:
        if re.search(pattern, request):
            return True
    
    # Check for web keywords
    for keyword in WEB_TASK_KEYWORDS:
        if keyword in request_lower:
            return True
    
    return False


def _get_web_task_prompt_suffix() -> str:
    """Get the prompt suffix for web task generation.
    
    Execution Traces:
    - Happy: Returns web-specific prompt with browser_act examples
    - Failure: N/A (always returns string)
    - Edge: N/A
    
    Returns:
        String to append to system prompt for web tasks
    """
    return """

--- WEB TASK EXTENSION ---

IMPORTANT: This request involves web automation. When generating steps for web tasks,
use the "browser_act" command with the following schema:

```json
{
  "steps": [
    {
      "id": 1,
      "description": "Navigate to example.com",
      "command": "browser_act",
      "args": {
        "action": "navigate",
        "url": "https://example.com"
      }
    },
    {
      "id": 2,
      "description": "Click the login button",
      "command": "browser_act",
      "args": {
        "action": "click",
        "target": "login button"
      }
    },
    {
      "id": 3,
      "description": "Enter username",
      "command": "browser_act",
      "args": {
        "action": "type",
        "target": "username field",
        "value": "myuser"
      }
    },
    {
      "id": 4,
      "description": "Extract first result name",
      "command": "browser_act",
      "args": {
        "action": "extract",
        "target": "first result name"
      }
    }
  ]
}
```

WEB ACTION TYPES:
- "navigate" - Go to a URL (requires "url" in args)
- "click" - Click an element (requires "target" in args)
- "type" - Type text into element (requires "target" and "value" in args)
- "extract" - Extract data from element (requires "target" in args)
- "wait" - Wait for condition (optional "condition" in args)
- "assert" - Assert element exists or has certain state (requires "condition" in args)
- "scroll" - Scroll the page

Each browser_act step must have "action" in args. Other fields are optional based on action.

Example Goals:
- "Go to github.com and search for scout" -> navigate + type + click + extract
- "Log in to example.com with user test pass secret" -> navigate + type + type + click
- "Book a hotel in NYC" -> navigate + click + type + extract

When the goal involves a website, ALL steps should use "browser_act" command.
Do NOT use other scout commands for web interactions.
"""


def _gather_context(request: str, repo_root: Path, include_source: bool = True) -> str:
    """
    Gather context for the request using LLM-native approach:
    - Query index with FULL request (let LLM decide relevance)
    - Load living docs for any files found
    - Include CLI context for Scout-related requests
    """
    context_parts = []

    # 1. Query index with the FULL request - no keyword filtering
    nav_results = _query_index_for_context(request, repo_root)

    if nav_results:
        # Get file paths from nav results
        file_paths = [r.get("target_file", "") for r in nav_results]

        # Include nav results in context (let LLM filter)
        nav_summary = "\n".join(
            [
                f"- {r.get('target_file')}::{r.get('target_function')} "
                f"(confidence: {r.get('confidence')}%)"
                for r in nav_results[:5]
            ]
        )
        context_parts.append(f"## Scout Index matches\n{nav_summary}")

        # KEY IMPROVEMENT: Load actual source code from index hits
        if include_source:
            source_context = _load_source_code(file_paths, repo_root)
            if source_context:
                context_parts.append(f"### Source Code\n{source_context}")

        # Load living docs for found files
        doc_context = _load_living_docs(file_paths, repo_root)
        if doc_context:
            context_parts.append(f"### Living Documentation\n{doc_context}")

    # 3. Always include CLI context for Scout-related requests
    cli_context = _load_cli_context(repo_root)
    if cli_context:
        context_parts.append(cli_context)

    # 4. Include policy context - always relevant for governance/tool requests
    policy_context = _load_policy_context(repo_root)
    if policy_context:
        context_parts.append(policy_context)

    return "\n\n---\n\n".join(context_parts)


async def generate_plan(
    request: str, model: str = "minimax", progress: PlanProgress = None
) -> tuple[str, int, float]:
    """
    Call MiniMax to generate a plan with real codebase context.
    Returns (plan_text, total_tokens, cost_estimate)

    Uses recursive sub-planning for complex requests:
    1. Decompose into independent sub-tasks
    2. Plan each in parallel (for speed)
    3. Synthesize into unified master plan
    """
    _audit.log("plan_start", request=request[:100], model=model)
    
    # Get repo root
    repo_root = Path.cwd().resolve()

    # Run cleanup of old plans on startup (low priority, non-blocking)
    try:
        from scout.plan_state import get_plan_state_manager
        state_manager = get_plan_state_manager(repo_root)
        cleanup_result = state_manager.cleanup_old_plans()
        if progress and (cleanup_result.get("archived") or cleanup_result.get("deleted")):
            progress.info(f"Cleanup: archived {cleanup_result.get('archived')}, deleted {cleanup_result.get('deleted')}")
    except Exception as e:
        # Cleanup is non-critical, log and continue
        import logging
        logging.getLogger(__name__).warning(f"startup_cleanup_failed: {e}")

    # Check complexity - use sub-planning for multi-phase/complex requests
    complexity_indicators = [
        " and ",
        " also ",
        " phase",
        " multiple",
        " various",
        " different",
        " several",
        " step",
        "1)",
        "2)",
        "3)",
        "first",
        "then",
        "after",
        "before",
        "however",
        "but",
    ]
    is_complex = any(ind in request.lower() for ind in complexity_indicators)

    # Also check length - longer requests are likely more complex
    if len(request) > 200 and not is_complex:
        # Could be complex - do a quick check
        is_complex = True

    _audit.log("plan_complexity", is_complex=is_complex, request_length=len(request))

    if is_complex:
        _audit.log("plan_using", mode="recursive")
        return await _generate_recursive_plan(request, repo_root, model, progress)

    # Simple request - use single planning
    _audit.log("plan_using", mode="single")
    return await _generate_single_plan(request, repo_root, model, progress)


# =============================================================================
# Critical Task Detection & Verification (Phase 5)
# =============================================================================

CRITICAL_KEYWORDS = {
    "security", "auth", "encryption", "password", "key", "token", 
    "safe", "critical", "production", "payment", "banking", "PII",
    "credential", "secret", "api_key", "oauth", "jwt", "ssl", "tls"
}


def _is_critical_task(request: str) -> bool:
    """
    Detect if a task is critical based on keywords.
    
    Args:
        request: The user's request string
        
    Returns:
        True if task appears to be critical
    """
    request_lower = request.lower()
    return any(keyword in request_lower for keyword in CRITICAL_KEYWORDS)


async def _critical_verification_pass(
    plan_text: str, 
    request: str,
    progress: "PlanProgress" = None
) -> Optional[str]:
    """
    Run a verification pass on a critical plan using a large model.
    
    This is an advisory check - it logs issues but doesn't block the plan.
    
    Args:
        plan_text: The generated plan to verify
        request: The original request
        progress: Optional progress tracker
        
    Returns:
        Verification result string or None if verification passes
    """
    if progress:
        progress.spin("Verifying critical plan...")
    
    verification_prompt = f"""You are a senior software architect reviewing a plan for a CRITICAL task.

Original request: {request}

Plan to verify:
{plan_text}

Provide a brief verification review as JSON:
{{
    "issues": ["issue 1", "issue 2"],  // List any serious issues found
    "score": 0-100,  // Overall plan quality for this critical task
    "recommendation": "approve" or "needs_work" or "reject"
}}

Focus on:
- Security implications
- Error handling
- Edge cases
- Completeness for a critical system"""

    try:
        from scout.llm import call_llm
        result = await call_llm(
            prompt=verification_prompt,
            system="You are a critical reviewer. Provide concise, actionable feedback.",
            max_tokens=500,
        )
        response = result.content
        cost = result.cost_usd
        
        # Parse JSON response
        import re
        import json
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            result = json.loads(json_match.group())
            issues = result.get("issues", [])
            recommendation = result.get("recommendation", "approve")
            
            if issues:
                return f"Found {len(issues)} issues: {', '.join(issues[:3])}"
            elif recommendation == "reject":
                return "Verification failed - recommendation: reject"
            elif recommendation == "needs_work":
                return "Verification: needs work"
            
    except Exception as e:
        if progress:
            progress.warning(f"Verification failed: {e}")
    
    return None  # Verification passed


def _structured_to_markdown(data: dict) -> str:
    """Convert structured JSON plan to markdown for display."""
    lines = []
    
    if data.get("overview"):
        lines.append(f"## Overview\n{data['overview']}\n")
    
    if data.get("steps"):
        lines.append("## Steps\n")
        for step in data["steps"]:
            step_id = step.get("id", "?")
            desc = step.get("description", "No description")
            cmd = step.get("command", "nav")
            args = step.get("args", {})
            lines.append(f"### {step_id}. {desc}")
            lines.append(f"**Command:** `{cmd}`")
            if args:
                lines.append(f"**Args:** `{args}`")
            lines.append("")
    
    if data.get("files_affected"):
        lines.append("## Files Affected")
        for f in data["files_affected"]:
            lines.append(f"- {f}")
        lines.append("")
    
    if data.get("dependencies"):
        lines.append("## Dependencies")
        for d in data["dependencies"]:
            lines.append(f"- {d}")
        lines.append("")
    
    if data.get("risk"):
        lines.append(f"## Risk Assessment\n{data['risk']}")
    
    return "\n".join(lines)


async def _generate_single_plan(
    request: str,
    repo_root: Path,
    model: str,
    progress: PlanProgress = None,
    use_smart_context: bool = False,
) -> tuple[str, int, float]:
    """Generate a single plan for a request.

    Args:
        use_smart_context: If True
    """
    _audit.log("plan_single_start", request=request[:100], model=model, use_smart_context=use_smart_context)
    
    if progress:
        progress.spin("Gathering context...")

    # Use smart context gathering if requested (for complex plans)
    if use_smart_context:
        context = await _smart_gather_context(request, repo_root, progress)
    else:
        context = _gather_context(request, repo_root)

    context_info = ""
    if context:
        # Allow more context for high-quality plans
        max_ctx = 6000 if use_smart_context else 2500
        context_info = (
            f"\n\nHere is relevant context from the codebase:\n{context[:max_ctx]}\n"
        )

    if progress:
        progress.spin("Generating plan...")

    # Detect web task - if request involves websites, generate browser_act steps
    is_web_task = _is_web_task(request)
    web_task_suffix = ""
    if is_web_task:
        _audit.log("plan_detected", task_type="web")
        web_task_suffix = _get_web_task_prompt_suffix()
    else:
        _audit.log("plan_detected", task_type="code")

    system_prompt = """You are a senior software architect. Generate implementation plans as JSON.

OUTPUT FORMAT - ALWAYS respond with valid JSON in this exact structure:

```json
{
  "overview": "2-3 sentence summary of the plan",
  "steps": [
    {
      "id": 1,
      "description": "Clear description of this step",
      "command": "scout_lint|scout_run|scout_edit|scout_create_file|scout_nav|scout_git|scout_audit|scout_roast|scout_query|scout_doc_sync|scout_env|scout_index|scout_plan|scout_brief|scout_ci_guard|browser_act",
      "args": {"path": "file.py", "instruction": "what to do"}
    }
  ],
  "files_affected": ["path/to/file.py"],
  "dependencies": ["dep1", "dep2"],
  "risk": "Low|Medium|High - brief note"
}
```

COMMAND MAPPING - Choose the most specific scout command:
- lint/fix/check errors → "scout_lint"
- run tests → "scout_run"  
- edit existing file → "scout_edit"
- create new file → "scout_create_file"
- explore/navigate codebase → "scout_nav"
- git operations → "scout_git"
- audit code quality → "scout_audit"
- roast/critique → "scout_roast"
- search/query code → "scout_query"
- docs generation → "scout_doc_sync"
- environment check → "scout_env"
- index/rebuild → "scout_index"
- plan sub-tasks → "scout_plan"
- brief/summary → "scout_brief"
- ci/guard checks → "scout_ci_guard"

EXECUTION METADATA - Add optional markers to steps for advanced execution control:
- [input: prompt text] - Pause execution and get user input before running step
- [parallel: 1,2,3] - Run these steps in parallel with current step
- [if: condition] - Only run step if condition is true (e.g., "if: user_approves")
- [on_success: next_step] - Jump to step on success (e.g., "on_success: 5")
- [on_failure: next_step] - Jump to step on failure (e.g., "on_failure: rollback")

Example with execution metadata:
## Steps
1. Create file test.py [input: Enter filename:] - pause for filename
2. Write hello world to file
3. Run tests [parallel: 4,5] - run steps 4 and 5 in parallel with this one
4. Check output
5. Verify results [if: tests_passed] - only run if tests passed
6. Deploy [on_success: done] [on_failure: rollback] - flow control

ADAPTIVE DETAIL LEVEL:
- Simple request ("simple", "quick", "just do it"): 2-3 steps, skip overview/dependencies/risk
- Standard request: 3-5 steps with overview
- Complex request ("detailed", "comprehensive"): 5-10 steps with full metadata

MINIMAL (use when user says: "simple", "quick", "just do it", "mvp", "bare bones", "lightweight", "small", "easy"):
- Skip overview, risk assessment, dependencies
- Just provide essential numbered steps (3-5 max)
- Skip "Files Affected" section or keep to one line

STANDARD (neutral requests - no strong cues either way):
- Include: Overview, Steps (numbered), Files Affected
- Skip: Risk Assessment, Dependencies unless critical

DETAILED (use when user says: "detailed", "thorough", "full", "complete", "comprehensive", "badass", "extensive", "in-depth"):
- Include ALL sections: Overview, Implementation Order, Detailed Steps, Files Affected, Dependencies, Risk Assessment
- Break down into phases
- Consider edge cases and testing

CONTEXT AWARELESS: If the request is very short (< 30 chars) or extremely simple, always default to MINIMAL.

IMPORTANT: Use the context provided. Do not ask for more info.

--- FEW-SHOT EXAMPLES ---

Example 1 (MINIMAL - user said "quick"):
User: "Create a file called test.txt with hello world"
Plan:
1. Open test.txt in write mode and write 'hello world'
2. Close the file.

Example 2 (STANDARD - neutral request):
User: "Add user authentication to the API"
Plan:
## Overview
This plan adds JWT-based authentication to the API endpoints.

## Steps
1. Create auth middleware in middleware/auth.py
2. Add /login and /register endpoints
3. Update user model with password hashing
4. Add auth required decorator

## Files Affected
- api/routes.py
- middleware/auth.py
- models/user.py

Example 3 (DETAILED - user said "comprehensive"):
User: "Fully document the authentication module with comprehensive test coverage"
Plan:
## Overview
This comprehensive plan documents the entire authentication module, covering all functions, classes, and edge cases. It includes detailed test coverage strategies.

## Implementation Order
Phase 1: Documentation Extraction (Day 1)
- Parse auth.py using AST
- Extract all function signatures and docstrings

Phase 2: Documentation Generation (Day 2)
- Generate markdown for each function
- Create index page with module overview

Phase 3: Test Coverage (Day 3)
- Write unit tests for each public function
- Add integration tests for auth flows

## Detailed Steps
1. Use `ast` to parse auth.py and collect all function definitions
2. For each function, extract:
   - Function signature
   - Docstring (first paragraph)
   - Parameters and return type
3. Generate markdown files in docs/auth/
4. Create __init__.py index
5. Write pytest tests in tests/test_auth.py

## Files Affected
- auth.py (source)
- docs/auth/*.md (generated)
- tests/test_auth.py

## Dependencies
- pytest for testing
- sphinx for documentation (optional)

## Risk Assessment
- Medium: Docstring extraction may miss edge cases
- Mitigation: Manual review of generated docs
- Testing: Add integration test to verify all functions documented"""

    # Append web task suffix if this is a web task
    system_prompt = system_prompt + web_task_suffix

    user_prompt = f"""Create an implementation plan for: {request}
{context_info}

Respond ONLY with JSON. No markdown formatting, no explanations."""

    # Use the new quality loop for generation (now default)
    use_quality_loop = os.environ.get("SCOUT_USE_QUALITY_LOOP", "true").lower() == "true"
    
    # Check if this is a critical task
    is_critical = _is_critical_task(request)
    
    if use_quality_loop or is_critical:
        # Use the new quality loop for generation
        result_text, metadata, attempts = await generate_with_quality_loop(
            prompt=user_prompt,
            system=system_prompt,
            task_type="plan",
            max_tokens=2000,
            context=[context] if context else None,
            max_budget=float(os.environ.get("SCOUT_MAX_BUDGET", "0.10")),
            progress=progress,
        )
        response_text = result_text
        cost = metadata.get("cost", 0)
        
        if progress:
            progress.info(f"Quality loop: {len(attempts)} attempts, escalated={metadata.get('escalated', False)}")
        
        # Run critical verification if needed
        if is_critical and progress:
            progress.spin("Running critical verification pass...")
            verification_result = await _critical_verification_pass(result_text, request, progress)
            if verification_result:
                progress.info(f"Verification: {verification_result}")
    
    # Parse JSON response to extract structured data
    import re
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    structured_data = {"steps": [], "overview": "", "files_affected": [], "dependencies": [], "risk": ""}
    markdown_plan = ""
    
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            structured_data = {
                "steps": parsed.get("steps", []),
                "overview": parsed.get("overview", ""),
                "files_affected": parsed.get("files_affected", []),
                "dependencies": parsed.get("dependencies", []),
                "risk": parsed.get("risk", "")
            }
            # Convert structured data back to markdown for display
            markdown_plan = _structured_to_markdown(structured_data)
        except json.JSONDecodeError:
            markdown_plan = response_text
    else:
        markdown_plan = response_text
    
    total_tokens = len(request) // 4 + len(response_text) // 4

    if progress:
        progress.complete("Plan generated", tokens=total_tokens, cost=cost)

    # Return tuple: (markdown_plan, tokens, cost, structured_steps)
    return markdown_plan, total_tokens, cost, structured_data.get("steps", [])


async def _generate_recursive_plan(
    request: str, repo_root: Path, model: str, progress: PlanProgress = None
) -> tuple[str, int, float]:
    """
    Generate a plan using recursive sub-planning:
    1. Decompose into independent sub-tasks
    2. Plan each in parallel
    3. Synthesize into master plan
    4. Handle dynamic re-synthesis if discoveries trigger pivot
    """
    _audit.log("plan_recursive_start", request=request[:100], model=model)
    
    if not progress:
        progress = PlanProgress()

    # Create initial planning context
    planning_context = PlanningContext(
        request=request,
        depth=0,
        max_depth=3,
        summary=request,
    )

    progress.start("Decomposing complex request")

    # Step 1: Decompose with context
    progress.spin("Analyzing request...")
    sub_tasks = await _decompose_request(request, repo_root, progress, planning_context)
    progress.info(f"Found {len(sub_tasks)} sub-tasks")

    if len(sub_tasks) <= 1:
        # Not actually decomposable, fall back to single plan
        return await _generate_single_plan(request, repo_root, model, progress)

    # Step 2: Plan each sub-task in PARALLEL with context
    progress.start(f"Generating {len(sub_tasks)} sub-plans in parallel")
    progress.info("All LLM calls running simultaneously...")
    sub_plans, planning_context = await _plan_sub_tasks_parallel(
        sub_tasks, repo_root, model, progress, planning_context
    )
    progress.info("All sub-plans complete")

    # Check if re-synthesis is needed due to discoveries
    max_replan_iterations = 2
    replan_count = 0

    while planning_context.replan_required and replan_count < max_replan_iterations:
        replan_count += 1
        progress.start(f"Re-synthesizing plan (attempt {replan_count})")
        progress.spin(f"Pivot triggered: {planning_context.pivot_reason}")

        # Prune irrelevant sub-plans
        sub_plans = _prune_sub_plans(planning_context, sub_plans)

        # Re-synthesize with discoveries (includes validation retry)
        synthesis_result = await _synthesize_with_validation_retry(
            request, sub_plans, repo_root, progress, planning_context
        )

        if len(synthesis_result) == 4:
            master_plan, synthesis_cost, synthesis_tokens, _ = synthesis_result
        else:
            master_plan, synthesis_cost, synthesis_tokens = synthesis_result

        # Reset pivot flags after re-synthesis
        planning_context.replan_required = False
        planning_context.is_pivoting = False

        # Update sub_plans with new master plan
        break  # Only one re-synthesis pass for now

    if not planning_context.replan_required:
        # Step 3: Synthesize (normal path)
        progress.start("Synthesizing into unified plan")
        progress.spin("Merging and ordering...")
        synthesis_result = await _synthesize_plans(
            request, sub_plans, repo_root, progress
        )
        # Handle both 3-tuple (legacy) and 4-tuple returns
        if len(synthesis_result) == 4:
            master_plan, synthesis_cost, synthesis_tokens, _ = synthesis_result
        else:
            master_plan, synthesis_cost, synthesis_tokens = synthesis_result

    # Calculate total cost
    sub_costs = sum(p.get("cost", 0) for p in sub_plans)
    total_cost = sub_costs + synthesis_cost
    total_tokens = sum(p.get("tokens", 0) for p in sub_plans) + synthesis_tokens

    # Build discovery summary for the plan
    discovery_summary = ""
    if planning_context.discoveries:
        discovery_summary = f"""
---
## Discovery Summary
- **Total discoveries**: {len(planning_context.discoveries)}
- **Pivot triggered**: {planning_context.is_pivoting}
- **Pivot reason**: {planning_context.pivot_reason or 'none'}
"""
        for i, d in enumerate(planning_context.discoveries[:5], 1):
            discovery_summary += f"- {i}. [{d.get('type', 'unknown')}] {d.get('detail', '')}\n"

    # Add metadata about recursive planning
    enhanced_plan = (
        f"# Implementation Plan (Synthesized from {len(sub_plans)} Sub-Plans)\n"
        f"{discovery_summary}\n"
        f"\n"
        f"{master_plan}\n"
        f"\n"
        f"---\n"
        f"\n"
        f"## Sub-Plan Summary\n"
    )
    for i, sp in enumerate(sub_plans, 1):
        # Skip pruned sub-plans in summary
        if sp.get("_pruned"):
            enhanced_plan += f"\n### Part {i}: {sp['title']} [PRUNED: {sp.get('_pruned_reason', '')}]\n"
        else:
            enhanced_plan += f"\n### Part {i}: {sp['title']}\n{sp.get('summary', '')[:200]}...\n"

    progress.complete("Synthesis complete", tokens=total_tokens, cost=total_cost)

    # Step 4: Quality assessment and improvement loop
    enhanced_plan, total_cost, total_tokens = await _quality_loop(
        enhanced_plan, request, total_cost, total_tokens, progress
    )

    return enhanced_plan, total_tokens, total_cost


async def _quality_loop(
    plan_text: str,
    request: str,
    current_cost: float,
    current_tokens: int,
    progress: PlanProgress = None,
    max_budget: float = DEFAULT_MAX_BUDGET,
) -> tuple[str, float, int]:
    """
    Quality assessment and improvement loop.

    Iteratively improves plan until:
    - Quality score is "good enough" (>= 70)
    - Budget is exhausted (< $0.005 remaining)
    - Max iterations reached (3)
    """
    if progress:
        progress.spin("Assessing plan quality...")

    # First pass: programmatic quality check (free)
    quality = _assess_plan_quality_programmatic(plan_text)

    if progress:
        progress.info(f"Quality score: {quality['score']}/100")

    # Add quality assessment to plan
    strengths = (
        ", ".join(quality["strengths"]) if quality["strengths"] else "None detected"
    )
    issues = ", ".join(quality["issues"]) if quality["issues"] else "None detected"
    quality_report = f"""
---

## Quality Assessment
- **Score**: {quality['score']}/100
- **Strengths**: {strengths}
- **Issues**: {issues}
"""

    plan_text += quality_report

    # Check if we should improve
    should_continue, reason = _should_improve(quality, current_cost, max_budget)

    if progress:
        progress.info(f"Quality check: {reason}")

    # Limit improvements
    max_iterations = 2
    iteration = 0

    while should_continue and iteration < max_iterations:
        iteration += 1

        if progress:
            progress.start(f"Improving plan (attempt {iteration}/{max_iterations})")

        # Build feedback from quality assessment
        feedback = f"Score: {quality['score']}/100. Issues: {quality['issues']}"

        # Try to improve with LLM
        improved_plan, improve_cost = await _improve_plan(
            plan_text, request, feedback, progress
        )

        current_cost += improve_cost
        current_tokens += improve_cost * 1000  # Rough estimate

        # Re-assess
        quality = _assess_plan_quality_programmatic(improved_plan)

        if quality["score"] > int(quality["score"]):
            # Improved! Use new plan
            plan_text = improved_plan
            plan_text += f"\n\n---\n## Quality Assessment (Attempt {iteration})\n"
            plan_text += f"- **Score**: {quality['score']}/100\n"
            plan_text += f"- **Improvement cost**: ${improve_cost:.4f}\n"

        # Check if we should continue
        should_continue, reason = _should_improve(quality, current_cost, max_budget)

        if progress:
            progress.info(f"After improvement: {reason}")

    # Final verification pass (free)
    final_verification = await _verify_plan_with_tools(plan_text, Path.cwd(), progress)
    if final_verification:
        plan_text += f"\n{final_verification}"

    return plan_text, current_cost, current_tokens


async def _decompose_request(
    request: str, repo_root: Path, progress: PlanProgress = None,
    context: PlanningContext = None
) -> list[dict]:
    """Decompose a complex request into independent sub-tasks."""
    _audit.log("plan_decompose_start", request=request[:100])
    
    if progress:
        progress.spin("Breaking into sub-tasks...")

    context_str = _gather_context(request, repo_root)

    # Build context info from PlanningContext if provided
    context_info = ""
    if context:
        context_info = f"""
PARENT CONTEXT (for sub-plan awareness):
- Parent summary: {context.summary}
- Parent goals: {', '.join(context.parent_goals) if context.parent_goals else 'none'}
- Constraints: {', '.join(context.constraints) if context.constraints else 'none'}
- Current depth: {context.depth}/{context.max_depth}
"""

    system_prompt = (
        "You are a senior software architect. "
        "Your job is to decompose complex feature requests "
        "into independent, parallelizable sub-tasks.\n"
        "\n"
        "Each sub-task should:\n"
        "1. Be independently implementable\n"
        "2. Have a clear, specific goal\n"
        "3. Not depend on other sub-tasks completing first "
        "(unless absolutely necessary)\n"
        "\n"
        "Return a JSON array of sub-tasks, each with:\n"
        '- "title": short descriptive title\n'
        '- "description": what this sub-task accomplishes\n'
        '- "dependencies": list of other sub-task indices '
        "this depends on (empty if none)"
    )

    user_prompt = f"""Decompose this feature request into 3-5 independent sub-tasks:

{request}
{context_info}
Context:
{context_str[:1500]}

Return ONLY valid JSON array, no other text. Example format:
[{{"title": "Add auth module", "description": "...", "dependencies": []}}]"""

    response_text, _ = await call_minimax_async(
        prompt=user_prompt, system=system_prompt, max_tokens=1500
    )

    # Parse JSON
    try:
        # Extract JSON from response
        import re

        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if json_match:
            sub_tasks = json.loads(json_match.group())
        else:
            sub_tasks = json.loads(response_text)
        # Add IDs to sub-tasks for tracking
        for i, st in enumerate(sub_tasks):
            st["id"] = i + 1
        return sub_tasks
    except (json.JSONDecodeError, AttributeError):
        # Fallback: treat as single task
        return [{"title": "Main Task", "description": request, "dependencies": [], "id": 1}]


async def _plan_sub_tasks_parallel(
    sub_tasks: list[dict], repo_root: Path, model: str, progress: PlanProgress = None,
    parent_context: PlanningContext = None
) -> tuple[list[dict], PlanningContext]:
    """Plan multiple sub-tasks in parallel for speed.

    Returns tuple of (sub_plans, updated_context with discoveries).
    """

    async def plan_single(
        sub_task: dict, idx: int, total: int, sub_context: PlanningContext = None
    ) -> dict:
        title = sub_task.get("title", "Unknown")

        if progress:
            progress.spin(f"Planning: {title[:30]}...")

        description = sub_task.get("description", sub_task.get("title", ""))

        # Build context for this sub-task if parent context provided
        extra_context = ""
        if sub_context:
            extra_context = f"""
SUB-PLAN CONTEXT:
- Parent goal: {sub_context.parent_goals[0] if sub_context.parent_goals else sub_context.request}
- Parent summary: {sub_context.summary}
- Depth: {sub_context.depth}/{sub_context.max_depth}
"""

        # Use simple context gathering for sub-tasks - saves 1 LLM call per sub-task
        # The simple index-based context is usually sufficient for sub-planning
        plan_result = await _generate_single_plan(
            description, repo_root, model, progress, use_smart_context=False
        )
        # Handle both 3-tuple (legacy) and 4-tuple returns
        if len(plan_result) == 4:
            plan_text, tokens, cost, _ = plan_result
        else:
            plan_text, tokens, cost = plan_result

        # Extract summary (first paragraph)
        summary = plan_text.split("\n\n")[0] if "\n\n" in plan_text else plan_text[:200]

        # Extract potential discoveries from the plan text
        # This is a simple heuristic - could be enhanced with LLM analysis
        discoveries = _extract_discoveries_from_plan(plan_text, sub_task.get("id", idx))

        if progress:
            progress.sub_complete(idx, total, title)

        return {
            "title": title,
            "description": description,
            "task_id": sub_task.get("id", idx),
            "plan": plan_text,
            "summary": summary,
            "tokens": tokens,
            "cost": cost,
            "discoveries": discoveries,
        }

    # Build sub-contexts for each task
    sub_contexts = []
    if parent_context:
        for st in sub_tasks:
            sc = _build_sub_plan_context(parent_context, st)
            sub_contexts.append(sc)
    else:
        for _ in sub_tasks:
            sub_contexts.append(None)

    # Run all in parallel using asyncio.gather
    tasks = [
        plan_single(st, i + 1, len(sub_tasks), sub_contexts[i])
        for i, st in enumerate(sub_tasks)
    ]
    results = await asyncio.gather(*tasks)

    # Aggregate discoveries into parent context if provided
    updated_context = parent_context
    if parent_context:
        for result in results:
            # Add discoveries
            parent_context.discoveries.extend(result.get("discoveries", []))
            # Record sub-plan outcome
            parent_context.sub_plan_outcomes.append({
                "task_id": result.get("task_id"),
                "task": result["title"],
                "success": True,  # If we got a result, it succeeded
                "findings": [d.get("detail", "") for d in result.get("discoveries", [])],
            })

        # Check if any finding triggers pivot
        if _should_pivot(parent_context):
            parent_context.replan_required = True
            parent_context.pivot_reason = _determine_pivot_reason(parent_context)
            parent_context.is_pivoting = True

    return list(results), updated_context


def _extract_discoveries_from_plan(plan_text: str, task_id: int) -> list[dict]:
    """Extract potential discoveries from plan text using simple heuristics."""
    discoveries = []
    plan_lower = plan_text.lower()

    # Check for dependency mentions
    if "depend" in plan_lower and ("new" in plan_lower or "need" in plan_lower):
        discoveries.append({
            "type": "dependency_found",
            "task_id": task_id,
            "detail": "Plan mentions new dependencies",
        })

    # Check for scope change indicators
    if any(word in plan_lower for word in ["additional", "also need", "also require", "extended"]):
        discoveries.append({
            "type": "scope_change",
            "task_id": task_id,
            "detail": "Plan suggests scope expansion",
        })

    # Check for potential blockers
    if any(word in plan_lower for word in ["cannot", "impossible", "can't", "blocked"]):
        discoveries.append({
            "type": "impossible_step",
            "task_id": task_id,
            "detail": "Plan mentions potential blocker",
        })

    return discoveries


async def _synthesize_with_pivot(
    original_request: str,
    sub_plans: list[dict],
    repo_root: Path,
    progress: PlanProgress,
    context: PlanningContext,
) -> tuple[str, float, int]:
    """Synthesize plans with pivot context - handles re-synthesis triggered by discoveries.

    This is called when sub-plans report discoveries that require the plan to be
    re-generated with new information.
    """
    if progress:
        progress.spin("Re-synthesizing with pivot context...")

    # Build context from sub-plans with discoveries - use list + join for efficiency
    plans_parts = []
    pruned_count = 0
    for i, sp in enumerate(sub_plans, 1):
        if sp.get("_pruned"):
            pruned_count += 1
            plans_parts.append(f"""=== SUB-PLAN {i}: {sp['title']} === [PRUNED: {sp.get('_pruned_reason', 'no reason')}]
---
""")
        else:
            plans_parts.append(f"""=== SUB-PLAN {i}: {sp['title']} ===
{sp.get('plan', '')}
---
""")
    plans_text = "\n".join(plans_parts)

    # Build discovery report
    discovery_report = json.dumps(context.discoveries, indent=2)
    pivot_info = f"""
PIVOT TRIGGERED: {context.pivot_reason}
Discovery Count: {len(context.discoveries)}
Pruned Sub-Plans: {pruned_count}
"""

    system_prompt = (
        "You are a senior software architect re-synthesizing a plan after pivot triggered.\n"
        "\n"
        "Your job is to:\n"
        "1. Review discoveries from sub-plans\n"
        "2. If findings require pivot, regenerate steps accordingly\n"
        "3. Remove any steps that are now irrelevant due to pivot\n"
        "4. Add new steps discovered as necessary\n"
        "5. Create a coherent plan that accounts for the discoveries\n"
        "\n"
        "Output a complete, polished implementation plan with these sections:\n"
        "- Overview (updated based on discoveries)\n"
        "- Implementation Order (phases with dependencies)\n"
        "- Detailed Steps (numbered, with phase labels)\n"
        "- Files Affected (consolidated)\n"
        "- Dependencies (consolidated)\n"
        "- Risk Assessment\n"
        "\n"
        "Be thorough - this is a high-stakes re-planning task."
    )

    user_prompt = f"""Re-synthesize these {len(sub_plans)} sub-plans with pivot awareness:

{pivot_info}

SUB-PLAN FINDINGS (may require pivoting):
{discovery_report}

Sub-Plan Outcomes:
{json.dumps(context.sub_plan_outcomes, indent=2)}

Plans:
{plans_text}

Original Request:
{original_request}

Provide the re-synthesized master plan now. Account for discoveries and pruned sub-plans."""

    # Use quality loop for synthesis
    try:
        response_text, metadata, attempts_history = await generate_with_quality_loop(
            prompt=user_prompt,
            system=system_prompt,
            task_type="plan",
            max_tokens=3000,
            max_budget=0.15,
            progress=progress,
        )
        cost = metadata.get("cost", 0.002)
    except Exception:
        # Fallback to minimax if quality loop fails
        response_text, cost = await call_minimax_async(
            prompt=user_prompt, system=system_prompt, max_tokens=2500
        )

    tokens = len(response_text) // 4

    # Verification pass
    verification = await _verify_plan_with_tools(response_text, repo_root, progress)
    if verification:
        response_text += f"\n{verification}"

    # Parse steps
    import re
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    steps = []
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            steps = parsed.get("steps", [])
        except json.JSONDecodeError:
            pass

    return response_text, cost, tokens, steps


async def _retry_with_validation_feedback(
    original_plan: str,
    validation_report,
    context,
    repo_root: Path,
    progress: PlanProgress,
) -> Optional[str]:
    """Retry plan synthesis with validation feedback.

    Takes the validation errors and prompts the LLM to fix them.
    Returns the fixed plan, or None if retry failed.
    """
    from scout.plan_validation import MAX_VALIDATION_RETRIES

    if progress:
        progress.spin("Retrying with validation feedback...")

    error_summary = "\n".join(f"- {err}" for err in validation_report.errors)
    warning_summary = "\n".join(f"- {warn}" for warn in validation_report.warnings)

    system_prompt = (
        "You are a senior software architect fixing a plan based on validation errors.\n"
        "\n"
        "Your job is to:\n"
        "1. Review the validation errors and warnings\n"
        "2. Modify the plan to fix all errors\n"
        "3. Address warnings where possible\n"
        "4. Maintain the overall structure and quality of the plan\n"
        "\n"
        "Output the fixed plan with the same sections as before."
    )

    user_prompt = f"""Fix this plan based on validation feedback:

VALIDATION ERRORS (must fix):
{error_summary}

VALIDATION WARNINGS (address if possible):
{warning_summary}

CURRENT PLAN:
{original_plan[:3000]}

Original request: {context.request}
Discoveries: {json.dumps(context.discoveries)}

Provide the corrected plan now. Address each error explicitly."""

    try:
        response, cost = await call_minimax_async(
            prompt=user_prompt,
            system=system_prompt,
            max_tokens=2500,
        )
        response_text = response if isinstance(response, str) else str(response)

        _audit.log("validation_retry_complete", error_count=len(validation_report.errors), new_length=len(response_text))
        return response_text

    except Exception as e:
        _audit.log("validation_retry_error", error=str(e))
        return None


async def _synthesize_with_validation_retry(
    original_request: str,
    sub_plans: list[dict],
    repo_root: Path,
    progress: PlanProgress,
    context: PlanningContext,
) -> tuple[str, float, int]:
    """Synthesize plans with pivot context and validation retry.

    This is called when sub-plans report discoveries that require the plan to be
    re-generated with new information. Includes auto-retry on validation failure.
    """
    from scout.plan_validation import (
        MAX_VALIDATION_RETRIES,
        VALIDATION_RETRY_ENABLED,
        validate_replan_consistency,
    )

    # First synthesis attempt
    synthesis_result = await _synthesize_with_pivot(
        original_request, sub_plans, repo_root, progress, context
    )

    if len(synthesis_result) == 4:
        master_plan, synthesis_cost, synthesis_tokens, _ = synthesis_result
    else:
        master_plan, synthesis_cost, synthesis_tokens = synthesis_result

    # Validate and retry if needed
    if VALIDATION_RETRY_ENABLED:
        validation_report = validate_replan_consistency(master_plan, context, sub_plans)

        retry_count = 0
        while not validation_report.is_valid and retry_count < MAX_VALIDATION_RETRIES:
            retry_count += 1
            _audit.log("validation_retry", attempt=retry_count, error_count=len(validation_report.errors))

            # Attempt to fix with LLM feedback
            fixed_plan = await _retry_with_validation_feedback(
                master_plan, validation_report, context, repo_root, progress
            )

            if fixed_plan:
                master_plan = fixed_plan
                # Re-validate the fixed plan
                validation_report = validate_replan_consistency(master_plan, context, sub_plans)
            else:
                # Retry failed, break out
                break

        if not validation_report.is_valid:
            _audit.log("validation_failed", total_attempts=retry_count + 1, error_count=len(validation_report.errors))
        else:
            _audit.log("validation_retry_done", attempts=retry_count, success=True)

    return master_plan, synthesis_cost, synthesis_tokens


async def _synthesize_plans(
    original_request: str,
    sub_plans: list[dict],
    repo_root: Path,
    progress: PlanProgress = None,
    use_70b: bool = True,
) -> tuple[str, float, int]:
    """Synthesize multiple sub-plans into a unified master plan.

    Uses 70b model for high-quality synthesis, then verifies programmatically.
    """
    _audit.log("plan_synthesize_start", sub_plan_count=len(sub_plans))
    if progress:
        progress.spin("Synthesizing sub-plans (70b model)...")

    # Build context from all sub-plans - use list + join for efficiency
    plans_parts = []
    for i, sp in enumerate(sub_plans, 1):
        plans_parts.append(f"""=== SUB-PLAN {i}: {sp['title']} ===
{sp['plan']}
---
""")
    plans_text = "\n".join(plans_parts)

    system_prompt = (
        "You are a senior software architect synthesizing multiple sub-plans "
        "into a unified master implementation plan.\n"
        "\n"
        "Your job is to:\n"
        "1. Merge overlapping steps\n"
        "2. Identify dependencies and ordering\n"
        "3. Consolidate file lists\n"
        "4. Create a coherent narrative\n"
        '5. Add a "Implementation Order" section showing dependency graph\n'
        "\n"
        "IMPORTANT: Also output a JSON section with structured steps for validation:\n"
        "```json\n"
        '{"steps": [\n'
        '  {"id": 1, "title": "Step title", "depends_on": [0], "content": "description"},\n'
        '  {"id": 2, "title": "Step title", "depends_on": [1], "content": "description"}\n'
        "]}\n"
        "```\n"
        "- id: step number (1-indexed)\n"
        "- title: brief step title\n"
        "- depends_on: array of step IDs this step depends on (0 = no dependencies)\n"
        "- content: step description\n"
        "\n"
        "Output a complete, polished implementation plan with these sections:\n"
        "- Overview\n"
        "- Implementation Order (phases with dependencies)\n"
        "- Detailed Steps (numbered, with phase labels)\n"
        "- Files Affected (consolidated)\n"
        "- Dependencies (consolidated)\n"
        "- Risk Assessment\n"
        "\n"
        "Be thorough - this is a high-stakes planning task."
    )

    user_prompt = (
        f"Synthesize these {len(sub_plans)} sub-plans into a unified master plan:\n"
        f"\n"
        f"Original Request:\n"
        f"{original_request}\n"
        f"\n"
        f"{sub_plans}\n"
        f"\n"
        "Provide the synthesized master plan now. "
        "Be comprehensive and ensure all parts work together."
    )

    # Use quality loop for synthesis - better quality for the final output
    if use_70b:
        if progress:
            progress.spin("Calling quality loop for synthesis...")
        try:
            response_text, metadata, attempts_history = await generate_with_quality_loop(
                prompt=user_prompt,
                system=system_prompt,
                task_type="plan",
                max_tokens=3000,
                max_budget=0.15,
                progress=progress,
            )
            cost = metadata.get("cost", 0.002)
        except Exception:
            # Fallback to minimax if quality loop fails
            response_text, cost = await call_minimax_async(
                prompt=user_prompt, system=system_prompt, max_tokens=2500
            )
    else:
        response_text, cost = await call_minimax_async(
            prompt=user_prompt, system=system_prompt, max_tokens=2500
        )

    tokens = len(response_text) // 4

    # FREE verification pass - uses filesystem, not LLM!
    verification = await _verify_plan_with_tools(response_text, repo_root, progress)
    if verification:
        response_text += f"\n{verification}"

    # For recursive plans, parse steps from the final response
    import re
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    steps = []
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            steps = parsed.get("steps", [])
        except json.JSONDecodeError:
            pass
    
    return response_text, cost, tokens, steps


def main():
    parser = argparse.ArgumentParser(
        description="Generate implementation plans using MiniMax."
    )
    parser.add_argument(
        "request", nargs="?", help="Natural language feature request"
    )
    parser.add_argument(
        "--pattern",
        "-p",
        help="Run a pre-built composite pattern (e.g., scout-improve). Use --list-patterns to see available.",
    )
    parser.add_argument(
        "--list-patterns",
        action="store_true",
        help="List available composite patterns",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/plans",
        help="Directory to save plans (default: docs/plans)",
    )
    parser.add_argument(
        "--json", action="store_true", help="Output JSON instead of saving file"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress output (useful for JSON output)"
    )
    parser.add_argument(
        "--model",
        default="minimax",
        help="LLM provider (minimax, groq, gemini) – default minimax",
    )
    parser.add_argument(
        "--max-budget",
        type=float,
        default=0.10,
        help="Maximum budget in dollars (default: $0.10)",
    )
    parser.add_argument(
        "--critical",
        action="store_true",
        help="Mark plan as critical (triggers verification pass with large model)",
    )
    parser.add_argument(
        "--use-quality-loop",
        action="store_true",
        help="Use the new quality loop for generation (enables tiered models and iteration)",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save output to registry and log audit event (Phase 1 pilot)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation pipeline on output before saving (Phase 2)",
    )
    args = parser.parse_args()

    # Handle --list-patterns first
    if args.list_patterns:
        from scout.cli.plan import COMPOSITE_PATTERNS
        print("Available composite patterns:")
        print()
        for pattern_doc in COMPOSITE_PATTERNS:
            # Extract pattern name from docstring
            lines = pattern_doc.strip().split("\n")
            if lines:
                print(lines[0])  # First line is the pattern definition
            print(f"  {pattern_doc.strip()}")
            print()
        return 0

    # Handle --pattern flag
    if args.pattern:
        from scout.cli.plan import run_composite_pattern
        import sys

        repo_root = Path.cwd()
        # Build args dict from remaining flags
        pattern_args = {
            "goal": args.request or "improve code quality",
            "target": None,  # Could be extracted from request
            "apply": False,
        }
        result = run_composite_pattern(args.pattern, pattern_args, repo_root)
        print(result)
        return 0

    # Original plan generation logic
    if not args.request:
        parser.print_help()
        return 1

    output_dir = Path(args.output_dir)
    if not args.json:
        output_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Create progress handler - quiet if --quiet flag set
        progress = PlanProgress(quiet=args.quiet)
        result = loop.run_until_complete(
            generate_plan(args.request, args.model, progress)
        )
        # Handle both old 3-tuple and new 4-tuple returns
        if len(result) == 4:
            plan_text, tokens, cost, steps = result
        else:
            plan_text, tokens, cost = result
            steps = []
        
        # Fallback: extract steps from plan text if not returned by planner
        if not steps and plan_text:
            from scout.batch_plan_parser import parse_plan_steps
            steps = parse_plan_steps(plan_text)
    finally:
        loop.close()

    AuditLog().log(
        "plan_generate",
        cost=cost,
        model=args.model,
        tokens=tokens,
    )

    # Phase 1/2 pilot: save output to registry if --save-output flag is set
    output_id = None
    validation_result = None
    if args.save_output:
        from scout.audit import AuditLog
        from scout.tool_output import ToolOutput, ToolOutputRegistry

        # Create ToolOutput with plan result
        tool_output = ToolOutput(
            tool_name="plan",
            content={
                "request": args.request,
                "plan_text": plan_text,
                "steps": steps,
                "output_file": str(filepath) if not args.json else None,
            },
            cost_usd=cost,
            confidence=0.8,  # Plans are LLM-generated, not deterministic
            metadata={
                "input_request": args.request,
                "model": args.model,
                "tokens": tokens,
            },
        )

        # Phase 2: Run validation pipeline if --validate flag is set
        if args.validate:
            from scout.validation_pipeline import validate_output, ValidationStage
            from pathlib import Path as RepoPath

            repo_root = RepoPath.cwd()
            validation_result = validate_output(tool_output, repo_root=repo_root)

            # Populate validation errors in output
            tool_output.validation_errors = validation_result.to_error_strings()

            # Log validation event
            audit = AuditLog()
            audit.log(
                "tool_output_validated",
                tool_name="plan",
                cost=cost,
                metadata={
                    "output_id": tool_output.output_id,
                    "is_valid": validation_result.is_valid,
                    "stage_reached": validation_result.stage_reached.value,
                    "error_count": len(validation_result.errors),
                    "warning_count": len(validation_result.warnings),
                    "duration_ms": validation_result.duration_ms,
                },
            )

            if not validation_result.is_valid:
                print(f"Validation FAILED: {len(validation_result.errors)} error(s)", file=sys.stderr)
                for error in validation_result.errors:
                    print(f"  [{error.code}] {error.message}", file=sys.stderr)

        # Save to registry
        registry = ToolOutputRegistry()
        registry.save(tool_output)

        # Log audit event
        audit = AuditLog()
        audit.log(
            "tool_output_saved",
            tool_name="plan",
            cost=cost,
            metadata={
                "output_id": tool_output.output_id,
                "input_request": args.request,
                "validated": args.validate,
                "validation_valid": validation_result.is_valid if validation_result else None,
            },
        )

        output_id = tool_output.output_id

    if args.json:
        # Always output JSON with text (markdown) and data (structured)
        # data is dynamic - could have steps, sub-steps, parallel groups, etc.
        # Consumer adapts to whatever structure is present
        result = {
            "text": plan_text,
            "data": {
                "steps": steps,
                "metadata": {
                    "tokens": tokens,
                    "cost": cost,
                    "model": args.model,
                }
            }
        }
        if output_id:
            result["data"]["metadata"]["output_id"] = output_id
        if validation_result:
            result["data"]["validation"] = {
                "is_valid": validation_result.is_valid,
                "stage_reached": validation_result.stage_reached.value,
                "error_count": len(validation_result.errors),
                "warning_count": len(validation_result.warnings),
                "duration_ms": validation_result.duration_ms,
            }
        print(json.dumps(result, indent=2))
    else:
        filename = slugify(args.request) + ".md"
        filepath = output_dir / filename
        filepath.write_text(plan_text)
        print(f"Plan saved to {filepath}")
        if output_id:
            print(f"Output ID: {output_id}")
            print(f"Cost: ${cost:.4f}" if cost else "Cost: $0.00")


if __name__ == "__main__":
    main()
