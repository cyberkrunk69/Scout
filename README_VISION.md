# Scout Vision README

**Audience:** Product/technical leads, architects, and anyone who needs to understand why Scout exists.  
**For implementation details, see README_TECHNICAL.md.

---

## The Problem We're Solving

Modern software teams share a quiet frustration: the tools that promise to make us faster often make us less certain about what's happening in our own codebases.

AI assistants hallucinate context. Documentation drifts from reality within days. Commit messages and PR descriptions become afterthoughts—repetitive, low-signal work that nobody enjoys. And when automation does step in, it often hides its costs, confidence levels, and failure modes behind a reassuring interface that says "everything is fine."

Meanwhile, your codebase grows. The mental model you built six months ago? It's already stale. The onboarding documentation for new engineers? Last updated when the project was half its current size. The architectural decisions that seemed obvious at the time? Now buried in Slack threads and Jira tickets nobody can find.

Scout exists because we believe there's a better way.

---

## What Scout Builds: The Subtext Layer

We call the persistent, versioned context that Scout generates **Subtext**.

The name is deliberate. In literature, subtext is the meaning that lies beneath the surface—the unspoken intent, the hidden connections, the deeper truth that gives words their weight. Scout's Subtext works the same way: it's the layer of understanding that sits alongside your source code, revealing what the code means rather than just what it does.

Subtext includes:

- **Living documentation** (`.tldr.md`, `.deep.md`) that stays synchronized with your code through diff-aware regeneration
- **Call graphs and dependency maps** that capture relationships between modules via AST-based analysis
- **Freshness metadata** that tracks what's changed and what hasn't
- **Audit logs** that record every meaningful automation decision

All of it stored as plain Markdown and JSON, versioned alongside your code in Git, inspectable by anyone with repository access.

Subtext is not a proprietary database or a cloud service. It's text. You can read it, edit it, diff it, and roll it back. It travels with your repository, survives vendor changes, and belongs to your team.

---

## Vision Statement

Scout should become the context operating layer for development workflows—the system that maintains Subtext and leverages it to make automation intelligent, auditable, and trustworthy.

We're building toward a world where:

| Principle | What It Means |
|-----------|---------------|
| Context is continuously generated and versioned | Symbol and module docs are generated from source, stored in-repo, and updated when code changes—no manual archaeology required. |
| Automation is inspectable by default | Every meaningful LLM operation is logged with cost, confidence, and metadata. You can see exactly what happened and why. |
| Authoring loops are low friction | Commit and PR drafts assemble from existing Subtext artifacts, minimizing expensive calls at decision time. |
| Failure modes are explicit | Budget limits, stale docs, and unresolved context surface as first-class states—not hidden side effects. |
| Humans remain decision owners | Scout prepares, summarizes, and proposes. You approve, merge, and ship. |

---

## Product Principles

### 1. Text Over Lock-In

Proprietary databases create exit costs. Cloud services create availability dependencies. Scout stores everything as plain text (Markdown, JSON, YAML) inside your repository. If Scout disappears tomorrow, your Subtext remains—readable, versioned, and portable.

### 2. Truth Over Style

A terse but accurate description beats a polished paragraph that subtly misleads. Scout optimizes for correctness and traceability. Generated docs cite their sources. Confidence scores reflect actual uncertainty. When Scout doesn't know, it says so.

### 3. Cost Is a Product Feature

Most AI tools treat spending as an afterthought—something to be monitored in a billing dashboard after the fact. Scout bakes cost awareness into the product itself:

- **Index-first routing**: The free FTS5 index handles what it can; expensive LLM calls happen only when genuinely needed. BM25F clarity scoring makes this routing data-driven and auditable.
- **Hard caps**: Per-event maximums and hourly budget ceilings that cannot be overridden by configuration.
- **Audit logs**: Every call logged with model, tokens, cost, and confidence.

You should never be surprised by a bill.

### 4. Progressive Autonomy

We don't believe in "set it and forget it" automation. Trust is earned incrementally:

1. Start with assistive tooling that prepares drafts and surfaces context.
2. Add CI guards that validate doc coverage and freshness.
3. Enable hooks that automate well-understood workflows.
4. Expand autonomy only where confidence is high and rollback is easy.

Automation without observability is a liability. Scout gives you the levers to scale autonomy at your pace.

### 5. Git-Native by Design

Subtext lives in your repository. Hooks integrate with your existing Git workflows. Drafts become commits and PRs through normal Git operations. There's no separate state to reconcile, no out-of-band sync to manage.

---

## What Scout Is Right Now

Scout delivers meaningful value as a working system with execution, observability, and self-improvement capabilities:

| Capability | Status |
|------------|--------|
| Living documentation | Production-ready. `.tldr.md` and `.deep.md` generated from source with diff-aware updates via AST validation. |
| Diff-aware regeneration | Source and symbol hashes detect changes; only stale docs are regenerated. |
| Call graph export | Full AST-based graph analysis with callers, callees, trace, impact, and usages subcommands. |
| Draft generation | Commit messages and PR descriptions assembled from Subtext artifacts. |
| Cost audit trail | Comprehensive JSONL audit logging with 40+ event types, cost tracking, and metrics aggregation. |
| Navigation with confidence scoring | Index-first routing with BM25F clarity scoring. Clear matches use the free FTS5 index; ambiguity triggers LLM escalation. |
| Workflow dashboards | CLI status commands provide visibility into doc coverage, spend, and workflow health. |
| Execution Engine | Full plan execution with SafetyGuard, RollbackManager, BudgetGuard, and dependency resolution. |
| Multi-Provider LLM | Anthropic, Google, Groq, and MiniMax with key rotation and health tracking. |
| Interactive REPL | CLI-enhanced chat with tool routing, signature introspection, and LLM-based selection. |
| Tool Framework | Unified registration across 11 tool modules: file_ops, git, llm, doc_gen, batch, validation, github, admin, anonymizer, and more. |
| Self-Improvement Pilot | Scout analyzes execution patterns and proposes improvements via quality gates. |
| Adaptive Audit Learning | EIN-enhanced monitor with Bayesian penalties and auto-repair. |
| Whimsy UI | Terminal-based observability UI with Hokusai-inspired theme, real-time sync, and cost tracking. |
| Doc Sync Pipeline | BM25-powered query system for documentation retrieval with freshness tracking. |
| Web Automation | Selenium and Playwright integration for browser-based workflows. |

---

## Recent Milestones

### February 2026 — Execution Engine + Tool Framework

After the 40k DeepSeek request incident, Scout received a complete overhaul:

- **Execution Engine** (`execution/executor.py`): SafetyGuard, RollbackManager, BudgetGuard, dependency resolution, parallel execution.
- **LLM Safeguards** (`circuit_breaker.py`, `retry.py`): Per-provider rate limiting, exponential backoff, circuit breakers.
- **Tool Framework** (`tools/`): Unified registration across 11 modules—file_ops, git, llm, doc_gen, batch, validation, github, admin, anonymizer, and more.
- **Multi-Provider LLM** (`llm/providers/`): Anthropic, Google, Groq, MiniMax with KeyState health tracking and key rotation.
- **Self-Improvement Pilot** (`self_improvement/`): Scout analyzes execution patterns and proposes improvements via quality gates.
- **Whimsy UI** (`ui/`): Terminal-based observability with Hokusai wave theme, real-time sync, and cost tracking.

### February 2026 — MCP Removed, CLI-First Governance

Scout no longer depends on MCP. The MCP infrastructure was removed. Governance now runs through `scout` CLI directly.

### February 2026 — BM25F Confidence Scoring

Navigation confidence was previously hardcoded. Replaced with dynamic scoring derived from normalized BM25 relevance, gap between top results, and multiplicative boosts for exact matches.

---

## Near-Term Mission

The execution engine and tool framework are now in place. The next phase is integration reliability and self-improvement validation:

1. **Whimsy UI hardening**: Terminal edge cases, production load testing.
2. **Self-improvement pilot**: Validate recommendations with real-world traffic.
3. **Output format migration**: Complete transition from legacy `{"plan": ...}` to `{"text": ..., "data": {...}}` format.
4. **Content guardrails**: Add filtering on LLM-generated outputs.
5. **Audit log coverage**: Expand test coverage for new audit points.
6. **Graph scalability**: Optimize AST-based call graph for large codebases.

The goal is simple: when someone runs Scout on a fresh clone, it should just work. When it doesn't work, the error should tell them exactly why.

---

## Non-Goals (For Now)

Scout is opinionated about what it won't do:

- **Not an IDE plugin**: We optimize for CLI and CI workflows, not editor integration. IDEs can consume Subtext; they don't need to host it.
- **Not a code review replacement**: Scout prepares context and drafts; humans still own the review and merge decisions.
- **Not a certainty theater**: We won't hide uncertainty to appear smarter. Low confidence is a valid output.
- **Not a demo-first product**: Operational clarity beats impressive demos that fall apart in real use.
- **Not a web dashboard (currently)**: The Cockpit is a terminal UI ("Whimsy"), not a full web application.

---

## Success Criteria

Scout is succeeding when teams can say:

| Criterion | Indicator |
|-----------|-----------|
| "Our AI context is usually current." | Subtext stays synchronized with the codebase. |
| "We can explain what automation did and what it cost." | Audit log tells the story; confidence scores explain decisions. |
| "Onboarding into this repo is easier because docs are attached to code." | New engineers find context next to the files they're reading. |
| "When Scout fails, it fails loudly and recoverably." | Failures are explicit, logs are helpful, recovery is a Git operation away. |
| "We can watch Scout work in real-time." | Whimsy UI shows agent execution and cost accumulation as they happen. |
| "Scout gets better at helping us." | Self-improvement pilot analyzes patterns and proposes improvements. |

---

## Companion Documents

| Document | Purpose |
|----------|---------|
| `README_TECHNICAL.md` | Implementation reference: modules, commands, workflows |
| `CONFIGURATION.md` | Config schema, layers, and environment variables |
| `docs/adr/*.md` | Architecture Decision Records |
| `docs/guides/*.md` | Usage guides and tutorials |

---

Scout builds Subtext. Subtext builds understanding. Understanding builds better software.
