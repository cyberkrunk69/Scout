# Scout Core — AI-Centric Development Intelligence

[![PyPI Version](https://img.shields.io/pypi/v/scout-core.svg)](https://pypi.org/project/scout-core/)
[![Python Versions](https://img.shields.io/pypi/pyversions/scout-core.svg)](https://pypi.org/project/scout-core/)
[![License](https://img.shields.io/pypi/l/scout-core.svg)](https://github.com/cyberkrunk69/Scout/blob/main/LICENSE)

**Status:** Pre-release (alpha). Core modules are stable; browser support and CLI are still under development.

Scout is an AI-centric development intelligence layer that generates **Subtext** — versioned, living documentation that travels with your code. It provides intelligent routing, execution engine, self-improvement, and observability for AI-assisted development.

---

## The Problem

AI assistants hallucinate context. Documentation drifts from reality within days. Commit messages and PR descriptions become afterthoughts. And when automation steps in, it often hides costs, confidence levels, and failure modes behind a reassuring interface.

**Scout fixes this.**

---

## What Scout Builds: The Subtext Layer

Scout generates **Subtext** — persistent, versioned context that lives alongside your source code:

- **Living documentation** (`.tldr.md`, `.deep.md`) synchronized via diff-aware regeneration
- **Call graphs and dependency maps** via AST-based analysis
- **Freshness metadata** tracking what changed and when
- **Audit logs** recording every meaningful automation decision

Subtext is plain Markdown and JSON, versioned in Git. No proprietary databases. No cloud dependencies.

---

## Quick Start

```bash
# Installation
pip install scout-core
# or from source:
git clone https://github.com/cyberkrunk69/Scout.git
cd Scout
pip install -e .

# Interactive REPL (default)
scout

# One-shot commands
scout nav "authentication flow"
scout plan "add user registration"
scout docs src/
scout graph callers validate_symbol
```

---

## Features

### Core Intelligence

| Feature | Description |
|---------|-------------|
| **LLM Routing** | Intelligent request routing with provider selection (Anthropic, Google, Groq, MiniMax) |
| **Navigation with Confidence** | BM25F-based index routing, escalates to LLM only when needed |
| **Symbol Validation** | Validates LLM suggestions against actual codebase via AST |
| **Intent Classification** | Routes user requests to appropriate execution paths |

### Execution Engine

| Feature | Description |
|---------|-------------|
| **Plan Execution** | Execute structured plans with dependency resolution |
| **SafetyGuard** | Pre-execution safety checks |
| **BudgetGuard** | Cost limits enforced before operations |
| **RollbackManager** | Automatic rollback on failure |
| **Parallel Execution** | Run independent tasks concurrently |

### Self-Improvement

| Feature | Description |
|---------|-------------|
| **Quality Gates** | Post-execution validation with configurable gates |
| **Suggestion Engine** | Analyzes patterns, proposes improvements |
| **Auto-Approval** | Governance integration via ToolOutputGate |
| **Audit Analyzer** | Pattern detection in execution history |

### Observability

| Feature | Description |
|---------|-------------|
| **Audit Logging** | Comprehensive JSONL logs with 40+ event types |
| **Cost Tracking** | Per-operation costs, hourly budgets |
| **Whimsy UI** | Terminal-based real-time observability |
| **Confidence Scores** | Every decision annotated with confidence |

### Documentation

| Feature | Description |
|---------|-------------|
| **Doc Generation** | `.tldr.md` and `.deep.md` from source |
| **Graph Export** | AST-based call graphs, impact analysis |
| **Doc Sync** | BM25-powered documentation queries |
| **Draft Generation** | Auto-generated commit messages and PR descriptions |

### Batch Processing

| Feature | Description |
|---------|-------------|
| **Pipeline Executor** | Stateful batch operations with conditionals |
| **Expression Evaluator** | `${variable}` interpolation |
| **Circuit Breaker** | Failure isolation and recovery |
| **Retry Policies** | Configurable exponential backoff |

### Tool Framework

Unified registration across 12+ tool modules:

- `file_ops` — File read/write/search
- `git` — Git operations
- `llm` — LLM utilities
- `doc_gen` — Documentation generation
- `batch` — Batch processing
- `validation` — Output validation
- `github` — GitHub integration
- `admin` — Admin utilities
- `anonymizer` — PII anonymization
- `doc_sync` — Documentation sync

---

## CLI Commands

### Navigation & Search

```bash
scout nav <symbol>              # Navigate to symbol with confidence scoring
scout search <pattern>           # Search codebase
scout graph callers <symbol>    # Find who calls a symbol
scout graph callees <symbol>    # Find what a symbol calls
scout graph trace <from> <to>  # Find path between symbols
scout graph impact <file>       # Find affected symbols
scout graph usages <symbol>     # Find all usages
```

### Documentation

```bash
scout docs <files>              # Generate documentation
scout doc-sync query <query>    # Query docs via BM25
```

### Execution

```bash
scout plan <query>             # Generate implementation plan
scout execute <plan_id>        # Execute a plan
scout improve <target>          # Autonomous code improvement
scout web <goal>               # Browser automation
```

### Git Integration

```bash
scout git commit                # Auto-generate commit message
scout git pr                   # Generate PR description
scout on-commit                # Post-commit hook
scout prepare-commit-msg       # Prepare commit message hook
```

### Development

```bash
scout chat                     # Interactive REPL
scout edit <file> -p "<prompt>" # AI-assisted editing
scout status                   # Show session info
scout config show              # View configuration
scout self-improve             # Run self-improvement pilot
```

---

## Architecture

```
scout/
├── cli/                      # Command-line interface
│   ├── commands/              # Individual command handlers
│   ├── context/              # Session and file context
│   ├── formatting/           # Output formatting
│   ├── mcp_bridge/           # MCP protocol bridge
│   ├── repl.py               # Interactive REPL
│   └── main.py               # Entry point
│
├── adaptive_engine/          # Adaptive execution system
│   ├── blind_voting.py      # Consensus voting
│   ├── cost.py              # Cost optimization
│   ├── gates.py             # Quality gates
│   ├── state.py             # State management
│   └── triggers.py          # Event triggers
│
├── self_improvement/         # Self-improvement system
│   ├── analyzer.py           # Pattern analysis
│   ├── engine.py             # Core engine
│   ├── recommender.py        # Recommendation engine
│   ├── applier.py           # Apply improvements
│   └── pr_creator.py         # Auto-PR creation
│
├── tools/                    # Tool framework
│   ├── base.py              # Base tool interface
│   ├── file_ops.py          # File operations
│   ├── git.py               # Git tools
│   ├── llm.py               # LLM utilities
│   ├── batch.py             # Batch processing
│   ├── validation.py        # Output validation
│   ├── github.py            # GitHub integration
│   ├── admin.py             # Admin utilities
│   ├── anonymizer.py        # PII anonymization
│   └── doc_gen.py           # Documentation
│
├── doc_sync/                 # Documentation sync
│   ├── ast_facts.py         # AST extraction
│   ├── relationship_formatter.py
│   └── synthesizer.py       # Doc synthesis
│
├── doc_generation/           # Doc generation
│   ├── generator.py          # Main generator
│   ├── models.py            # Data models
│   ├── trace.py             # Tracing
│   ├── validation.py        # Validation
│   └── graph_export.py      # Graph export
│
├── ui/                       # Terminal UI (Whimsy)
│   ├── components/           # UI components
│   ├── theme/                # Theme system
│   ├── whimsy.py            # Main UI
│   └── whimsy_manager.py    # UI management
│
├── quality_gates/           # Quality gates
│   └── runtime.py           # Gate execution
│
├── adapters/                # Output adapters
│   ├── base.py              # Base adapter
│   ├── plain_text.py        # Plain text
│   ├── python.py             # Python objects
│   └── registry.py          # Adapter registry
│
├── utils/                   # Utilities
│   └── summarize.py         # Summarization
│
├── config/                  # Configuration
│   └── paths.py             # Path utilities
│
├── router.py                # Main router
├── batch_pipeline.py        # Batch executor
├── validation_pipeline.py  # Validation pipeline
├── persistence.py           # State persistence
├── big_brain.py            # High-level orchestration
├── middle_manager.py       # Middle-tier management
├── improve.py              # Improvement pipeline
└── raw_briefs.py           # Brief generation
```

---

## Product Principles

### 1. Text Over Lock-In
Everything stored as plain Markdown/JSON in your repo. Subtext survives vendor changes.

### 2. Truth Over Style
Correctness and traceability over polished output. Generated docs cite sources. Confidence scores reflect actual uncertainty.

### 3. Cost Is a Product Feature
- Index-first routing: free FTS5 handles what it can
- Hard caps: per-event and hourly limits
- Audit logs: every call logged with model, tokens, cost

### 4. Progressive Autonomy
Start with assistive tooling → add CI guards → enable hooks → expand autonomy only where confidence is high.

### 5. Git-Native by Design
Subtext lives in your repo. Hooks integrate with Git. Drafts become commits through normal Git operations.

---

## Configuration

Scout reads from `.env` and `.scout/config.yaml`:

```yaml
llm:
  providers:
    - anthropic
    - groq
    - google

limits:
  hourly_budget: 5.0
  max_cost_per_event: 0.50

drafts:
  enable_commit_drafts: true
  enable_pr_snippets: true

validation:
  trust_level: normal  # permissive, normal, strict
```

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Build documentation
pip install -e ".[docs]"
mkdocs serve

# Lint
ruff check src/

# Type check
mypy src/
```

---

## Dependencies

### Core
- `python-dotenv` — Environment variables
- `PyYAML` — Configuration
- `pydantic` — Data validation
- `httpx` — HTTP client
- `anthropic` — Anthropic API
- `tiktoken` — Token counting
- `aiosqlite` — Async SQLite
- `aiofiles` — Async file I/O
- `numpy` — Numerical computing
- `watchdog` — File watching
- `rich` — Terminal formatting
- `prompt-toolkit` — CLI prompts

### Optional
- `selenium`, `playwright` — Browser automation

---

## License

MIT License

---

## Companion Documents

| Document | Purpose |
|----------|---------|
| `README_VISION.md` | Product vision and philosophy |
| `CONFIGURATION.md` | Config schema reference |
| `docs/adr/*.md` | Architecture Decision Records |
| `docs/guides/*.md` | Usage guides |

---

*Scout builds Subtext. Subtext builds understanding. Understanding builds better software.*
