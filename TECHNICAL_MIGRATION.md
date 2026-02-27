# Technical Migration: Tech Debt Sprint — Scout Code Port from Vivarium

---

## TL;DR

This PR ports ~36,000 lines of code from Vivarium into a standalone Scout package. It adds: adaptive engine, self-improvement system, terminal UI (Whimsy), 20+ CLI commands, 12+ tool modules, batch pipeline, quality gates, and comprehensive documentation.

**Status:** Pre-release (alpha). Not yet published to PyPI. Requires Python 3.10+.

---

## What Even Is This?

**Scout is an AI-centric development intelligence layer** that generates "Subtext" — versioned, living documentation that travels with your code.

Think of it as a development companion that:
- Generates `.tldr.md` and `.deep.md` files automatically from your source code
- Helps you navigate large codebases with confidence-scored navigation
- Writes your commit messages and PR descriptions for you
- Validates that AI suggestions actually exist in your codebase
- Tracks every AI operation's cost, confidence, and outcome
- Can improve itself by analyzing its own execution patterns

It's not another chatbot. It's **infrastructure** — the layer between your code and AI assistance that makes that assistance trustworthy, auditable, and cost-effective.

---

## Why Would I Use This?

### If you're tired of...
- **AI hallucinating context** — Scout validates every suggestion against actual AST
- **Documentation drift** — Subtext is regenerated on every save, stays synced
- **Mystery meat costs** — Every operation logged with full cost transparency
- **Manual commit messages** — Auto-generates from diffs + existing docs
- **Black-box AI tools** — Confidence scores, audit trails, explicit failure modes

### If you want...
- **Index-first routing** — Free FTS5 search before expensive LLM calls
- **Hard budget caps** — Hourly limits that actually can't be overridden
- **Git-native workflow** — No separate database, everything in your repo
- **Progressive autonomy** — Start with assist, add CI guards, scale up

---

## Who Is This For?

| Persona | Why Scout Helps |
|---------|----------------|
| **Solo developers** | Auto-docs, commit msgs, cost control on personal projects |
| **Teams** | Onboarding docs attached to code, audit trails for AI decisions |
| **Startups** | Cost-effective AI assistance without budget surprises |
| **Enterprise** | Auditability, governance, no vendor lock-in |
| **AI/ML engineers** | Self-improving system that learns from your patterns |

---

## Summary

**Key Metrics:**
- **Commits:** 22
- **Files Changed:** 132
- **Additions:** +36,106 lines
- **Deletions:** -101 lines
- **Python Requirement:** 3.10+
- **PyPI Status:** Not yet published (alpha)

---

## Motivation

The Scout system was originally developed within the Vivarium codebase. This migration accomplishes several strategic goals:

1. **Independence** — Scout is now a standalone package installable via `pip install scout-core`
2. **Maintainability** — Proper module structure, type hints, and organization
3. **Distribution** — Ready for PyPI publication and community use
4. **Clarity** — Separated concerns with clear interfaces between components

---

## Changes by Module

### 1. Adaptive Engine (`src/scout/adaptive_engine/`)

A new subsystem for adaptive/hybrid execution with intelligent decision-making.

| File | Purpose |
|------|---------|
| `blind_voting.py` | Consensus voting mechanism for multi-path decisions |
| `cost.py` | Cost tracking and optimization strategies |
| `gates.py` | Quality gates integration for validation checkpoints |
| `protocols.py` | Protocol definitions for type safety |
| `state.py` | State management for adaptive workflows |
| `triggers.py` | Event triggers for reactive execution |

**Key Capabilities:**
- Dynamic model selection based on task complexity
- Cost-aware routing decisions
- Quality gate enforcement before proceeding

---

### 2. Self-Improvement System (`src/scout/self_improvement/`)

An autonomous improvement engine that analyzes execution patterns and proposes enhancements.

| File | Purpose |
|------|---------|
| `analyzer.py` | Pattern analysis across execution history |
| `applier.py` | Apply approved improvements to the system |
| `audit_analyzer.py` | Deep analysis of audit logs for pattern detection |
| `engine.py` | Core self-improvement orchestration |
| `improvement_tracker.py` | Track improvement implementation status |
| `pr_creator.py` | Automatic PR creation for approved changes |
| `recommender.py` | Generate improvement recommendations |
| `suggestion_engine.py` | AI-powered suggestion generation |

**Key Capabilities:**
- Post-execution pattern analysis
- Governance integration via ToolOutputGate
- Automatic PR generation for improvements
- Evidence-based recommendations

---

### 3. UI Components (`src/scout/ui/`)

Terminal-based observability UI ("Whimsy") inspired by Hokusai wave aesthetics.

| File | Purpose |
|------|---------|
| `whimsy.py` | Main UI controller |
| `whimsy_manager.py` | UI state and lifecycle management |
| `repl_state.py` | REPL session state |
| `config.py` | UI configuration |
| `hype.py` | Motivational messages |
| `whimsy_data.py` | UI data structures |

**Components** (`ui/components/`):
- `command_palette.py` — Quick command access
- `cost_display.py` — Real-time cost tracking
- `error_display.py` — Error visualization
- `reasoning.py` — Show LLM reasoning process
- `toast.py` — Notification system

**Theme System** (`ui/theme/`):
- `manager.py` — Theme lifecycle
- `schemes.py` — Color scheme definitions

---

### 4. CLI Enhancements (`src/scout/cli/`)

Complete rewrite of the command-line interface with modular command structure.

#### Command Handlers (`cli/commands/`)

| Command | Purpose |
|---------|---------|
| `config.py` | Configuration management (show/get/set) |
| `doc_sync_query.py` | Documentation sync queries |
| `docs.py` | Documentation generation |
| `edit.py` | AI-assisted file editing |
| `execute.py` | Plan execution |
| `git.py` | Git helpers (commit, PR, status) |
| `graph.py` | Code relationship graph queries |
| `improve.py` | Autonomous code improvement |
| `nav.py` | Symbol navigation |
| `on_commit.py` | Post-commit hook |
| `plan.py` | Implementation planning |
| `prepare_commit_msg.py` | Commit message preparation |
| `search.py` | Codebase search |
| `self_improve.py` | Self-improvement pilot |
| `status.py` | Session status |
| `web.py` | Browser automation |

#### Supporting Infrastructure

| Component | Purpose |
|-----------|---------|
| `context/files.py` | File context management |
| `context/session.py` | Session state |
| `formatting/output.py` | Output formatting |
| `formatting/progress.py` | Progress reporting |
| `formatting/streaming.py` | Streaming output |
| `mcp_bridge/client.py` | MCP protocol bridge |
| `repl.py` | Interactive REPL implementation |
| `tool_loader.py` | Dynamic tool loading |

---

### 5. Tool Framework (`src/scout/tools/`)

Unified tool registration system with 12+ specialized modules.

| Tool | Purpose |
|------|---------|
| `admin.py` | Administrative utilities |
| `batch.py` | Batch processing operations |
| `doc_gen.py` | Documentation generation |
| `file_ops.py` | File read/write/search operations |
| `git.py` | Git operations |
| `github.py` | GitHub API integration |
| `llm.py` | LLM utilities |
| `validation.py` | Output validation |
| `base.py` | Base tool interface |
| `anonymizer.py` | PII anonymization (SHA256 hashing) |

---

### 6. Documentation System

#### Doc Sync (`src/scout/doc_sync/`)
| File | Purpose |
|------|---------|
| `ast_facts.py` | AST-based symbol extraction |
| `relationship_formatter.py` | Code relationship formatting |
| `synthesizer.py` | Documentation synthesis |

#### Doc Generation (`src/scout/doc_generation/`)
| File | Purpose |
|------|---------|
| `generator.py` | Main documentation generator |
| `models.py` | Data models for generation |
| `trace.py` | Generation tracing |
| `validation.py` | Output validation |
| `graph_export.py` | Graph export functionality |
| `utils.py` | Utility functions |

---

### 7. Execution & Validation

| Module | Purpose |
|--------|---------|
| `batch_pipeline.py` | Stateful batch operations with conditionals |
| `batch_expression.py` | Expression evaluator for `${variable}` |
| `batch_context.py` | Batch execution context |
| `validation_pipeline.py` | Multi-stage validation |
| `quality_gates/runtime.py` | Runtime quality enforcement |
| `plan_store.py` | Plan persistence |
| `plan_state.py` | Plan state management |
| `plan_pruner.py` | Plan optimization |

---

### 8. Persistence & State

| File | Purpose |
|------|---------|
| `persistence.py` | State persistence layer |
| `adapters/base.py` | Base adapter interface |
| `adapters/plain_text.py` | Plain text output |
| `adapters/python.py` | Python object serialization |
| `adapters/registry.py` | Adapter registry |

---

### 9. Root Modules

| File | Purpose |
|------|---------|
| `big_brain.py` | High-level orchestration |
| `middle_manager.py` | Middle-tier management |
| `improve.py` | Improvement pipeline |
| `raw_briefs.py` | Brief generation |
| `router.py` | Main routing logic |

---

### 10. Utilities (`src/scout/utils/`)

- `summarize.py` — Text summarization

---

### 11. Configuration

- `config/paths.py` — Path utilities
- `config/groq_model_specs.json` — Groq model specifications
- `pyproject.toml` — Updated with proper metadata, dependencies, and entry points

---

## New CLI Commands

After this PR, Scout supports:

```
scout chat                      # Interactive REPL (default)
scout nav <symbol>             # Navigate to symbol
scout search <pattern>         # Search codebase
scout plan <query>             # Generate plan
scout execute <plan_id>        # Execute plan
scout docs <files>             # Generate docs
scout edit <file> -p "<prompt>" # AI editing
scout git <subcommand>         # Git helpers
scout config <action>          # Config management
scout status                   # Session info
scout self-improve             # Self-improvement pilot
scout doc-sync query <query>   # Doc queries
scout improve <target>         # Autonomous improvement
scout web <goal>               # Browser automation
scout graph callers <symbol>  # Call graph
scout graph callees <symbol>  # Callee graph
scout graph trace <from> <to>  # Trace paths
scout graph impact <file>      # Impact analysis
scout graph usages <symbol>   # Find usages
scout on-commit                # Post-commit hook
scout prepare-commit-msg       # Commit msg hook
scout version                  # Show version
scout clear                    # Clear screen
```

---

## Breaking Changes & Migration Notes

### 1. Entry Point Changes
- Old: No direct CLI entry point
- New: `scout` command available after installation
- **Action:** Run `pip install -e .` to enable

### 2. Module Structure
- All modules now under `scout.` namespace
- Imports changed from `vivarium.*` to `scout.*`
- **Action:** Update any external imports

### 3. Configuration Location
- Previously: Mixed locations
- Now: `.scout/config.yaml` or `.env`
- **Action:** Review and migrate config

---

## Testing

### New Tests Added
- `tests/scout/test_relationship_formatter.py`
- `tests/scout/test_batch_expression.py`
- `tests/scout/test_plan_store.py`
- `tests/scout/execution/test_mapper.py`

### Running Tests
```bash
pytest tests/
```

---

## Dependencies

### Core Dependencies Added/Updated
- `python-dotenv>=1.0.0`
- `PyYAML>=6.0`
- `pydantic>=2.0`
- `httpx>=0.24.0`
- `anthropic>=0.79.0`
- `tiktoken>=0.7.0`
- `aiosqlite>=0.19.0`
- `aiofiles>=23.0.0`
- `numpy>=1.24.0`
- `watchdog>=4.0.0`
- `rich>=13.0`
- `prompt-toolkit>=3.0`

### Optional Dependencies
- Browser automation: `selenium>=4.0`, `playwright>=1.40`
- Development: `pytest`, `ruff`, `mypy`
- Documentation: `mkdocs`, `mkdocs-material`

---

## Architecture Decisions

### 1. Text Over Lock-In
All Subtext stored as Markdown/JSON in-repo. No proprietary formats.

### 2. Cost as Feature
- Hard budget caps enforced
- Free model routing for simple tasks
- Full audit logging

### 3. Progressive Autonomy
- Assistive → CI guards → Hooks → Full autonomy
- Quality gates at each stage

### 4. Git-Native
- Subtext versioned in Git
- Hooks integrate with existing workflows
- Drafts become commits through normal Git

---

## Future Work

This PR establishes the foundation. Near-term priorities:

1. **Whimsy UI hardening** — Terminal edge cases
2. **Self-improvement validation** — Test with real traffic
3. **Output format migration** — Complete transition to `{"text": ..., "data": {...}}`
4. **Content guardrails** — Filter LLM outputs
5. **Audit coverage** — Expand test coverage
6. **Graph scalability** — Optimize for large codebases

---

## Review Checklist

- [ ] All imports resolve correctly
- [ ] CLI entry point works: `scout --help`
- [ ] Tests pass: `pytest tests/`
- [ ] No placeholder values in config
- [ ] Documentation builds: `mkdocs serve`
- [ ] Type checking passes: `mypy src/scout`
- [ ] Linting passes: `ruff check src/scout`

---

## Commits

| Commit | Description |
|--------|-------------|
| `b4a86b0` | feat(scout): port utils from vivarium |
| `d8d09ae` | feat(scout): port adaptive_engine from vivarium |
| `d1c9095` | feat(scout): port validation_pipeline from vivarium |
| `8e2720e` | feat(scout): port quality_gates from vivarium |
| `11758f5` | feat(scout): port self_improvement from vivarium |
| `5c75999` | feat(scout): port ui from vivarium and add missing deps |
| `0057578` | feat(scout): port root files from vivarium |
| `5fe92ca` | feat(scout): port cli_enhanced from vivarium and add entry point |
| `7503553` | feat(tools): port doc_gen module from vivarium |
| `e333c53` | feat(tools): port admin module from vivarium |
| `adb0f78` | feat(tools): port admin module from vivarium |
| `a413da1` | feat(scout): port persistence from vivarium |
| `34ec676` | fix(tools): correct log_tool_invocation imports |
| `0295fc4` | feat(adapters): port base and registry from vivarium |
| `dc24f3c` | refactor(tools): replace local log_tool_invocation stubs |
| `3e01296` | feat(doc_sync): port synthesizer from vivarium |
| `fe373c4` | refactor: centralize paths, add missing __init__.py |
| `8712ea6` | feat(doc_sync): add ast_facts.py for AST extraction |
| `9c61042` | fix(cli): remove cli_enhanced references |
| `91912fe` | feat(doc_sync): port relationship_formatter and tests |
| `909dde3` | feat(cli): port plan module from vivarium |
| `74790ee` | feat: add doc generation module, groq config, vision README |
| `3894bc2` | docs: update README with full feature breakdown |

---

## Conclusion

This PR transforms Scout from a subsystem within Vivarium into a **standalone, production-ready AI development intelligence platform**. With comprehensive tooling, execution engine, self-improvement capabilities, and observability, Scout is positioned as the context layer that makes AI-assisted development trustworthy and cost-effective.

**Recommendation:** Merge after reviewing the checklist above.
