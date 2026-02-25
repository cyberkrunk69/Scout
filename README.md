# Scout Core – LLM Routing & Search Toolkit

[![PyPI Version](https://img.shields.io/pypi/v/scout-core.svg)](https://pypi.org/project/scout-core/)
[![Python Versions](https://img.shields.io/pypi/pyversions/scout-core.svg)](https://pypi.org/project/scout-core/)
[![License](https://img.shields.io/pypi/l/scout-core.svg)](https://github.com/cyberkrunk69/Scout/blob/main/LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/cyberkrunk69/Scout/main.yml)](https://github.com/cyberkrunk69/Scout/actions)
[![Test Coverage](https://img.shields.io/codecov/c/github/cyberkrunk69/Scout)](https://codecov.io/gh/cyberkrunk69/Scout)
[![Documentation Status](https://img.shields.io/readthedocs/scout-core)](https://scout-core.readthedocs.io/)

**Status:** Pre-release (alpha). Core modules are stable; browser support and CLI are still under development.

LLM routing, validation, and audit layer for AI-assisted development.

## Features

- **LLM Routing**: Intelligent request routing with provider selection
- **Validation**: Input/output validation for AI responses
- **Trust System**: Confidence scoring and trust metrics
- **Circuit Breaker**: Failure isolation and recovery
- **Retry Mechanisms**: Configurable retry policies with exponential backoff
- **Caching**: Cost-effective response caching
- **Budget Management**: Cost tracking and budget enforcement
- **Batch Processing**: Execute multiple tasks with dependency support

## Installation

```bash
pip install scout-core
# or from source:
git clone https://github.com/cyberkrunk69/Scout.git
cd Scout
pip install -e .
```

## Quick Example

```python
from scout.search import SearchIndex

# Build a search index
index = SearchIndex()
index.build([
    {"id": "1", "title": "Authentication", "content": "Handle user authentication"},
    {"id": "2", "title": "Database", "content": "Database connection handling"},
])

# Search the index
results = index.search("auth")
print(results)
```

## Documentation

- [Getting Started](https://scout-core.readthedocs.io/) - Full documentation
- [API Reference](https://scout-core.readthedocs.io/api/scout.html) - Auto-generated API docs
- [Guides](https://scout-core.readthedocs.io/guides/) - How-to guides
- [Architecture Decision Records](docs/adr/index.md) - Design decisions and rationale
- [Contributing Guide](CONTRIBUTING.md) - How to contribute

## Development

Install with dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

Build documentation:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Architecture

Scout Core is organized into several key modules:

| Module | Purpose |
|--------|---------|
| `scout.llm` | LLM provider integration, routing, and budget management |
| `scout.execution` | Plan execution and state management |
| `scout.trust` | Trust system and confidence scoring |
| `scout.cache` | Response caching |
| `scout.circuit_breaker` | Failure isolation |

For detailed architecture decisions, see the [ADR documentation](docs/adr/index.md).

## License

MIT License
