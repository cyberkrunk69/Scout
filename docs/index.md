# Scout Core

**Status:** Pre-release (alpha). Core modules are stable; browser support and CLI are still under development.

Scout Core is an LLM routing, validation, and audit layer for AI-assisted development. It provides intelligent request routing, input/output validation, cost tracking, and usage auditing.

## Features

- **LLM Routing**: Intelligent request routing with provider selection
- **Validation**: Input/output validation for AI responses
- **Trust System**: Confidence scoring and trust metrics
- **Circuit Breaker**: Failure isolation and recovery
- **Retry Mechanisms**: Configurable retry policies with exponential backoff
- **Caching**: Cost-effective response caching
- **Budget Management**: Cost tracking and budget enforcement

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

## Installation

```bash
pip install scout-core
# or from source:
git clone https://github.com/cyberkrunk69/Scout.git
cd Scout
pip install -e .
```

## Documentation

- [Guides](guides/index.md) - How-to guides and tutorials
- [API Reference](api/scout.md) - API documentation
- [Architecture Decision Records](adr/index.md) - Design decisions and rationale

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

## License

MIT License
