# Scout Core – LLM Routing & Search Toolkit

**Status:** Pre-release (alpha). Core modules are stable; browser support and CLI are still under development.

LLM routing, validation, and audit layer for AI-assisted development.

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

## Features

- **Search**: Full-text search with FTS5
- **Routing**: Intelligent LLM request routing
- **Validation**: Input/output validation for AI responses
- **Audit**: Cost tracking and usage auditing

## Development

Install with dev dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

## License

MIT License
