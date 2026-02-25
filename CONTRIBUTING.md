# Contributing to Scout Core

Thank you for your interest in contributing to Scout Core. This guide will help you get started with development.

## Table of Contents

- [Setting Up Your Development Environment](#setting-up-your-development-environment)
- [Running Tests](#running-tests)
- [Adding a New Provider](#adding-a-new-provider)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## Setting Up Your Development Environment

### Prerequisites

- Python 3.10 or higher
- pip or poetry

### Installation

1. Clone the repository:

```bash
git clone https://github.com/cyberkrunk69/Scout.git
cd Scout
```

2. Install the package with development dependencies:

```bash
pip install -e ".[dev]"
```

3. Install documentation dependencies (optional):

```bash
pip install -e ".[docs]"
```

4. Set up environment variables:

Create a `.env` file in the project root with your API keys:

```bash
# Required for most operations
ANTHROPIC_API_KEY=your_key_here

# Optional providers
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
MINIMAX_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
```

Alternatively, you can set these in your shell environment.

## Running Tests

Run all tests:

```bash
pytest tests/
```

Run tests with coverage:

```bash
pytest tests/ --cov=scout --cov-report=html
```

Run specific test files:

```bash
pytest tests/scout/test_router.py
```

Run tests in watch mode:

```bash
pytest tests/ --watch
```

## Adding a New Provider

See the detailed guide: [Adding a New LLM Provider](docs/guides/adding_a_provider.md)

### Quick Overview

1. Create the base API implementation in `src/scout/llm/{provider}.py`
2. Create the provider wrapper in `src/scout/llm/providers/{provider}.py`
3. Register the provider in `src/scout/llm/providers/__init__.py`
4. Add pricing information in `src/scout/llm/pricing.py`
5. Add tests in `tests/scout/llm/test_{provider}.py`

## Code Style

This project uses:

- **ruff** for linting
- **mypy** for type checking

### Running Linters

Check for issues:

```bash
ruff check src/scout/
```

Auto-fix issues:

```bash
ruff check src/scout/ --fix
```

Run type checking:

```bash
mypy src/scout/
```

### Docstring Style

Use Google-style docstrings for all public functions and classes:

```python
def example_function(param1: str, param2: int) -> bool:
    """Short description of what the function does.

    Longer description if needed.

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: Description of when this is raised.
    """
    pass
```

## Documentation

### Building the Docs

```bash
mkdocs serve
```

This will start a local development server at `http://localhost:8000`.

### Writing Documentation

- API documentation is auto-generated from docstrings using mkdocstrings
- Guides are in `docs/guides/`
- Architecture Decision Records (ADRs) are in `docs/adr/`

### Creating a New ADR

1. Copy `docs/adr/TEMPLATE.md` to `docs/adr/ADR-XXX-new-decision.md`
2. Fill in all sections
3. Update `docs/adr/index.md` with the new ADR

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests and linters
5. Commit with descriptive messages
6. Push to your fork
7. Submit a pull request

### Commit Message Format

```
type(scope): description

[optional body]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:

```
feat(llm): add Groq provider support

- Added groq.py for API calls
- Added provider wrapper
- Added pricing information
- Added tests

Closes #123
```

## Getting Help

- Open an issue for bugs or feature requests
- Check existing ADRs in `docs/adr/` for architectural context
- Review the API documentation at `docs/api/`
