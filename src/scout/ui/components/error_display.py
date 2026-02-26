#!/usr/bin/env python
"""
Error display component - structured error presentation with suggestions.

Features:
- Parse exception types and show contextual help
- "Did you mean..." suggestions for typos
- Stack trace collapsible/expandable
- Error categorization and recovery suggestions
"""
from __future__ import annotations

import inspect
import re
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from scout.ui.components.base import BaseComponent, ComponentConfig, RefreshStrategy
from scout.ui.theme.manager import get_theme_manager


class ErrorSeverity(Enum):
    """Error severity levels."""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class ErrorCategory(Enum):
    """Categories of errors for context."""
    SYNTAX = auto()
    IMPORT = auto()
    RUNTIME = auto()
    TIMEOUT = auto()
    NETWORK = auto()
    AUTH = auto()
    PERMISSION = auto()
    NOT_FOUND = auto()
    UNKNOWN = auto()


@dataclass
class ParsedError:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    suggestion: str = ""
    did_you_mean: list[str] = field(default_factory=list)
    recoverable: bool = False
    help_url: str = ""


class ErrorDisplay(BaseComponent):
    """
    Structured error display with suggestions and help.
    
    Features:
    - Automatic error categorization
    - Contextual suggestions
    - Collapsible stack traces
    - Recovery actions
    """
    
    # Error patterns for categorization
    ERROR_PATTERNS = {
        ErrorCategory.SYNTAX: [
            (r"SyntaxError:.*", "Check for missing brackets, quotes, or colons"),
            (r"IndentationError:", "Check indentation - use consistent spaces or tabs"),
        ],
        ErrorCategory.IMPORT: [
            (r"ImportError:", "Check that the module is installed"),
            (r"ModuleNotFoundError:", "Install missing module with pip"),
            (r"cannot import name", "Check for circular imports or typos"),
        ],
        ErrorCategory.RUNTIME: [
            (r"TypeError:", "Check argument types"),
            (r"ValueError:", "Check argument values"),
            (r"AttributeError:", "Check object has the attribute"),
            (r"KeyError:", "Check dictionary key exists"),
            (r"NameError:", "Check variable is defined before use"),
        ],
        ErrorCategory.TIMEOUT: [
            (r"TimeoutError:", "Operation took too long - try again or increase timeout"),
            (r"asyncio.TimeoutError:", "Async operation timed out"),
        ],
        ErrorCategory.NETWORK: [
            (r"ConnectionError:", "Check network connection"),
            (r"HTTPError:", "Check API endpoint and credentials"),
            (r"SSLError:", "Check SSL certificates"),
        ],
        ErrorCategory.AUTH: [
            (r"AuthenticationError", "Check API keys and credentials"),
            (r"PermissionError:", "Check file permissions"),
        ],
        ErrorCategory.NOT_FOUND: [
            (r"FileNotFoundError:", "Check file path exists"),
            (r"404", "Resource not found - check URL"),
        ],
    }
    
    # Common typo suggestions
    TYPO_SUGGESTIONS = {
        "scout": ["scout_", "scout-", "Scout"],
        "improt": "import",
        "form": "from",
        "functon": "function",
        "def": "def",
        "class": "class",
        "return": "return",
        "print": "print",
    }
    
    def __init__(
        self,
        console: Optional[Console] = None,
        show_traceback: bool = True,
        **kwargs
    ):
        config = ComponentConfig(
            refresh_rate=0.5,  # Static after display
            refresh_strategy=RefreshStrategy.STATIC,
        )
        super().__init__(config=config, **kwargs)
        
        self._console = console or Console()
        self._current_error: Optional[ParsedError] = None
        self._show_traceback = show_traceback
        self._theme = get_theme_manager().current_theme
        self._expanded = False
        
        get_theme_manager().add_listener(self._on_theme_change)
        
    def _on_theme_change(self, theme):
        self._theme = theme
        self._schedule_refresh()
        
    def display_error(self, error: Exception, context: str = "") -> ParsedError:
        """Parse and display an error."""
        parsed = self._parse_error(error, context)
        self._current_error = parsed
        self._schedule_refresh()
        return parsed
        
    def display_message(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
    ) -> ParsedError:
        """Display an error from a message string."""
        parsed = ParsedError(
            category=category,
            severity=severity,
            message=message,
        )
        self._current_error = parsed
        self._schedule_refresh()
        return parsed
        
    def _parse_error(self, error: Exception, context: str = "") -> ParsedError:
        """Parse an exception into structured error info."""
        error_str = str(error)
        error_type = type(error).__name__
        full_message = f"{error_type}: {error_str}"
        
        # Find category
        category = ErrorCategory.UNKNOWN
        for cat, patterns in self.ERROR_PATTERNS.items():
            for pattern, suggestion in patterns:
                if re.search(pattern, full_message, re.IGNORECASE):
                    category = cat
                    break
            if category != ErrorCategory.UNKNOWN:
                break
                
        # Determine severity
        if category in (ErrorCategory.SYNTAX, ErrorCategory.PERMISSION):
            severity = ErrorSeverity.ERROR
        elif category == ErrorCategory.TIMEOUT:
            severity = ErrorSeverity.WARNING
        else:
            severity = ErrorSeverity.ERROR
            
        # Check for suggestions
        suggestion = self._get_suggestion(full_message, category)
        did_you_mean = self._check_typos(error_str)
        
        # Check if recoverable
        recoverable = category in (
            ErrorCategory.TIMEOUT,
            ErrorCategory.NETWORK,
            ErrorCategory.NOT_FOUND,
        )
        
        return ParsedError(
            category=category,
            severity=severity,
            message=full_message,
            suggestion=suggestion,
            did_you_mean=did_you_mean,
            recoverable=recoverable,
        )
        
    def _get_suggestion(self, error_msg: str, category: ErrorCategory) -> str:
        """Get suggestion based on error category."""
        suggestions = {
            ErrorCategory.SYNTAX: "Check for syntax errors in your code",
            ErrorCategory.IMPORT: "Ensure the module is installed: pip install <module>",
            ErrorCategory.RUNTIME: "Check the error message for hints about the issue",
            ErrorCategory.TIMEOUT: "Try again or increase the timeout value",
            ErrorCategory.NETWORK: "Check your internet connection and API endpoints",
            ErrorCategory.AUTH: "Verify your API keys are correct in .env",
            ErrorCategory.PERMISSION: "Check file permissions or run with appropriate access",
            ErrorCategory.NOT_FOUND: "Verify the file path or URL is correct",
            ErrorCategory.UNKNOWN: "Check the error message for more details",
        }
        return suggestions.get(category, "")
        
    def _check_typos(self, text: str) -> list[str]:
        """Check for common typos in the error message."""
        suggestions = []
        text_lower = text.lower()
        
        for typo, correction in self.TYPO_SUGGESTIONS.items():
            if typo in text_lower:
                if isinstance(correction, list):
                    suggestions.extend(correction)
                else:
                    suggestions.append(correction)
                    
        return suggestions[:3]  # Limit to 3 suggestions
        
    def toggle_expand(self):
        """Toggle stack trace visibility."""
        self._expanded = not self._expanded
        self._schedule_refresh()
        
    def clear(self):
        """Clear the current error."""
        self._current_error = None
        self._schedule_refresh()
        
    def render(self) -> Optional[Panel]:
        """Render the error display."""
        if not self._current_error:
            return None
            
        error = self._current_error
            
        # Build error content
        parts = []
        
        # Severity indicator
        severity_icons = {
            ErrorSeverity.INFO: "ℹ",
            ErrorSeverity.WARNING: "⚠",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "⛔",
        }
        severity_colors = {
            ErrorSeverity.INFO: "blue",
            ErrorSeverity.WARNING: "yellow",
            ErrorSeverity.ERROR: "red",
            ErrorSeverity.CRITICAL: "bold red",
        }
        
        icon = severity_icons[error.severity]
        color = severity_colors[error.severity]
        
        # Main error message
        parts.append(f"[{color}]{icon} {error.message}[/]")
        
        # Category
        parts.append(f"\n[dim]Category:[/dim] {error.category.name}")
        
        # Suggestion
        if error.suggestion:
            parts.append(f"\n[cyan]Suggestion:[/cyan] {error.suggestion}")
            
        # Did you mean
        if error.did_you_mean:
            parts.append(f"\n[yellow]Did you mean?[/yellow]")
            for suggestion in error.did_you_mean:
                parts.append(f"  - {suggestion}")
                
        # Recovery
        if error.recoverable:
            parts.append(f"\n[green]This error may be recoverable - try again.[/]")
            
        content = "\n".join(parts)
        
        # Determine border style
        border_style = self._theme.colors.error
        if error.severity == ErrorSeverity.WARNING:
            border_style = self._theme.colors.warning
            
        title = f"[bold {border_style}]Error[/bold {border_style}]"
        
        return Panel(
            content,
            title=title,
            border_style=border_style,
        )


class ErrorBuilder:
    """Builder for creating structured errors."""
    
    def __init__(self):
        self._category = ErrorCategory.UNKNOWN
        self._severity = ErrorSeverity.ERROR
        self._message = ""
        self._suggestion = ""
        self._did_you_mean: list[str] = []
        self._recoverable = False
        
    def category(self, cat: ErrorCategory) -> "ErrorBuilder":
        self._category = cat
        return self
        
    def severity(self, sev: ErrorSeverity) -> "ErrorBuilder":
        self._severity = sev
        return self
        
    def message(self, msg: str) -> "ErrorBuilder":
        self._message = msg
        return self
        
    def suggestion(self, sug: str) -> "ErrorBuilder":
        self._suggestion = sug
        return self
        
    def did_you_mean(self, *options: str) -> "ErrorBuilder":
        self._did_you_mean = list(options)
        return self
        
    def recoverable(self, flag: bool = True) -> "ErrorBuilder":
        self._recoverable = flag
        return self
        
    def build(self) -> ParsedError:
        return ParsedError(
            category=self._category,
            severity=self._severity,
            message=self._message,
            suggestion=self._suggestion,
            did_you_mean=self._did_you_mean,
            recoverable=self._recoverable,
        )
