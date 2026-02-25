"""Mapper for converting StructuredStep objects to tool invocation dictionaries."""
import logging
from typing import Dict, Any, Optional

from .actions import StructuredStep, ActionType
from .registry import ExecutionToolRegistry

logger = logging.getLogger(__name__)


class StepToToolMapper:
    """Maps StructuredStep objects to generic tool invocation dictionaries."""

    def __init__(self, registry: ExecutionToolRegistry):
        """Initialize mapper with an execution tool registry.

        Args:
            registry: The ExecutionToolRegistry to look up tool names.
        """
        self._registry = registry

    def map(self, step: StructuredStep) -> Optional[Dict[str, Any]]:
        """Transform a StructuredStep into a tool invocation dictionary.

        Args:
            step: The StructuredStep to transform.

        Returns:
            A dictionary with 'tool_name' and 'arguments' keys, or None if
            the step cannot be mapped.
        """
        # Get tool name from registry
        tool_name = self._registry.get_tool_name(step.action_type)
        if tool_name is None:
            logger.error(f"No tool registered for action type: {step.action_type}")
            return None

        # Build arguments based on action type
        args = self._build_arguments(step)
        if args is None:
            return None

        return {
            'tool_name': tool_name,
            'arguments': args
        }

    def _build_arguments(self, step: StructuredStep) -> Optional[Dict[str, Any]]:
        """Build arguments dictionary based on action type.

        Args:
            step: The StructuredStep to build arguments for.

        Returns:
            Arguments dictionary or None if validation fails.
        """
        action_type = step.action_type

        if action_type == ActionType.CREATE_FILE:
            # Validate file_path for file operations
            if step.file_path is None:
                logger.error("CREATE_FILE action requires file_path")
                return None
            return {
                'path': step.file_path,
                'content': step.content or ''
            }

        elif action_type == ActionType.MODIFY_FILE:
            if step.file_path is None:
                logger.error("MODIFY_FILE action requires file_path")
                return None
            return {
                'path': step.file_path,
                'content': step.content or ''
            }

        elif action_type == ActionType.DELETE_FILE:
            if step.file_path is None:
                logger.error("DELETE_FILE action requires file_path")
                return None
            return {
                'path': step.file_path
            }

        elif action_type == ActionType.RUN_COMMAND:
            # Validate command for RUN operations
            if step.command is None:
                logger.error("RUN_COMMAND action requires command")
                return None
            return {
                'command': step.command
            }

        elif action_type == ActionType.READ_FILE:
            if step.file_path is None:
                logger.error("READ_FILE action requires file_path")
                return None
            return {
                'path': step.file_path
            }

        elif action_type == ActionType.UNKNOWN:
            logger.warning(f"UNKNOWN action type for step: {step.description}")
            return None

        # Fallback for any unhandled action types
        logger.warning(f"Unhandled action type: {action_type}")
        return None
