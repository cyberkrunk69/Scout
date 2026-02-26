#!/usr/bin/env python
"""
REPL state machine - manages REPL state transitions and events.

Provides:
- State enumeration (IDLE, THINKING, EXECUTING, WAITING_INPUT, ERROR)
- Transition handlers
- Event bus for UI state changes
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional


class REPLState(Enum):
    """REPL execution states."""
    IDLE = auto()           # Waiting for user input
    THINKING = auto()       # Analyzing user request
    PLANNING = auto()       # Generating a plan
    EXECUTING = auto()      # Running a plan/tool
    WAITING_INPUT = auto() # Paused for user input
    ERROR = auto()          # Error state
    CANCELLED = auto()      # Operation cancelled


class REPLEvent(Enum):
    """REPL events that trigger state transitions."""
    INPUT_RECEIVED = auto()
    ANALYSIS_STARTED = auto()
    ANALYSIS_COMPLETE = auto()
    PLAN_STARTED = auto()
    PLAN_COMPLETE = auto()
    EXECUTION_STARTED = auto()
    EXECUTION_STEP = auto()
    EXECUTION_COMPLETE = auto()
    EXECUTION_FAILED = auto()
    INPUT_REQUESTED = auto()
    INPUT_PROVIDED = auto()
    ERROR_OCCURRED = auto()
    CANCEL_REQUESTED = auto()
    CANCEL_COMPLETE = auto()
    RESET = auto()


@dataclass
class StateTransition:
    """Records a state transition."""
    from_state: REPLState
    to_state: REPLState
    event: REPLEvent
    timestamp: float
    metadata: dict = field(default_factory=dict)


class EventBus:
    """Simple event bus for REPL state changes."""
    
    def __init__(self):
        self._listeners: dict[REPLEvent, list[Callable]] = {}
        self._all_listeners: list[Callable] = []
        
    def subscribe(self, event: REPLEvent, callback: Callable):
        """Subscribe to a specific event."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
        
    def unsubscribe(self, event: REPLEvent, callback: Callable):
        """Unsubscribe from an event."""
        if event in self._listeners and callback in self._listeners[event]:
            self._listeners[event].remove(callback)
            
    def subscribe_all(self, callback: Callable):
        """Subscribe to all events."""
        self._all_listeners.append(callback)
        
    def unsubscribe_all(self, callback: Callable):
        """Unsubscribe from all events."""
        if callback in self._all_listeners:
            self._all_listeners.remove(callback)
            
    def publish(self, event: REPLEvent, data: Any = None):
        """Publish an event to all subscribers."""
        # Call event-specific listeners
        if event in self._listeners:
            for callback in self._listeners[event]:
                try:
                    callback(event, data)
                except Exception:
                    pass
                    
        # Call all-listeners
        for callback in self._all_listeners:
            try:
                callback(event, data)
            except Exception:
                pass


class REPLStateMachine:
    """
    State machine for REPL execution.
    
    Manages state transitions and emits events for UI updates.
    """
    
    # Valid transitions: state -> set of events that can trigger transition
    TRANSITIONS = {
        REPLState.IDLE: {
            REPLEvent.INPUT_RECEIVED: REPLState.THINKING,
        },
        REPLState.THINKING: {
            REPLEvent.ANALYSIS_COMPLETE: REPLState.PLANNING,
            REPLEvent.ERROR_OCCURRED: REPLState.ERROR,
            REPLEvent.CANCEL_REQUESTED: REPLState.CANCELLED,
        },
        REPLState.PLANNING: {
            REPLEvent.PLAN_COMPLETE: REPLState.EXECUTING,
            REPLEvent.ERROR_OCCURRED: REPLState.ERROR,
            REPLEvent.CANCEL_REQUESTED: REPLState.CANCELLED,
        },
        REPLState.EXECUTING: {
            REPLEvent.EXECUTION_COMPLETE: REPLState.IDLE,
            REPLEvent.EXECUTION_FAILED: REPLState.ERROR,
            REPLEvent.INPUT_REQUESTED: REPLState.WAITING_INPUT,
            REPLEvent.CANCEL_REQUESTED: REPLState.CANCELLED,
        },
        REPLState.WAITING_INPUT: {
            REPLEvent.INPUT_PROVIDED: REPLState.EXECUTING,
            REPLEvent.CANCEL_REQUESTED: REPLState.CANCELLED,
        },
        REPLState.ERROR: {
            REPLEvent.RESET: REPLState.IDLE,
        },
        REPLState.CANCELLED: {
            REPLEvent.RESET: REPLState.IDLE,
        },
    }
    
    def __init__(self):
        self._current_state = REPLState.IDLE
        self._event_bus = EventBus()
        self._transition_history: list[StateTransition] = []
        self._current_task: Optional[asyncio.Task] = None
        self._metadata: dict = {}
        import time
        self._start_time = time.time()
        
    @property
    def state(self) -> REPLState:
        return self._current_state
    
    @property
    def event_bus(self) -> EventBus:
        return self._event_bus
    
    @property
    def is_busy(self) -> bool:
        """Check if REPL is currently processing."""
        return self._current_state in (
            REPLState.THINKING,
            REPLState.PLANNING,
            REPLState.EXECUTING,
            REPLState.WAITING_INPUT,
        )
        
    @property
    def can_accept_input(self) -> bool:
        """Check if REPL can accept new input."""
        return self._current_state in (
            REPLState.IDLE,
            REPLState.ERROR,
            REPLState.CANCELLED,
        )
        
    def transition(self, event: REPLEvent, metadata: dict = None) -> bool:
        """
        Attempt a state transition.
        
        Args:
            event: The event triggering the transition
            metadata: Optional metadata about the transition
            
        Returns:
            True if transition was successful, False otherwise
        """
        import time
        
        # Check if transition is valid
        if self._current_state not in self.TRANSITIONS:
            return False
            
        valid_events = self.TRANSITIONS[self._current_state]
        if event not in valid_events:
            return False
            
        # Record transition
        new_state = valid_events[event]
        transition = StateTransition(
            from_state=self._current_state,
            to_state=new_state,
            event=event,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._transition_history.append(transition)
        
        # Update state
        old_state = self._current_state
        self._current_state = new_state
        self._metadata.update(metadata or {})
        
        # Publish event
        self._event_bus.publish(event, {
            "old_state": old_state,
            "new_state": new_state,
            "metadata": metadata,
        })
        
        return True
        
    def force_state(self, state: REPLState):
        """Force a state change (for error handling)."""
        import time
        
        old_state = self._current_state
        self._current_state = state
        
        self._event_bus.publish(REPLEvent.ERROR_OCCURRED if state == REPLState.ERROR else REPLEvent.RESET, {
            "old_state": old_state,
            "new_state": state,
        })
        
    def get_history(self) -> list[StateTransition]:
        """Get transition history."""
        return self._transition_history.copy()
        
    def set_task(self, task: Optional[asyncio.Task]):
        """Set the current async task."""
        self._current_task = task
        
    def get_task(self) -> Optional[asyncio.Task]:
        """Get the current async task."""
        return self._current_task
        
    def cancel_current_task(self):
        """Cancel the current task if running."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            
    def reset(self):
        """Reset to idle state."""
        self.transition(REPLEvent.RESET)
        self._transition_history.clear()
        self._metadata.clear()
        import time
        self._start_time = time.time()
        
    def get_status(self) -> dict:
        """Get current status as dict."""
        return {
            "state": self._current_state.name,
            "is_busy": self.is_busy,
            "can_accept_input": self.can_accept_input,
            "transition_count": len(self._transition_history),
        }


class CommandQueue:
    """
    Queue for commands entered during execution.
    
    Allows users to queue commands while another command
    is executing, which will be processed after completion.
    """
    
    def __init__(self, max_size: int = 10):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._processing = False
        
    async def enqueue(self, command: str):
        """Add a command to the queue."""
        await self._queue.put(command)
        
    async def dequeue(self, timeout: float = 0.1) -> Optional[str]:
        """Get next command from queue."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
            
    def clear(self):
        """Clear all queued commands."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
    @property
    def size(self) -> int:
        return self._queue.qsize()
        
    @property
    def is_empty(self) -> bool:
        return self._queue.empty()


# Global state machine instance
_state_machine: Optional[REPLStateMachine] = None
_command_queue: Optional[CommandQueue] = None


def get_state_machine() -> REPLStateMachine:
    """Get the global REPL state machine."""
    global _state_machine
    if _state_machine is None:
        _state_machine = REPLStateMachine()
    return _state_machine


def get_command_queue() -> CommandQueue:
    """Get the global command queue."""
    global _command_queue
    if _command_queue is None:
        _command_queue = CommandQueue()
    return _command_queue


def reset_state():
    """Reset global state."""
    global _state_machine, _command_queue
    if _state_machine:
        _state_machine.reset()
    if _command_queue:
        _command_queue.clear()
