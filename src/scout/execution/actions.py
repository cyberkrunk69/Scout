from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ActionType(Enum):
    CREATE_FILE = "create_file"
    MODIFY_FILE = "modify_file"
    DELETE_FILE = "delete_file"
    RUN_COMMAND = "run_command"
    READ_FILE = "read_file"
    GET_USER_INPUT = "get_user_input"
    BROWSER_ACT = "browser_act"
    UNKNOWN = "unknown"


@dataclass
class StructuredStep:
    action_type: ActionType
    description: str
    file_path: Optional[str] = None
    content: Optional[str] = None
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    working_directory: Optional[str] = None
    rollback_info: Optional[dict] = None
    # Enhanced fields for execution engine
    step_id: int = 0
    depends_on: List[int] = field(default_factory=list)
    success_conditions: Optional[Dict[str, Any]] = None
    success_condition_type: str = "programmatic"
    retry_count: int = 0
    max_retries: int = 2
    timeout_seconds: int = 300
    parallel_group: Optional[str] = None
    rollback_on_fail: bool = True
    input_prompt: Optional[str] = None
    input_schema: Optional[Dict] = None
    input_source: str = "repl"

    def __post_init__(self):
        file_related_types = (ActionType.CREATE_FILE, ActionType.MODIFY_FILE, ActionType.DELETE_FILE)
        if self.action_type in file_related_types and self.file_path is None:
            logger.warning(f"File-related action {self.action_type} missing file_path, setting to UNKNOWN")
            self.action_type = ActionType.UNKNOWN


@dataclass
class StructuredPlan:
    steps: List[StructuredStep]
    raw_plan: str
    summary: str = ""


@dataclass
class StepResult:
    step_id: int
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost: float = 0.0
    duration_ms: int = 0


@dataclass
class ExecutionResult:
    steps_completed: int
    steps_failed: int
    total_cost: float
    total_duration: float
    discoveries: List[Dict] = field(default_factory=list)
    rollback_commands: List[str] = field(default_factory=list)


@dataclass
class WebStep:
    """Represents a single step in a web automation task.
    
    Execution Traces:
    - Happy: All fields valid, executor can map to browser_act call
    - Failure: Missing required field (action), executor raises error
    - Edge: Unknown action type, executor logs warning and uses default "click"
    """
    action: str  # "navigate", "click", "type", "extract", "wait", "assert"
    target: Optional[str] = None  # natural language description of target element
    value: Optional[str] = None   # value to type or extract
    url: Optional[str] = None     # for navigate action
    condition: Optional[str] = None  # for assert/wait (e.g., "element with text 'Success' appears")
    
    # Execution metadata
    step_index: int = 0
    max_retries: int = 1
    timeout_seconds: int = 30
    
    def __post_init__(self):
        """Validate and set defaults for WebStep.
        
        Execution Traces:
        - Happy: action is valid string
        - Failure: action is empty string, defaults to "click"
        - Edge: unknown action type, logs warning
        """
        if not self.action:
            logger.warning("WebStep missing action, defaulting to 'click'")
            self.action = "click"
        
        valid_actions = {"navigate", "click", "type", "extract", "wait", "assert", "scroll"}
        if self.action.lower() not in valid_actions:
            logger.warning(f"WebStep unknown action '{self.action}', proceeding anyway")
    
    def to_browser_act_params(self) -> Dict[str, Any]:
        """Convert WebStep to browser_act() parameters.
        
        Returns dict suitable for passing to browser_act function.
        """
        params = {
            "action": self.action,
        }
        if self.target:
            params["goal"] = self.target
        if self.value:
            params["value"] = self.value
        if self.url:
            params["url"] = self.url
        return params


@dataclass
class PlanContext:
    """State passed between steps during plan execution.
    
    Execution Traces:
    - Happy: Context accumulates extracted data across steps
    - Failure: Step fails, context preserved for debugging
    - Edge: Session expires mid-plan, context cleared
    """
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    current_url: Optional[str] = None
    cookies: Optional[Dict] = None
    session_id: Optional[str] = None
    plan_id: Optional[str] = None


@dataclass
class StepResult:
    """Result of executing a single web step."""
    step_index: int
    action: str
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    failure_reason: Optional[str] = None  # Detailed reason for failure
    retry_count: int = 0
    cost: float = 0.0  # Cost in USD from browser agent
    target: Optional[str] = None  # Target element for the action


@dataclass
class PlanResult:
    """Result of executing a complete plan."""
    success: bool
    plan_id: Optional[str]
    steps_executed: int
    steps_failed: int
    step_results: List[StepResult]
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    total_cost: float = 0.0
    total_duration_ms: int = 0
