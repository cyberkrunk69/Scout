"""Tool Registry with contract support and versioning."""

from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
import json
import logging

from .actions import ActionType

logger = logging.getLogger(__name__)


class ToolContract:
    """Represents a tool contract with versioning."""
    
    def __init__(self, contract: dict):
        self.id = contract.get("id", "")
        self.version = contract.get("version", "1.0.0")
        self.name = contract.get("name", "")
        self.description = contract.get("description", "")
        self.input_schema = contract.get("input_schema", {})
        self.output_schema = contract.get("output_schema", {})
        self.success_conditions = contract.get("success_conditions", {})
        self.safety = contract.get("safety", {})
        self.cost_estimate = contract.get("cost_estimate", {})
        self.fallback_tools = contract.get("fallback_tools", [])
        self.capabilities = contract.get("capabilities", [])
        self.examples = contract.get("examples", [])
        self.inverse = contract.get("inverse")
        
    def is_compatible_with(self, other: "ToolContract") -> bool:
        """Check if versions are backward compatible (same major version)."""
        if self.id != other.id:
            return False
        self_major = self.version.split(".")[0]
        other_major = other.version.split(".")[0]
        return self_major == other_major
        
    def estimate_cost(self, input_data: dict) -> float:
        """Estimate cost based on input size."""
        cost_cfg = self.cost_estimate
        if cost_cfg.get("type") == "function":
            params = cost_cfg.get("params", {})
            if "input_size_kb" in params:
                # Simple linear estimation
                input_size = len(json.dumps(input_data)) / 1024
                multiplier = float(params["input_size_kb"].split(":")[1])
                return min(cost_cfg.get("max", 1.0), max(cost_cfg.get("min", 0.001), input_size * multiplier))
        return cost_cfg.get("min", 0.001)


class ExecutionToolRegistry:
    """Tool registry with contract support and versioning."""
    
    def __init__(self, contracts_path: Optional[Path] = None):
        self._mapping: Dict[ActionType, Dict[str, Any]] = {}
        self._contracts: Dict[str, ToolContract] = {}
        if contracts_path and contracts_path.exists():
            self._load_contracts(contracts_path)
        self._register_defaults()
        
    def _load_contracts(self, path: Path):
        """Load tool contracts from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
                for contract_data in data.get("tool_contracts", []):
                    contract = ToolContract(contract_data)
                    self._contracts[contract.id] = contract
                    logger.info(f"Loaded contract: {contract.id} v{contract.version}")
        except Exception as e:
            logger.warning(f"Failed to load contracts from {path}: {e}")
            
    def _register_defaults(self):
        self.register(ActionType.CREATE_FILE, "scout_create_file")
        self.register(ActionType.MODIFY_FILE, "scout_edit")
        self.register(ActionType.RUN_COMMAND, "bash")
        self.register(ActionType.READ_FILE, "scout_read_file")
        self.register(ActionType.GET_USER_INPUT, "scout_get_user_input")

    def register(self, action_type: ActionType, tool_name: str, adapter_func: Optional[Callable] = None):
        self._mapping[action_type] = {
            'tool': tool_name,
            'adapter': adapter_func
        }

    def get_tool_name(self, action_type: ActionType) -> Optional[str]:
        entry = self._mapping.get(action_type)
        return entry['tool'] if entry else None

    def get_adapter(self, action_type: ActionType) -> Optional[Callable]:
        entry = self._mapping.get(action_type)
        return entry['adapter'] if entry else None
        
    def get_contract(self, tool_id: str) -> Optional[ToolContract]:
        """Retrieve contract for a tool."""
        return self._contracts.get(tool_id)
        
    def find_fallback(self, tool_id: str) -> Optional[str]:
        """Find fallback tool if primary fails."""
        contract = self._contracts.get(tool_id)
        if contract and contract.fallback_tools:
            return contract.fallback_tools[0]
        return None
        
    def estimate_cost(self, tool_id: str, input_data: dict) -> float:
        """Estimate cost using contract's cost function."""
        contract = self._contracts.get(tool_id)
        if contract:
            return contract.estimate_cost(input_data)
        return 0.001  # Default estimate
        
    def get_examples(self, tool_id: str) -> List[dict]:
        """Get usage examples for LLM prompt engineering."""
        contract = self._contracts.get(tool_id)
        if contract:
            return contract.examples
        return []
