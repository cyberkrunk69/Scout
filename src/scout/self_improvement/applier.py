"""Self-Improvement Applier - Advisory Application Module.

Generates change proposal files for approved improvements.
The spaceship suggests; humans apply.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from scout.audit import AuditLog


def apply_improvement(recommendation) -> dict:
    """Generate an advisory change proposal file.
    
    The spaceship suggests; humans apply.
    """
    audit = AuditLog()
    
    # Generate proposal filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    proposal_dir = Path("docs/self_improvement_proposals")
    proposal_dir.mkdir(exist_ok=True)
    
    proposal_path = proposal_dir / f"{recommendation.tool_name}_{timestamp}.json"
    
    # Build proposal structure
    proposal = {
        "id": f"SI-{timestamp}",
        "tool_name": recommendation.tool_name,
        "action": recommendation.action,
        "reason": recommendation.reason,
        "suggestion": recommendation.suggestion,
        "evidence": recommendation.evidence,
        "generated_at": datetime.now().isoformat(),
        "status": "pending_human_review",
        "apply_instructions": _get_apply_instructions(recommendation)
    }
    
    # Write proposal file
    proposal_path.write_text(json.dumps(proposal, indent=2))
    
    # Log the proposal
    audit.log("self_improvement_proposal_created",
        tool=recommendation.tool_name,
        action=recommendation.action,
        proposal_path=str(proposal_path)
    )
    
    return {
        "status": "proposal_created",
        "proposal_path": str(proposal_path),
        "proposal_id": proposal["id"]
    }


def _get_apply_instructions(recommendation) -> dict:
    """Generate instructions for applying the approved change."""
    instructions = {
        "update_prompt": {
            "file": "scout/cli/plan.py",
            "target": "system_prompt around line 2094",
            "action": "Add JSON schema example to prompt",
            "backup": "Create backup before editing",
            "test": "Run scout_plan with sample input to verify"
        },
        "add_validator": {
            "file": "scout/tools/__init__.py",
            "target": "TOOL_METADATA",
            "action": "Add validator to tool's validator list",
            "test": "Run validation pipeline on tool output"
        },
        "adjust_metadata": {
            "file": "scout/tools/__init__.py",
            "target": "TOOL_METADATA",
            "action": "Adjust cost_tier or timeout_seconds",
            "test": "Run tool and verify behavior"
        },
        "escalate_to_human": {
            "action": "Manual investigation required",
            "suggestion": "Review raw validation errors in audit log"
        }
    }
    
    return instructions.get(recommendation.action, {"action": "unknown"})
