# TODO: This module is not yet integrated into the main application.
# It is planned for future use as part of the Plan Execution framework.
# See ADR-007 for design context.
from __future__ import annotations
"""
Plan I/O utilities for Scout plan storage.

Provides functions to read, write, list, delete, and validate plan files.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

PLAN_STORAGE_DIR = Path("docs/plans")
BY_DATE_DIR = PLAN_STORAGE_DIR / "by-date"
BY_MODEL_DIR = PLAN_STORAGE_DIR / "by-model"
INDEX_FILE = PLAN_STORAGE_DIR / "index.json"


def generate_plan_id(content: str, request: str) -> str:
    """Generate a unique SHA256 hash for a plan."""
    combined = f"{request}:{content[:500]}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def get_date_path(plan: dict) -> Path:
    """Get the date-based path for a plan (YYYY-MM format)."""
    created = plan.get("created_at", datetime.utcnow().isoformat())
    date_part = created[:7]
    return BY_DATE_DIR / date_part


def get_model_path(model: str) -> Path:
    """Get the model-based path for a plan."""
    return BY_MODEL_DIR / model


def read_plan(plan_id: str) -> Optional[dict]:
    """Read a plan from storage by ID."""
    for plan_file in PLAN_STORAGE_DIR.glob("*.json"):
        try:
            with open(plan_file, "r") as f:
                plan = json.load(f)
                if plan.get("id") == plan_id:
                    return plan
        except (json.JSONDecodeError, IOError):
            continue
    return None


def write_plan(plan: dict, overwrite: bool = False) -> Path:
    """
    Write a plan to storage.

    Returns the path where the plan was saved.
    """
    plan_id = plan.get("id") or generate_plan_id(
        plan.get("plan", ""), plan.get("request", "")
    )
    plan["id"] = plan_id

    if not plan.get("created_at"):
        plan["created_at"] = datetime.utcnow().isoformat()
    plan["updated_at"] = datetime.utcnow().isoformat()

    date_path = get_date_path(plan)
    date_path.mkdir(parents=True, exist_ok=True)

    model = plan.get("model", "unknown")
    model_path = get_model_path(model)
    model_path.mkdir(parents=True, exist_ok=True)

    filename = f"{plan_id}.json"

    date_file = date_path / filename
    model_file = model_path / filename

    if not overwrite and (date_file.exists() or model_file.exists()):
        raise FileExistsError(f"Plan {plan_id} already exists")

    with open(date_file, "w") as f:
        json.dump(plan, f, indent=2)

    if model_file.parent != date_file.parent:
        with open(model_file, "w") as f:
            json.dump(plan, f, indent=2)

    _update_index(plan)

    return date_file


def list_plans(path: Optional[Path] = None) -> list[dict]:
    """List all plans in a directory."""
    target = path or PLAN_STORAGE_DIR
    plans = []

    for plan_file in target.glob("**/*.json"):
        if plan_file.name == "index.json" or plan_file.name == "schema.json":
            continue
        try:
            with open(plan_file, "r") as f:
                plans.append(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue

    return sorted(plans, key=lambda p: p.get("created_at", ""), reverse=True)


def delete_plan(plan_id: str) -> bool:
    """Delete a plan from storage."""
    deleted = False

    for plan_file in PLAN_STORAGE_DIR.glob(f"**/{plan_id}.json"):
        if plan_file.name in ("index.json", "schema.json"):
            continue
        plan_file.unlink()
        deleted = True

    _update_index()
    return deleted


def validate_plan(plan_data: dict) -> tuple[bool, list[str]]:
    """Validate a plan against the schema."""
    errors = []

    required = ["id", "request", "plan", "created_at", "model", "tokens", "cost"]
    for field in required:
        if field not in plan_data:
            errors.append(f"Missing required field: {field}")

    if "tokens" in plan_data and not isinstance(plan_data["tokens"], int):
        errors.append("Field 'tokens' must be an integer")

    if "cost" in plan_data and not isinstance(plan_data["cost"], (int, float)):
        errors.append("Field 'cost' must be a number")

    return len(errors) == 0, errors


def _update_index(plan: Optional[dict] = None):
    """Update the master index file."""
    plans = list_plans()

    index = {
        "total_plans": len(plans),
        "last_updated": datetime.utcnow().isoformat(),
        "by_model": {},
        "by_date": {},
        "by_id": {},
    }

    for p in plans:
        model = p.get("model", "unknown")
        date = p.get("created_at", "")[:7]
        pid = p.get("id", "")

        if model not in index["by_model"]:
            index["by_model"][model] = []
        index["by_model"][model].append(pid)

        if date not in index["by_date"]:
            index["by_date"][date] = []
        index["by_date"][date].append(pid)

        index["by_id"][pid] = {
            "request": p.get("request", "")[:100],
            "created_at": p.get("created_at", ""),
            "model": model,
        }

    with open(INDEX_FILE, "w") as f:
        json.dump(index, f, indent=2)


def get_plan_count() -> int:
    """Get total number of stored plans."""
    return len(list_plans())
