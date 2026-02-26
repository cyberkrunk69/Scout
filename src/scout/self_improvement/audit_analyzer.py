"""Self-Improvement Audit Analyzer - Metrics Calculation Module.

Computes baseline metrics from aggregated audit logs from multiple sources:
- Scout Core: ~/.scout/audit.jsonl (and archived .gz files)
- Audit Monitor DB: ~/.scout/audit_monitor.db
- Explorer Events: stored via /api/explorer-event endpoint
"""

import gzip
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# Default paths
AUDIT_JSONL = Path("~/.scout/audit.jsonl").expanduser()
METRICS_DB = Path("~/.scout/metrics.db").expanduser()
AUDIT_MONITOR_DB = Path("~/.scout/audit_monitor.db").expanduser()


def get_metrics_db_path() -> Path:
    """Get or create metrics database path."""
    METRICS_DB.parent.mkdir(parents=True, exist_ok=True)
    return METRICS_DB


def init_metrics_db() -> sqlite3.Connection:
    """Initialize metrics database with schema."""
    conn = sqlite3.connect(str(get_metrics_db_path()))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_type TEXT NOT NULL,
            metric_key TEXT NOT NULL,
            value REAL NOT NULL,
            computed_at TEXT NOT NULL
        )
    """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_metrics_type_key 
        ON metrics(metric_type, metric_key)
    """
    )
    conn.commit()
    return conn


def load_audit_events(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[dict]:
    """Load audit events from all sources.

    Args:
        since: Start datetime (default: 7 days ago)
        until: End datetime (default: now)

    Returns:
        List of event dictionaries with consistent schema
    """
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(days=7)
    if until is None:
        until = datetime.now(timezone.utc)

    events = []

    # 1. Load from Scout Core audit.jsonl
    events.extend(_load_from_audit_jsonl(since, until))

    # 2. Load from audit-monitor DB
    events.extend(_load_from_audit_monitor_db(since, until))

    # Sort by timestamp
    events.sort(key=lambda e: e.get("ts", ""))

    return events


def _load_from_audit_jsonl(since: datetime, until: datetime) -> list[dict]:
    """Load events from ~/.scout/audit.jsonl and archived .gz files."""
    events = []
    audit_dir = AUDIT_JSONL.parent

    # Load current audit.jsonl
    if AUDIT_JSONL.exists():
        events.extend(_parse_audit_file(AUDIT_JSONL, since, until))

    # Load archived .gz files
    for gz_file in audit_dir.glob("audit_*.jsonl.gz"):
        events.extend(_parse_audit_file(gz_file, since, until))

    return events


def _parse_audit_file(file_path: Path, since: datetime, until: datetime) -> list[dict]:
    """Parse audit file (jsonl or gz) and filter by date range."""
    events = []

    try:
        if file_path.suffix == ".gz":
            opener = gzip.open
        else:
            opener = open

        with opener(file_path, "rt") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    event_ts = _parse_timestamp(event.get("ts"))
                    if event_ts and since <= event_ts <= until:
                        events.append(event)
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return events


def _load_from_audit_monitor_db(since: datetime, until: datetime) -> list[dict]:
    """Load events from audit-monitor SQLite database."""
    events = []

    if not AUDIT_MONITOR_DB.exists():
        return events

    try:
        conn = sqlite3.connect(str(AUDIT_MONITOR_DB))
        conn.row_factory = sqlite3.Row

        cursor = conn.execute(
            """
            SELECT data FROM events
            WHERE timestamp >= ? AND timestamp <= ?
        """,
            (since.isoformat(), until.isoformat()),
        )

        for row in cursor.fetchall():
            try:
                data = json.loads(row["data"])
                data["_source"] = "audit_monitor"
                events.append(data)
            except json.JSONDecodeError:
                continue

        conn.close()
    except Exception:
        pass

    return events


def _parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO8601 timestamp string to datetime."""
    if not ts_str:
        return None

    try:
        # Handle various formats
        if ts_str.endswith("Z"):
            ts_str = ts_str[:-1] + "+00:00"
        return datetime.fromisoformat(ts_str)
    except ValueError:
        return None


def compute_frequency_metrics(events: list[dict]) -> dict:
    """Compute frequency-based metrics from events.

    Returns:
        Dictionary with:
        - most_searched_symbols: list of (symbol, count)
        - most_used_models: list of (model, count)
        - most_failed_validations: list of (reason, count)
        - most_clicked_nodes: list of (node_id, count)
    """
    # Most searched symbols (from tldr/deep nav events)
    symbol_counts: dict[str, int] = defaultdict(int)
    for event in events:
        event_type = event.get("event", "")
        if event_type in ("tldr", "deep", "nav"):
            # Try various fields where symbol might be stored
            symbol = (
                event.get("symbol")
                or event.get("data", {}).get("symbol")
                or event.get("file")  # fallback to file
            )
            if symbol:
                symbol_counts[symbol] += 1

    # Most used models (from llm_response events)
    model_counts: dict[str, int] = defaultdict(int)
    for event in events:
        if event.get("event") == "llm_response":
            model = event.get("model") or event.get("data", {}).get("model")
            if model:
                model_counts[model] += 1

    # Most failed validations
    validation_fail_counts: dict[str, int] = defaultdict(int)
    for event in events:
        if event.get("event") == "validation_fail":
            reason = event.get("reason") or event.get("data", {}).get("reason", "unknown")
            validation_fail_counts[reason] += 1

    # Most clicked nodes (Explorer UI)
    node_click_counts: dict[str, int] = defaultdict(int)
    for event in events:
        if event.get("event", "").startswith("explorer_"):
            if event.get("event") == "explorer_NODE_SELECTED":
                node_id = event.get("data", {}).get("nodeId") or event.get("nodeId")
                if node_id:
                    node_click_counts[node_id] += 1

    return {
        "most_searched_symbols": sorted(symbol_counts.items(), key=lambda x: -x[1])[:20],
        "most_used_models": sorted(model_counts.items(), key=lambda x: -x[1])[:10],
        "most_failed_validations": sorted(validation_fail_counts.items(), key=lambda x: -x[1])[:10],
        "most_clicked_nodes": sorted(node_click_counts.items(), key=lambda x: -x[1])[:20],
    }


def compute_cost_metrics(events: list[dict]) -> dict:
    """Compute cost-based metrics from events.

    Returns:
        Dictionary with:
        - hourly_spend: float (USD)
        - per_model_cost: dict of (model, total_cost)
        - session_costs: list of (session_id, cost)
    """
    hourly_total = 0.0
    model_costs: dict[str, float] = defaultdict(float)
    session_costs: dict[str, float] = defaultdict(float)

    for event in events:
        # Get cost from various possible locations
        cost = event.get("cost") or event.get("data", {}).get("cost") or 0.0

        if cost and isinstance(cost, (int, float)):
            hourly_total += cost

            # Per-model cost
            model = event.get("model") or event.get("data", {}).get("model")
            if model:
                model_costs[model] += cost

            # Per-session cost
            session_id = event.get("session_id") or event.get("data", {}).get("session_id")
            if session_id:
                session_costs[session_id] += cost

    return {
        "hourly_spend": round(hourly_total, 6),
        "per_model_cost": dict(sorted(model_costs.items(), key=lambda x: -x[1])),
        "session_costs": sorted(session_costs.items(), key=lambda x: -x[1])[:20],
    }


def compute_performance_metrics(events: list[dict]) -> dict:
    """Compute performance and quality metrics from events.

    Returns:
        Dictionary with:
        - avg_llm_latency_ms: float
        - validation_success_rate: float (0-1)
        - gate_pass_rate: float (0-1)
    """
    # LLM latency
    latencies = []
    for event in events:
        if event.get("event") == "llm_response":
            duration = event.get("duration_ms") or event.get("data", {}).get("duration_ms")
            if duration:
                latencies.append(duration)

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    # Validation success rate
    validation_total = 0
    validation_failures = 0
    for event in events:
        event_type = event.get("event", "")
        if event_type == "validation_fail":
            validation_total += 1
            validation_failures += 1
        elif event_type in ("validation_pass", "llm_response"):
            validation_total += 1

    validation_success_rate = (
        (validation_total - validation_failures) / validation_total if validation_total > 0 else 1.0
    )

    # Gate pass rate
    gate_compress = 0
    gate_escalate = 0
    for event in events:
        event_type = event.get("event", "")
        if event_type == "gate_compress":
            gate_compress += 1
        elif event_type == "gate_escalate":
            gate_escalate += 1

    total_gates = gate_compress + gate_escalate
    gate_pass_rate = gate_compress / total_gates if total_gates > 0 else 1.0

    return {
        "avg_llm_latency_ms": round(avg_latency, 2),
        "validation_success_rate": round(validation_success_rate, 4),
        "gate_pass_rate": round(gate_pass_rate, 4),
    }


def compute_and_store_metrics(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> dict:
    """Compute all metrics and store in metrics database.

    Returns:
        Dictionary with all computed metrics
    """
    if since is None:
        since = datetime.now(timezone.utc) - timedelta(days=7)
    if until is None:
        until = datetime.now(timezone.utc)

    # Load events
    events = load_audit_events(since, until)

    # Compute metrics
    frequency = compute_frequency_metrics(events)
    cost = compute_cost_metrics(events)
    performance = compute_performance_metrics(events)

    # Store in database
    conn = init_metrics_db()
    computed_at = datetime.now(timezone.utc).isoformat()

    # Store frequency metrics
    for symbol, count in frequency["most_searched_symbols"]:
        conn.execute(
            "INSERT INTO metrics (metric_type, metric_key, value, computed_at) VALUES (?, ?, ?, ?)",
            ("frequency", f"symbol:{symbol}", count, computed_at),
        )

    for model, count in frequency["most_used_models"]:
        conn.execute(
            "INSERT INTO metrics (metric_type, metric_key, value, computed_at) VALUES (?, ?, ?, ?)",
            ("frequency", f"model:{model}", count, computed_at),
        )

    # Store cost metrics
    conn.execute(
        "INSERT INTO metrics (metric_type, metric_key, value, computed_at) VALUES (?, ?, ?, ?)",
        ("cost", "hourly_spend", cost["hourly_spend"], computed_at),
    )

    for model, total_cost in cost["per_model_cost"].items():
        conn.execute(
            "INSERT INTO metrics (metric_type, metric_key, value, computed_at) VALUES (?, ?, ?, ?)",
            ("cost", f"model:{model}", total_cost, computed_at),
        )

    # Store performance metrics
    conn.execute(
        "INSERT INTO metrics (metric_type, metric_key, value, computed_at) VALUES (?, ?, ?, ?)",
        ("performance", "avg_llm_latency_ms", performance["avg_llm_latency_ms"], computed_at),
    )
    conn.execute(
        "INSERT INTO metrics (metric_type, metric_key, value, computed_at) VALUES (?, ?, ?, ?)",
        (
            "performance",
            "validation_success_rate",
            performance["validation_success_rate"],
            computed_at,
        ),
    )
    conn.execute(
        "INSERT INTO metrics (metric_type, metric_key, value, computed_at) VALUES (?, ?, ?, ?)",
        ("performance", "gate_pass_rate", performance["gate_pass_rate"], computed_at),
    )

    conn.commit()
    conn.close()

    return {
        "frequency": frequency,
        "cost": cost,
        "performance": performance,
        "event_count": len(events),
        "computed_at": computed_at,
    }


def get_latest_metrics() -> dict:
    """Retrieve the latest computed metrics from database."""
    if not METRICS_DB.exists():
        return {}

    conn = sqlite3.connect(str(METRICS_DB))
    conn.row_factory = sqlite3.Row

    # Get most recent computation time
    cursor = conn.execute(
        "SELECT DISTINCT computed_at FROM metrics ORDER BY computed_at DESC LIMIT 1"
    )
    row = cursor.fetchone()

    if not row:
        conn.close()
        return {}

    computed_at = row["computed_at"]

    # Get all metrics from that computation
    cursor = conn.execute(
        "SELECT metric_type, metric_key, value FROM metrics WHERE computed_at = ?", (computed_at,)
    )

    metrics = {"computed_at": computed_at, "frequency": {}, "cost": {}, "performance": {}}

    for row in cursor.fetchall():
        metric_type = row["metric_type"]
        metric_key = row["metric_key"]
        value = row["value"]

        if metric_type == "frequency":
            if metric_key.startswith("symbol:"):
                metrics["frequency"].setdefault("most_searched_symbols", []).append(
                    (metric_key[7:], value)
                )
            elif metric_key.startswith("model:"):
                metrics["frequency"].setdefault("most_used_models", []).append(
                    (metric_key[6:], value)
                )
        elif metric_type == "cost":
            if metric_key == "hourly_spend":
                metrics["cost"]["hourly_spend"] = value
            elif metric_key.startswith("model:"):
                metrics["cost"].setdefault("per_model_cost", {})[metric_key[6:]] = value
        elif metric_type == "performance":
            metrics["performance"][metric_key] = value

    # Sort the lists
    if "most_searched_symbols" in metrics["frequency"]:
        metrics["frequency"]["most_searched_symbols"] = sorted(
            metrics["frequency"]["most_searched_symbols"], key=lambda x: -x[1]
        )[:20]
    if "most_used_models" in metrics["frequency"]:
        metrics["frequency"]["most_used_models"] = sorted(
            metrics["frequency"]["most_used_models"], key=lambda x: -x[1]
        )[:10]
    if "per_model_cost" in metrics["cost"]:
        metrics["cost"]["per_model_cost"] = dict(
            sorted(metrics["cost"]["per_model_cost"].items(), key=lambda x: -x[1])
        )

    conn.close()
    return metrics
