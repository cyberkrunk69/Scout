"""
Internal constants for trust subsystem.

Philosophy: Private internal params separated from user-facing config
"""

from __future__ import annotations

from pathlib import Path

# === Private BM25 constants – internal, not exposed to users ===
_BM25_CONSTANTS = {
    "base": 50,
    "range": 30,
    "clarity_max": 30,
    "exact_boost": 1.03,
    "class_boost": 1.01,
}

# Export for direct import elsewhere
BM25_BASE = _BM25_CONSTANTS["base"]
BM25_RANGE = _BM25_CONSTANTS["range"]
BM25_CLARITY_MAX = _BM25_CONSTANTS["clarity_max"]
BM25_EXACT_BOOST = _BM25_CONSTANTS["exact_boost"]
BM25_CLASS_BOOST = _BM25_CONSTANTS["class_boost"]

# === User‑facing defaults (may be overridden by ScoutConfig) ===
DEFAULT_PENALTIES = {
    "trusted": 0.0,
    "partial": 0.1,
    "no_checksum": 0.2,
    "stale": 0.4,
    "missing": 0.5,
    "unverified": 0.3,
}
DEFAULT_MIN_CONFIDENCE = 70
DEFAULT_FALLBACKS = {
    "HALLUCINATED_PATH": 0,
    "HALLUCINATED_SYMBOL": 20,
    "WRONG_LINE": 40,
    "LOW_CONFIDENCE": 50,
    "VALID": 100,
}

# === Audit settings ===
DEFAULT_AUDIT_LEVEL = "INFO"          # "DEBUG", "INFO", "WARN", "ERROR"
DEFAULT_AUDIT_SAMPLE_RATE = 0.1       # 10% sample for DEBUG events
AUDIT_LOG_FILENAME = "audit.jsonl"

# === Database ===
TRUST_DB_FILENAME = "trust.db"

# === Learning ===
DEFAULT_LEARNER_MIN_SAMPLES = 10
DEFAULT_LEARNER_ADJUSTMENT_RATE = 0.05   # used in simple update (we'll replace with Bayesian)
DEFAULT_LEARNER_CONFIDENCE_THRESHOLD = 0.8

# === Auto‑repair ===
DEFAULT_AUTO_REPAIR_ENABLED = True
DEFAULT_AUTO_REPAIR_THRESHOLD = 5        # refresh after 5 queries
