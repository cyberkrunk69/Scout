"""Scout Analysis Package - Hotspot detection and impact analysis."""

from scout.analysis.hotspots import (
    compute_hotspot_score,
    get_error_rates,
    get_file_churn,
    scout_hotspots,
)

__all__ = [
    "compute_hotspot_score",
    "get_error_rates",
    "get_file_churn",
    "scout_hotspots",
]
