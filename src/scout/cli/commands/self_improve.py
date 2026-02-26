#!/usr/bin/env python
"""Self-improve command wrapper for enhanced CLI.

This wraps the self-improvement functionality without importing from legacy cli/.
"""

import argparse
import sys


def run_self_improve(
    tool: str = "scout_plan",
    days: int = 7,
    dry_run: bool = False,
    budget_threshold: float = 10.0,
) -> int:
    """
    Run self-improvement analysis.
    
    This is a simplified wrapper that calls the underlying modules.
    """
    from scout.self_improvement import (
        analyze_improvements,
        get_improvement_stats,
    )
    
    print(f"Analyzing validation data for tool: {tool}")
    print(f"Analysis window: {days} days")
    print(f"Budget threshold: ${budget_threshold}/hour")
    
    if dry_run:
        print("[DRY RUN] Would generate suggestions without submitting")
    
    # Get stats
    stats = get_improvement_stats(days=days)
    print(f"\nImprovement stats:")
    print(f"  Total improvements: {stats.get('total_improvements', 0)}")
    print(f"  Success rate: {stats.get('success_rate', 0):.1%}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Self-improvement pilot")
    parser.add_argument("--tool", default="scout_plan", help="Tool to analyze")
    parser.add_argument("--days", type=int, default=7, help="Analysis window in days")
    parser.add_argument("--dry-run", action="store_true", help="Generate without submitting")
    parser.add_argument("--budget-threshold", type=float, default=10.0, help="Max $/hour")
    
    args = parser.parse_args()
    return run_self_improve(
        tool=args.tool,
        days=args.days,
        dry_run=args.dry_run,
        budget_threshold=args.budget_threshold,
    )


if __name__ == "__main__":
    sys.exit(main())
