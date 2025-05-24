#!/usr/bin/env python3
"""
Create ARC evaluation charts and visualizations.

This script generates comprehensive charts for ARC Easy evaluation results
including accuracy comparisons and performance metrics.
"""

import json
import os
from pathlib import Path

# Remove sys.path.append line for clean imports

try:
    from src.errors import setup_logging
    from src.visualization import generate_all_visualizations
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure to run from project root: python create_arc_charts.py")
    exit(1)


def main():
    # Load the ARC results
    results_file = "results/arc_full_evaluation.json"

    print(f"Loading results from {results_file}...")

    with open(results_file, "r") as f:
        data = json.load(f)

    # Transform data to match the expected format for visualization
    visualization_data = {"statistics": data["statistics"], "config": data["config"]}

    # Create output directory for ARC charts
    output_dir = "results/arc_charts"

    print(f"Generating visualizations in {output_dir}...")

    # Generate all visualizations
    generate_all_visualizations(visualization_data, output_dir)

    print("✅ Charts generated successfully!")
    print(f"📊 View charts in: {output_dir}/")

    # Show quick summary
    stats = data["statistics"]
    print(f"\n🎯 ARC Easy Results Summary:")
    print(
        f"DCBS:   {stats['dcbs']['accuracy']*100:.1f}% ({stats['dcbs']['correct']}/{stats['dcbs']['total']})"
    )
    print(
        f"Greedy: {stats['greedy']['accuracy']*100:.1f}% ({stats['greedy']['correct']}/{stats['greedy']['total']})"
    )
    print(
        f"Top-p:  {stats['top_p']['accuracy']*100:.1f}% ({stats['top_p']['correct']}/{stats['top_p']['total']})"
    )
    print(
        f"Random: {stats['random']['accuracy']*100:.1f}% ({stats['random']['correct']}/{stats['random']['total']})"
    )


if __name__ == "__main__":
    main()
