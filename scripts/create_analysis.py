#!/usr/bin/env python3
"""
Create comprehensive analysis with figures and summary for the full ARC evaluation.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.create_final_analysis import (
    load_results,
    create_accuracy_comparison_chart,
    create_timing_comparison_chart,
    create_cache_comparison_chart,
    create_optimization_summary_chart,
    create_detailed_analysis_report
)


def run_analysis(results_dir=None, output_dir=None):
    """
    Generate complete analysis with figures and reports.
    
    Args:
        results_dir: Directory containing evaluation results
        output_dir: Directory to save analysis outputs
    """
    if output_dir is None:
        output_dir = Path('results/final_analysis')
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load all results
    print("Loading evaluation results...")
    results = load_results()
    
    # Generate figures
    print("Creating accuracy comparison chart...")
    create_accuracy_comparison_chart(results, output_dir)
    
    print("Creating timing comparison chart...")
    create_timing_comparison_chart(results, output_dir)
    
    print("Creating cache comparison chart...")
    create_cache_comparison_chart(results, output_dir)
    
    print("Creating optimization summary chart...")
    create_optimization_summary_chart(results, output_dir)
    
    # Generate detailed report
    print("Creating detailed analysis report...")
    create_detailed_analysis_report(results, output_dir)
    
    print(f"\nAnalysis complete. Files saved to: {output_dir}")
    print("Generated files:")
    print("   - accuracy_comparison.png")
    print("   - timing_comparison.png") 
    print("   - cache_comparison.png")
    print("   - optimization_summary.png")
    print("   - COMPLETE_ANALYSIS_REPORT.md")
    
    # Show key findings
    full_stats = results['full_no_cache']['statistics']
    greedy_acc = full_stats.get('greedy', {}).get('accuracy', 0)
    dcbs_acc = full_stats.get('dcbs', {}).get('accuracy', 0)
    greedy_time = full_stats.get('greedy', {}).get('avg_time_ms', 0)
    dcbs_time = full_stats.get('dcbs', {}).get('avg_time_ms', 0)
    
    print(f"\nKey findings:")
    print(f"   DCBS Accuracy: {dcbs_acc:.1%} vs Greedy {greedy_acc:.1%}")
    print(f"   DCBS Time: {dcbs_time:.0f}ms vs Greedy {greedy_time:.0f}ms")
    print(f"   Performance: {((greedy_time - dcbs_time) / greedy_time * 100):+.1f}% vs Greedy")
    print(f"   Dataset: 2,946 questions (complete ARC Easy)")
    
    return 0


def main():
    """Parse arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description="Create comprehensive analysis of DCBS evaluation results"
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory containing evaluation results (default: results/)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save analysis outputs (default: results/final_analysis/)"
    )
    
    args = parser.parse_args()
    return run_analysis(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main() 