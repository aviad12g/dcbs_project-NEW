#!/usr/bin/env python3
"""
Create comprehensive analysis with figures and summary for evaluation results.
"""

import argparse
import os
import sys
from pathlib import Path
import json
import glob

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.visualization import generate_all_visualizations
from src.statistical_analysis import comprehensive_statistical_analysis


def load_results(results_dir=None):
    """Load the latest evaluation results."""
    if results_dir is None:
        results_dir = Path('results')
    else:
        results_dir = Path(results_dir)
    
    # Find the latest results file
    result_files = list(results_dir.glob('evaluation_results_*.json'))
    if not result_files:
        raise FileNotFoundError("No evaluation results found")
    
    latest_file = max(result_files, key=os.path.getctime)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results


def create_detailed_analysis_report(results, output_dir):
    """Create a detailed analysis report."""
    report_path = output_dir / 'ANALYSIS_REPORT.md'
    
    # Perform statistical analysis
    stats_analysis = comprehensive_statistical_analysis(results)
    
    with open(report_path, 'w') as f:
        f.write("# DCBS Evaluation Analysis Report\n\n")
        
        # Summary statistics
        f.write("## Summary Statistics\n\n")
        f.write("| Method | Accuracy | Correct/Total | 95% CI |\n")
        f.write("|--------|----------|---------------|--------|\n")
        
        for method, stats in results['statistics'].items():
            ci = stats.get('confidence_interval', (0, 0))
            f.write(f"| {method.upper()} | {stats['accuracy']:.1f}% | "
                   f"{stats['correct']}/{stats['total']} | "
                   f"({ci[0]:.1f}%, {ci[1]:.1f}%) |\n")
        
        # Statistical analysis
        f.write("\n## Statistical Analysis\n\n")
        
        for comparison, result in stats_analysis['pairwise_comparisons'].items():
            f.write(f"### {comparison}\n")
            f.write(f"- p-value: {result['p_value']:.6f}\n")
            f.write(f"- Corrected p-value: {result['corrected_p_value']:.6f}\n")
            f.write(f"- Effect size: {result['effect_size']:.3f} ({result['effect_interpretation']})\n")
            f.write(f"- Significant: {'Yes' if result['significant_corrected'] else 'No'}\n\n")
        
        f.write("## Recommendations\n\n")
        for rec in stats_analysis['recommendations']:
            f.write(f"- {rec}\n")
    
    print(f"Analysis report saved to: {report_path}")


def run_analysis(results_dir=None, output_dir=None):
    """
    Generate complete analysis with figures and reports.
    """
    if output_dir is None:
        output_dir = Path('results')
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    print("Loading evaluation results...")
    results = load_results(results_dir)
    
    # Generate visualizations using existing function
    print("Creating charts and visualizations...")
    generate_all_visualizations(results, str(output_dir))
    
    # Generate detailed report
    print("Creating detailed analysis report...")
    create_detailed_analysis_report(results, output_dir)
    
    print(f"\nAnalysis complete. Files saved to: {output_dir}")
    
    # Show key findings
    statistics = results['statistics']
    methods = list(statistics.keys())
    best_method = max(methods, key=lambda m: statistics[m]['accuracy'])
    
    print(f"\nKey findings:")
    for method, stats in statistics.items():
        symbol = "🥇" if method == best_method else "  "
        print(f"   {symbol} {method.upper()}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    
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
        help="Directory to save analysis outputs (default: results/)"
    )
    
    args = parser.parse_args()
    return run_analysis(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main() 