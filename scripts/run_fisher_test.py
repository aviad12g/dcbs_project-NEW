#!/usr/bin/env python3
"""
Fisher's Exact Test for statistical significance between Greedy and DCBS sampling methods.

This script loads evaluation results and performs a rigorous statistical
comparison using Fisher's Exact Test to determine if the performance difference
between sampling methods is statistically significant.
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fisher_exact_test import (
    load_evaluation_results,
    extract_method_stats,
    perform_fisher_exact_test,
    interpret_p_value
)


def run_fisher_test(results_path, method1="greedy", method2="dcbs", output_path=None):
    """
    Run Fisher's Exact Test analysis on evaluation results.
    
    Args:
        results_path: Path to the evaluation results JSON file
        method1: First method to compare (default: greedy)
        method2: Second method to compare (default: dcbs)
        output_path: Path to save results (default: auto-generated)
    """
    print(f"Loading evaluation results from: {results_path}")
    
    results = load_evaluation_results(results_path)
    
    # Extract statistics for both methods
    method1_correct, method1_total, method1_accuracy = extract_method_stats(results, method1)
    method2_correct, method2_total, method2_accuracy = extract_method_stats(results, method2)
    
    # Calculate actual percentages
    method1_pct = (method1_correct / method1_total) * 100
    method2_pct = (method2_correct / method2_total) * 100
    
    print("\nEvaluation Statistics:")
    print("=" * 50)
    print(f"{method1}:  {method1_correct:,}/{method1_total:,} correct ({method1_pct:.1f}%)")
    print(f"{method2}:    {method2_correct:,}/{method2_total:,} correct ({method2_pct:.1f}%)")
    print(f"Dataset: {method1_total:,} questions")
    
    # Perform Fisher's Exact Test
    print("\nFisher's Exact Test Analysis:")
    print("=" * 50)
    
    odds_ratio, p_value = perform_fisher_exact_test(
        method1_correct, method1_total, method2_correct, method2_total
    )
    
    # Calculate effect size (difference in proportions)
    method1_prop = method1_correct / method1_total
    method2_prop = method2_correct / method2_total
    prop_diff = method1_prop - method2_prop
    
    print(f"Contingency Table:")
    print(f"                Correct    Incorrect    Total")
    print(f"{method1}:         {method1_correct:,}        {method1_total - method1_correct:,}       {method1_total:,}")
    print(f"{method2}:           {method2_correct:,}        {method2_total - method2_correct:,}       {method2_total:,}")
    print()
    print(f"Odds Ratio:     {odds_ratio:.4f}")
    print(f"P-value:        {p_value:.6f}")
    print(f"Significance:   {interpret_p_value(p_value)}")
    print(f"Effect Size:    {prop_diff:.4f} ({prop_diff*100:+.2f} percentage points)")
    
    # Statistical interpretation
    print("\nStatistical Interpretation:")
    print("=" * 50)
    
    if p_value < 0.05:
        if method1_accuracy > method2_accuracy:
            conclusion = f"{method1} sampling performs significantly better than {method2}"
        else:
            conclusion = f"{method2} performs significantly better than {method1} sampling"
    else:
        conclusion = f"No statistically significant difference between {method1} and {method2}"
    
    print(f"Conclusion: {conclusion}")
    print(f"Confidence: Based on {method1_total:,} samples with α = 0.05")
    
    # Save results to file
    if output_path is None:
        output_path = f"results/fisher_exact_{method1}_vs_{method2}.txt"
    
    with open(output_path, 'w') as f:
        f.write(f"Fisher's Exact Test: {method1} vs {method2} Sampling\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: ({method1_total:,} questions)\n")
        f.write(f"{method1}:  {method1_correct:,}/{method1_total:,} correct ({method1_pct:.1f}%)\n")
        f.write(f"{method2}:    {method2_correct:,}/{method2_total:,} correct ({method2_pct:.1f}%)\n\n")
        f.write("Statistical Test Results:\n")
        f.write(f"Odds Ratio: {odds_ratio:.4f}\n")
        f.write(f"P-value: {p_value:.6f}\n")
        f.write(f"Significance: {interpret_p_value(p_value)}\n")
        f.write(f"Effect Size: {prop_diff*100:+.2f} percentage points\n\n")
        f.write(f"Conclusion: {conclusion}\n")
    
    print(f"\nResults saved to: {output_path}")
    return 0


def main():
    """Parse arguments and run Fisher's Exact Test analysis."""
    parser = argparse.ArgumentParser(
        description="Run Fisher's Exact Test on evaluation results"
    )
    parser.add_argument(
        "--results", type=str, required=True, 
        help="Path to evaluation results JSON file"
    )
    parser.add_argument(
        "--method1", type=str, default="greedy",
        help="First method to compare (default: greedy)"
    )
    parser.add_argument(
        "--method2", type=str, default="dcbs",
        help="Second method to compare (default: dcbs)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results (default: auto-generated)"
    )
    
    args = parser.parse_args()
    return run_fisher_test(args.results, args.method1, args.method2, args.output)


if __name__ == "__main__":
    sys.exit(main()) 