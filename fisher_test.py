#!/usr/bin/env python3
"""
Fisher's Exact Test for Statistical Significance Analysis

This script performs Fisher's exact test to determine statistical significance
between different sampling methods in the DCBS evaluation results.
"""

import json
import sys
from pathlib import Path
from scipy.stats import fisher_exact
from typing import Dict, List, Tuple
import numpy as np


def load_latest_results(results_dir: str = "results") -> Dict:
    """Load the most recent evaluation results."""
    results_path = Path(results_dir)
    
    # Find the latest results file
    json_files = list(results_path.glob("evaluation_results_*.json"))
    if not json_files:
        print("No evaluation results found in results directory")
        sys.exit(1)
    
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def extract_method_statistics(results: Dict) -> Dict[str, Tuple[int, int]]:
    """Extract correct/total counts for each method."""
    statistics = results.get("statistics", {})
    method_stats = {}
    
    for method, stats in statistics.items():
        correct = stats.get("correct", 0)
        total = stats.get("total", 0)
        method_stats[method] = (correct, total)
        
    return method_stats


def perform_fisher_test(method1_stats: Tuple[int, int], 
                       method2_stats: Tuple[int, int],
                       method1_name: str, 
                       method2_name: str) -> Dict:
    """
    Perform Fisher's exact test between two methods.
    
    Returns:
        Dictionary with test results including p-value, odds ratio, etc.
    """
    correct1, total1 = method1_stats
    correct2, total2 = method2_stats
    
    incorrect1 = total1 - correct1
    incorrect2 = total2 - correct2
    
    # Create contingency table:
    # [[correct1, incorrect1],
    #  [correct2, incorrect2]]
    contingency_table = [[correct1, incorrect1], 
                        [correct2, incorrect2]]
    
    # Perform Fisher's exact test
    odds_ratio, p_value = fisher_exact(contingency_table, alternative='two-sided')
    
    # Calculate confidence interval for odds ratio
    log_or = np.log(odds_ratio) if odds_ratio > 0 else np.nan
    se_log_or = np.sqrt(1/correct1 + 1/incorrect1 + 1/correct2 + 1/incorrect2) if all([correct1, incorrect1, correct2, incorrect2]) else np.nan
    
    ci_lower = np.exp(log_or - 1.96 * se_log_or) if not np.isnan(se_log_or) else np.nan
    ci_upper = np.exp(log_or + 1.96 * se_log_or) if not np.isnan(se_log_or) else np.nan
    
    # Calculate effect size (difference in proportions)
    prop1 = correct1 / total1 if total1 > 0 else 0
    prop2 = correct2 / total2 if total2 > 0 else 0
    effect_size = prop1 - prop2
    
    return {
        "method1": method1_name,
        "method2": method2_name,
        "method1_correct": correct1,
        "method1_total": total1,
        "method1_accuracy": prop1,
        "method2_correct": correct2,
        "method2_total": total2,
        "method2_accuracy": prop2,
        "contingency_table": contingency_table,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "effect_size": effect_size,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < 0.05
    }


def format_fisher_results(result: Dict) -> str:
    """Format Fisher's test results for display."""
    output = []
    output.append("=" * 70)
    output.append("FISHER'S EXACT TEST RESULTS")
    output.append("=" * 70)
    output.append(f"Comparison: {result['method1'].upper()} vs {result['method2'].upper()}")
    output.append("-" * 70)
    
    # Method statistics
    output.append(f"{result['method1'].upper():<12} {result['method1_correct']}/{result['method1_total']:<10} {result['method1_accuracy']:.3f} ({result['method1_accuracy']*100:.1f}%)")
    output.append(f"{result['method2'].upper():<12} {result['method2_correct']}/{result['method2_total']:<10} {result['method2_accuracy']:.3f} ({result['method2_accuracy']*100:.1f}%)")
    output.append("-" * 70)
    
    # Test results
    output.append(f"P-value:        {result['p_value']:.6f}")
    output.append(f"Odds Ratio:     {result['odds_ratio']:.4f}")
    output.append(f"95% CI:         [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    output.append(f"Effect Size:    {result['effect_size']:.4f} ({result['effect_size']*100:+.1f} percentage points)")
    output.append(f"Significant:    {'YES' if result['significant'] else 'NO'} (α = 0.05)")
    
    # Interpretation
    output.append("-" * 70)
    if result['significant']:
        direction = "better" if result['effect_size'] > 0 else "worse"
        output.append(f"CONCLUSION: {result['method1'].upper()} performs significantly {direction} than {result['method2'].upper()}")
    else:
        output.append(f"CONCLUSION: No statistically significant difference between methods")
        
    output.append("=" * 70)
    
    return "\n".join(output)


def save_fisher_results(results: List[Dict], output_file: str = "results/fisher_exact_tests.txt"):
    """Save Fisher's test results to file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("STATISTICAL SIGNIFICANCE ANALYSIS\n")
        f.write("Generated from latest evaluation results\n\n")
        
        for result in results:
            f.write(format_fisher_results(result))
            f.write("\n\n")
    
    print(f"Fisher's test results saved to: {output_file}")


def main():
    """Main function to run Fisher's exact tests."""
    # Load results
    results = load_latest_results()
    method_stats = extract_method_statistics(results)
    
    print(f"Found methods: {list(method_stats.keys())}")
    
    # Perform pairwise comparisons
    fisher_results = []
    methods = list(method_stats.keys())
    
    # Key comparisons
    important_comparisons = [
        ("dcbs", "greedy"),
        ("dcbs", "top_p"),
        ("greedy", "top_p"),
        ("dcbs", "random"),
        ("greedy", "random"),
        ("top_p", "random")
    ]
    
    for method1, method2 in important_comparisons:
        if method1 in method_stats and method2 in method_stats:
            result = perform_fisher_test(
                method_stats[method1], 
                method_stats[method2],
                method1, 
                method2
            )
            fisher_results.append(result)
            print(format_fisher_results(result))
            print()
    
    # Save results
    save_fisher_results(fisher_results)
    
    # Summary
    print("SUMMARY:")
    significant_differences = [r for r in fisher_results if r['significant']]
    if significant_differences:
        print(f"Found {len(significant_differences)} statistically significant differences:")
        for result in significant_differences:
            direction = ">" if result['effect_size'] > 0 else "<"
            print(f"  {result['method1'].upper()} {direction} {result['method2'].upper()} (p = {result['p_value']:.6f})")
    else:
        print("No statistically significant differences found between methods.")


if __name__ == "__main__":
    main() 