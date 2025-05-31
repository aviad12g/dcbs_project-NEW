#!/usr/bin/env python3
"""
Statistical Analysis using Paired t-test

This script performs paired t-test to determine statistical significance
between different sampling methods in the DCBS evaluation results.
"""

import json
import sys
from pathlib import Path
from scipy.stats import ttest_rel
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


def extract_method_predictions(results: Dict) -> Dict[str, List[bool]]:
    """Extract correct/incorrect predictions for each method."""
    detailed_results = results.get("detailed_results", [])
    method_predictions = {}
    
    # Group by example ID first
    examples = {}
    for result in detailed_results:
        example_id = result.get("id")
        sampler = result.get("sampler")
        correct = result.get("correct", False)
        
        if example_id not in examples:
            examples[example_id] = {}
        examples[example_id][sampler] = correct
    
    # Convert to aligned lists for each method
    for sampler in ["greedy", "top_p", "dcbs", "random"]:
        predictions = []
        for example_id in sorted(examples.keys()):
            if sampler in examples[example_id]:
                predictions.append(examples[example_id][sampler])
        method_predictions[sampler] = predictions
    
    return method_predictions


def perform_paired_ttest(predictions1: List[bool], predictions2: List[bool], 
                        method1_name: str, method2_name: str) -> Dict:
    """
    Perform paired t-test between two methods.
    
    Returns:
        Dictionary with test results including t-statistic and p-value
    """
    # Convert boolean predictions to numeric (1 for correct, 0 for incorrect)
    scores1 = np.array([1 if p else 0 for p in predictions1])
    scores2 = np.array([1 if p else 0 for p in predictions2])
    
    # Perform paired t-test
    t_statistic, p_value = ttest_rel(scores1, scores2)
    
    # Calculate means and difference
    mean1 = np.mean(scores1)
    mean2 = np.mean(scores2)
    mean_diff = mean1 - mean2
    
    # Calculate 95% confidence interval for the difference
    diff = scores1 - scores2
    se_diff = np.std(diff, ddof=1) / np.sqrt(len(diff))
    ci_lower = mean_diff - 1.96 * se_diff
    ci_upper = mean_diff + 1.96 * se_diff
    
    return {
        "method1": method1_name,
        "method2": method2_name,
        "n_examples": len(predictions1),
        "method1_accuracy": mean1,
        "method2_accuracy": mean2,
        "mean_difference": mean_diff,
        "t_statistic": t_statistic,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": p_value < 0.05
    }


def calculate_dcbs_divergence(results: Dict) -> Dict:
    """Calculate percentage of times DCBS selected a different token than greedy."""
    detailed_results = results.get("detailed_results", [])
    
    # Group by example ID
    examples = {}
    for result in detailed_results:
        example_id = result.get("id")
        sampler = result.get("sampler")
        pred_id = result.get("pred_id")
        
        if example_id not in examples:
            examples[example_id] = {}
        examples[example_id][sampler] = pred_id
    
    # Calculate divergence
    total_examples = 0
    diverged_examples = 0
    
    for example_id, predictions in examples.items():
        if "greedy" in predictions and "dcbs" in predictions:
            total_examples += 1
            if predictions["greedy"] != predictions["dcbs"]:
                diverged_examples += 1
    
    divergence_rate = (diverged_examples / total_examples * 100) if total_examples > 0 else 0
    
    return {
        "total_examples": total_examples,
        "diverged_examples": diverged_examples,
        "divergence_rate": divergence_rate
    }


def format_ttest_results(result: Dict) -> str:
    """Format t-test results for display."""
    output = []
    output.append("=" * 70)
    output.append("PAIRED T-TEST RESULTS")
    output.append("=" * 70)
    output.append(f"Comparison: {result['method1'].upper()} vs {result['method2'].upper()}")
    output.append(f"Number of examples: {result['n_examples']}")
    output.append("-" * 70)
    
    # Method statistics
    output.append(f"{result['method1'].upper():<12} Accuracy: {result['method1_accuracy']:.3f} ({result['method1_accuracy']*100:.1f}%)")
    output.append(f"{result['method2'].upper():<12} Accuracy: {result['method2_accuracy']:.3f} ({result['method2_accuracy']*100:.1f}%)")
    output.append("-" * 70)
    
    # Test results
    output.append(f"Mean Difference:  {result['mean_difference']:.4f} ({result['mean_difference']*100:+.2f} percentage points)")
    output.append(f"T-statistic:      {result['t_statistic']:.4f}")
    output.append(f"P-value:          {result['p_value']:.6f}")
    output.append(f"95% CI:           [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
    output.append(f"Significant:      {'YES' if result['significant'] else 'NO'} (α = 0.05)")
    
    # Interpretation
    output.append("-" * 70)
    if result['significant']:
        direction = "better" if result['mean_difference'] > 0 else "worse"
        output.append(f"CONCLUSION: {result['method1'].upper()} performs significantly {direction} than {result['method2'].upper()}")
    else:
        output.append(f"CONCLUSION: No statistically significant difference between methods")
        
    output.append("=" * 70)
    
    return "\n".join(output)


def save_statistical_results(ttest_results: List[Dict], divergence: Dict, 
                           output_file: str = "results/statistical_analysis.txt"):
    """Save statistical analysis results to file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("STATISTICAL SIGNIFICANCE ANALYSIS\n")
        f.write("Generated from latest evaluation results\n\n")
        
        # Write DCBS divergence analysis first
        f.write("=" * 70 + "\n")
        f.write("DCBS DIVERGENCE ANALYSIS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total examples: {divergence['total_examples']}\n")
        f.write(f"Examples where DCBS diverged from Greedy: {divergence['diverged_examples']}\n")
        f.write(f"Divergence rate: {divergence['divergence_rate']:.2f}%\n")
        f.write("=" * 70 + "\n\n")
        
        # Write t-test results
        for result in ttest_results:
            f.write(format_ttest_results(result))
            f.write("\n\n")
    
    print(f"Statistical analysis results saved to: {output_file}")


def main():
    """Main function to run statistical analysis."""
    # Load results
    results = load_latest_results()
    method_predictions = extract_method_predictions(results)
    
    print(f"Found methods: {list(method_predictions.keys())}")
    
    # Perform pairwise comparisons
    ttest_results = []
    important_comparisons = [
        ("dcbs", "greedy"),
        ("dcbs", "top_p"),
        ("greedy", "top_p"),
        ("dcbs", "random"),
        ("greedy", "random"),
        ("top_p", "random")
    ]
    
    for method1, method2 in important_comparisons:
        if method1 in method_predictions and method2 in method_predictions:
            predictions1 = method_predictions[method1]
            predictions2 = method_predictions[method2]
            
            # Ensure we have paired data
            if len(predictions1) == len(predictions2):
                result = perform_paired_ttest(
                    predictions1, predictions2,
                    method1, method2
                )
                ttest_results.append(result)
                print(format_ttest_results(result))
                print()
    
    # Calculate DCBS divergence
    divergence = calculate_dcbs_divergence(results)
    print(f"\nDCBS Divergence Analysis:")
    print(f"DCBS selected a different token than Greedy in {divergence['divergence_rate']:.2f}% of examples")
    
    # Save results
    save_statistical_results(ttest_results, divergence)
    
    # Summary
    print("\nSUMMARY:")
    significant_differences = [r for r in ttest_results if r['significant']]
    if significant_differences:
        print(f"Found {len(significant_differences)} statistically significant differences:")
        for result in significant_differences:
            direction = ">" if result['mean_difference'] > 0 else "<"
            print(f"  {result['method1'].upper()} {direction} {result['method2'].upper()} (p = {result['p_value']:.6f})")
    else:
        print("No statistically significant differences found between methods.")


if __name__ == "__main__":
    main() 