#!/usr/bin/env python3
"""
Paired T-Test Analysis for DCBS Evaluation Results

Performs paired t-tests to determine statistical significance 
of performance differences between sampling methods.
"""

import json
import numpy as np
from scipy.stats import ttest_rel
from typing import Dict, List, Tuple

def load_evaluation_results(results_path: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def extract_paired_predictions(results: Dict) -> Dict[str, List[bool]]:
    """Extract paired predictions for each sampling method."""
    detailed_results = results['detailed_results']
    
    # Group by example ID and extract correctness for each sampler
    examples = {}
    for result in detailed_results:
        example_id = result['id']
        sampler = result['sampler']
        correct = result['correct']
        
        if example_id not in examples:
            examples[example_id] = {}
        examples[example_id][sampler] = correct
    
    # Convert to sampler-wise lists
    samplers = {}
    for example_data in examples.values():
        for sampler, correct in example_data.items():
            if sampler not in samplers:
                samplers[sampler] = []
            samplers[sampler].append(correct)
    
    return samplers

def perform_paired_ttest(method1_results: List[bool], method2_results: List[bool], 
                        method1_name: str, method2_name: str) -> Tuple[float, float, str]:
    """Perform paired t-test between two methods."""
    # Convert to numeric (1 for correct, 0 for incorrect)
    x1 = np.array([int(x) for x in method1_results])
    x2 = np.array([int(x) for x in method2_results])
    
    # Perform paired t-test
    statistic, p_value = ttest_rel(x1, x2)
    
    # Determine significance
    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**" 
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = "ns"
    
    return statistic, p_value, significance

def analyze_results(results_path: str):
    """Perform complete paired t-test analysis."""
    print("=" * 70)
    print("PAIRED T-TEST ANALYSIS - DCBS EVALUATION RESULTS")
    print("=" * 70)
    
    # Load results
    results = load_evaluation_results(results_path)
    samplers_data = extract_paired_predictions(results)
    
    print(f"Dataset: {results['config']['total_examples']} examples")
    print(f"Methods: {', '.join(samplers_data.keys())}")
    print()
    
    # Display accuracy summary
    print("ACCURACY SUMMARY:")
    print("-" * 50)
    statistics = results['statistics']
    for method in ['dcbs', 'top_p', 'greedy', 'random']:
        if method in statistics:
            acc = statistics[method]['accuracy']
            correct = statistics[method]['correct']
            total = statistics[method]['total']
            print(f"{method.upper():>8}: {acc:5.1f}% ({correct}/{total})")
    print()
    
    # Perform pairwise comparisons
    print("PAIRED T-TEST RESULTS:")
    print("-" * 70)
    print(f"{'Comparison':<20} {'t-statistic':<12} {'p-value':<12} {'Significance':<12}")
    print("-" * 70)
    
    methods = ['dcbs', 'top_p', 'greedy', 'random']
    available_methods = [m for m in methods if m in samplers_data]
    
    significant_comparisons = []
    
    for i, method1 in enumerate(available_methods):
        for method2 in available_methods[i+1:]:
            statistic, p_value, significance = perform_paired_ttest(
                samplers_data[method1], samplers_data[method2], method1, method2
            )
            
            comparison = f"{method1.upper()} vs {method2.upper()}"
            print(f"{comparison:<20} {statistic:>11.3f} {p_value:>11.4f} {significance:>11}")
            
            if significance != "ns":
                significant_comparisons.append((comparison, p_value, significance))
    
    print("-" * 70)
    print()
    
    # Summary of significant differences
    if significant_comparisons:
        print("SIGNIFICANT DIFFERENCES:")
        print("-" * 30)
        for comparison, p_value, significance in significant_comparisons:
            print(f"{comparison}: p = {p_value:.4f} {significance}")
    else:
        print("No statistically significant differences found at α = 0.05")
    
    print()
    print("Significance levels: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant")
    print("=" * 70)

if __name__ == "__main__":
    # Use the latest evaluation results
    results_path = "results/evaluation_results_20250603_161918.json"
    analyze_results(results_path) 