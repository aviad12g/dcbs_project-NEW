"""
Fisher's Exact Test for statistical significance between Greedy and DCBS sampling methods.

This module provides utility functions for performing Fisher's Exact Test
to determine statistical significance between different sampling methods.
"""

import json
import sys
from pathlib import Path
from scipy.stats import fisher_exact


def load_evaluation_results(results_path: str) -> dict:
    """Load evaluation results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in results file: {e}")
        sys.exit(1)


def extract_method_stats(results: dict, method: str) -> tuple:
    """Extract correct/total counts for a specific method."""
    try:
        stats = results['statistics'][method]
        correct = stats['correct']
        total = stats['total']
        accuracy = stats['accuracy']
        return correct, total, accuracy
    except KeyError as e:
        print(f"Error: Missing data for method '{method}': {e}")
        sys.exit(1)


def perform_fisher_exact_test(correct1: int, total1: int, correct2: int, total2: int) -> tuple:
    """
    Perform Fisher's Exact Test on two methods.
    
    Args:
        correct1: Number of correct predictions for method 1
        total1: Total predictions for method 1
        correct2: Number of correct predictions for method 2
        total2: Total predictions for method 2
        
    Returns:
        Tuple of (odds_ratio, p_value)
    """
    # Calculate incorrect counts
    incorrect1 = total1 - correct1
    incorrect2 = total2 - correct2
    
    # Create contingency table: [[correct1, incorrect1], [correct2, incorrect2]]
    contingency_table = [[correct1, incorrect1], [correct2, incorrect2]]
    
    # Perform Fisher's Exact Test (two-sided)
    odds_ratio, p_value = fisher_exact(contingency_table, alternative='two-sided')
    
    return odds_ratio, p_value


def interpret_p_value(p_value: float) -> str:
    """Interpret the p-value for statistical significance."""
    if p_value < 0.001:
        return "highly significant (p < 0.001)"
    elif p_value < 0.01:
        return "very significant (p < 0.01)"
    elif p_value < 0.05:
        return "significant (p < 0.05)"
    elif p_value < 0.1:
        return "marginally significant (p < 0.1)"
    else:
        return "not significant (p >= 0.1)" 