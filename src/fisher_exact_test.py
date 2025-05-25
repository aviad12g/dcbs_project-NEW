"""
Fisher's Exact Test for statistical significance between Greedy and DCBS sampling methods.

This script loads the latest evaluation results and performs a rigorous statistical
comparison using Fisher's Exact Test to determine if the performance difference
between Greedy and DCBS is statistically significant.
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


def main():
    """Main function to perform Fisher's Exact Test analysis."""
    
    # Load the latest evaluation results
    results_path = "results/arc_full_optimized_no_cache.json"
    print(f"Loading evaluation results from: {results_path}")
    
    results = load_evaluation_results(results_path)
    
    # Extract statistics for Greedy and DCBS
    greedy_correct, greedy_total, greedy_accuracy = extract_method_stats(results, 'greedy')
    dcbs_correct, dcbs_total, dcbs_accuracy = extract_method_stats(results, 'dcbs')
    
    # Calculate actual percentages
    greedy_pct = (greedy_correct / greedy_total) * 100
    dcbs_pct = (dcbs_correct / dcbs_total) * 100
    
    print("\nEvaluation Statistics:")
    print("=" * 50)
    print(f"Greedy:  {greedy_correct:,}/{greedy_total:,} correct ({greedy_pct:.1f}%)")
    print(f"DCBS:    {dcbs_correct:,}/{dcbs_total:,} correct ({dcbs_pct:.1f}%)")
    print(f"Dataset: {greedy_total:,} questions (ARC Easy)")
    
    # Perform Fisher's Exact Test
    print("\nFisher's Exact Test Analysis:")
    print("=" * 50)
    
    odds_ratio, p_value = perform_fisher_exact_test(
        greedy_correct, greedy_total, dcbs_correct, dcbs_total
    )
    
    # Calculate effect size (difference in proportions)
    greedy_prop = greedy_correct / greedy_total
    dcbs_prop = dcbs_correct / dcbs_total
    prop_diff = greedy_prop - dcbs_prop
    
    print(f"Contingency Table:")
    print(f"                Correct    Incorrect    Total")
    print(f"Greedy:         {greedy_correct:,}        {greedy_total - greedy_correct:,}       {greedy_total:,}")
    print(f"DCBS:           {dcbs_correct:,}        {dcbs_total - dcbs_correct:,}       {dcbs_total:,}")
    print()
    print(f"Odds Ratio:     {odds_ratio:.4f}")
    print(f"P-value:        {p_value:.6f}")
    print(f"Significance:   {interpret_p_value(p_value)}")
    print(f"Effect Size:    {prop_diff:.4f} ({prop_diff*100:+.2f} percentage points)")
    
    # Statistical interpretation
    print("\nStatistical Interpretation:")
    print("=" * 50)
    
    if p_value < 0.05:
        if greedy_accuracy > dcbs_accuracy:
            conclusion = "Greedy sampling performs significantly better than DCBS"
        else:
            conclusion = "DCBS performs significantly better than Greedy sampling"
    else:
        conclusion = "No statistically significant difference between Greedy and DCBS"
    
    print(f"Conclusion: {conclusion}")
    print(f"Confidence: Based on {greedy_total:,} samples with α = 0.05")
    
    # Save results to file
    output_path = "results/fisher_exact_greedy_vs_dcbs.txt"
    with open(output_path, 'w') as f:
        f.write("Fisher's Exact Test: Greedy vs DCBS Sampling\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: ARC Easy ({greedy_total:,} questions)\n")
        f.write(f"Greedy:  {greedy_correct:,}/{greedy_total:,} correct ({greedy_pct:.1f}%)\n")
        f.write(f"DCBS:    {dcbs_correct:,}/{dcbs_total:,} correct ({dcbs_pct:.1f}%)\n\n")
        f.write("Statistical Test Results:\n")
        f.write(f"Odds Ratio: {odds_ratio:.4f}\n")
        f.write(f"P-value: {p_value:.6f}\n")
        f.write(f"Significance: {interpret_p_value(p_value)}\n")
        f.write(f"Effect Size: {prop_diff*100:+.2f} percentage points\n\n")
        f.write(f"Conclusion: {conclusion}\n")
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main() 