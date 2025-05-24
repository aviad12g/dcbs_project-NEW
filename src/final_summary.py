#!/usr/bin/env python3
"""
Generate a final summary of DCBS evaluation results in Markdown format.
"""

import argparse
import datetime
import json
import os
from typing import Any, Dict, Optional

import pandas as pd


def generate_summary(input_csv: str, out_file: str) -> None:
    """Generate a summary of evaluation results in Markdown format.

    Args:
        input_csv: Path to CSV file with results
        out_file: Path to output markdown file
    """
    # Load the data
    df = pd.read_csv(input_csv)

    # Extract parameters used in evaluation
    params = {
        "top_n": df["top_n"].unique(),
        "k": df["k"].unique(),
        "p": df["p"].unique(),
    }

    # Calculate summary statistics by method
    summary = df.groupby("method").agg(
        {
            "correct": ["mean", "count", "std"],
            "elapsed_ms": ["mean", "min", "max", "std"],
        }
    )

    # Flatten the hierarchical column names
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

    # Calculate standard error
    summary["correct_se"] = summary["correct_std"] / (summary["correct_count"] ** 0.5)

    # Convert accuracy to percentage
    summary["accuracy"] = summary["correct_mean"] * 100

    # Sort by accuracy (descending)
    summary_sorted = summary.sort_values("accuracy", ascending=False)

    # Extract answer probabilities from the first few examples for demonstration
    answer_probs: Dict[str, Dict[str, float]] = {}
    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        if "answer_probs" in method_df.columns:
            # Take the first example that has answer probabilities
            for _, row in method_df.iterrows():
                if isinstance(row["answer_probs"], str) and row["answer_probs"]:
                    try:
                        answer_probs[method] = json.loads(row["answer_probs"])
                        break
                    except:
                        pass

    # Generate markdown content
    markdown = []

    # Title
    markdown.append("# DCBS Evaluation Results Summary")
    markdown.append("")
    markdown.append(
        f"*Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )
    markdown.append("")

    # Overview section
    markdown.append("## Overview")
    markdown.append("")
    markdown.append(
        "This report summarizes the evaluation of different token sampling methods:"
    )
    markdown.append("")
    markdown.append("- **DCBS**: Disjunctive Category Beam Search")
    markdown.append(
        "- **Greedy**: Standard greedy sampling (highest probability token)"
    )
    markdown.append("- **Top-p**: Nucleus sampling")
    markdown.append("- **Random**: Random sampling from allowed tokens")
    markdown.append("")
    markdown.append(
        "All methods were evaluated with filtering to allowed answer tokens."
    )
    markdown.append("")

    # Dataset section
    markdown.append("## Dataset")
    markdown.append("")
    num_examples = len(df["prompt_id"].unique())
    markdown.append(f"- **Examples evaluated**: {num_examples}")
    if len(params["top_n"]) == 1:
        markdown.append(f"- **top_n**: {params['top_n'][0]}")
    else:
        markdown.append(f"- **top_n values**: {', '.join(map(str, params['top_n']))}")

    if len(params["k"]) == 1:
        markdown.append(f"- **k (clusters)**: {params['k'][0]}")
    else:
        markdown.append(f"- **k values**: {', '.join(map(str, params['k']))}")

    if len(params["p"]) == 1:
        markdown.append(f"- **p (top-p threshold)**: {params['p'][0]}")
    else:
        markdown.append(f"- **p values**: {', '.join(map(str, params['p']))}")
    markdown.append("")

    # Results section
    markdown.append("## Results")
    markdown.append("")
    markdown.append("### Accuracy")
    markdown.append("")
    markdown.append("| Method | Accuracy | # Examples | Std Error |")
    markdown.append("|--------|----------|------------|-----------|")

    for method, row in summary_sorted.iterrows():
        accuracy = f"{row['accuracy']:.2f}%"
        count = int(row["correct_count"])
        std_err = f"±{row['correct_se']*100:.2f}%"
        markdown.append(f"| {method} | {accuracy} | {count} | {std_err} |")

    markdown.append("")

    # Processing time section
    markdown.append("### Processing Time")
    markdown.append("")
    markdown.append("| Method | Avg Time (ms) | Min (ms) | Max (ms) | Std Dev (ms) |")
    markdown.append("|--------|---------------|----------|----------|--------------|")

    for method, row in summary_sorted.iterrows():
        avg_time = f"{row['elapsed_ms_mean']:.2f}"
        min_time = f"{row['elapsed_ms_min']:.2f}"
        max_time = f"{row['elapsed_ms_max']:.2f}"
        std_dev = f"{row['elapsed_ms_std']:.2f}"
        markdown.append(
            f"| {method} | {avg_time} | {min_time} | {max_time} | {std_dev} |"
        )

    markdown.append("")

    # Add token probability analysis if available
    if answer_probs:
        markdown.append("### Token Probability Analysis")
        markdown.append("")
        markdown.append("Sample of answer token probabilities from one example:")
        markdown.append("")

        # Choose a representative method
        for method in ["dcbs", "greedy", "top-p", "random"]:
            if method in answer_probs:
                method_probs = answer_probs[method]

                markdown.append(f"**{method}**:")
                markdown.append("")
                markdown.append("| Token | Probability |")
                markdown.append("|-------|-------------|")

                # Sort tokens by probability (descending)
                sorted_tokens = sorted(
                    method_probs.items(), key=lambda x: x[1], reverse=True
                )

                for token, prob in sorted_tokens:
                    markdown.append(f"| {token} | {prob:.6f} |")

                markdown.append("")
                break

    # Visualizations section
    markdown.append("## Visualizations")
    markdown.append("")
    markdown.append(
        "The following visualizations were generated from the evaluation results:"
    )
    markdown.append("")
    markdown.append(
        "1. **Accuracy by Method**: Bar chart comparing accuracy across sampling methods"
    )
    markdown.append(
        "2. **Processing Time by Method**: Bar chart comparing latency across sampling methods"
    )
    markdown.append(
        "3. **Accuracy vs Latency**: Scatter plot showing the trade-off between accuracy and latency"
    )
    markdown.append(
        "4. **DCBS Latency vs top_n**: Line plot showing how DCBS latency scales with the top_n parameter"
    )
    markdown.append("")
    markdown.append("See the `figures` directory for the full set of visualizations.")
    markdown.append("")

    # Conclusions section
    markdown.append("## Conclusions")
    markdown.append("")

    # Determine best performing method
    best_method = summary_sorted.index[0]
    best_accuracy = summary_sorted["accuracy"].iloc[0]

    # Compare DCBS vs greedy
    dcbs_acc = (
        summary_sorted.loc["dcbs"]["accuracy"] if "dcbs" in summary_sorted.index else 0
    )
    greedy_acc = (
        summary_sorted.loc["greedy"]["accuracy"]
        if "greedy" in summary_sorted.index
        else 0
    )

    if dcbs_acc > greedy_acc:
        diff = dcbs_acc - greedy_acc
        markdown.append(
            f"- **DCBS outperformed greedy sampling** by {diff:.2f} percentage points."
        )
    elif greedy_acc > dcbs_acc:
        diff = greedy_acc - dcbs_acc
        markdown.append(
            f"- **Greedy sampling outperformed DCBS** by {diff:.2f} percentage points."
        )
    else:
        markdown.append(
            "- **DCBS and greedy sampling performed equally** in terms of accuracy."
        )

    # Compare DCBS vs top-p
    topp_acc = (
        summary_sorted.loc["top-p"]["accuracy"]
        if "top-p" in summary_sorted.index
        else 0
    )

    if dcbs_acc > topp_acc:
        diff = dcbs_acc - topp_acc
        markdown.append(
            f"- **DCBS outperformed top-p sampling** by {diff:.2f} percentage points."
        )
    elif topp_acc > dcbs_acc:
        diff = topp_acc - dcbs_acc
        markdown.append(
            f"- **Top-p sampling outperformed DCBS** by {diff:.2f} percentage points."
        )
    else:
        markdown.append(
            "- **DCBS and top-p sampling performed equally** in terms of accuracy."
        )

    # Overall conclusion on which sampling method to use
    markdown.append("")
    markdown.append(
        f"**Overall best method: {best_method}** with {best_accuracy:.2f}% accuracy."
    )

    # Write to output file
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        f.write("\n".join(markdown))

    print(f"Summary report generated at {out_file}")


def main():
    """Parse arguments and generate summary."""
    parser = argparse.ArgumentParser(
        description="Generate summary of DCBS evaluation results"
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--out_file", type=str, required=True, help="Path to output markdown file"
    )

    args = parser.parse_args()
    generate_summary(args.input_csv, args.out_file)


if __name__ == "__main__":
    main()
