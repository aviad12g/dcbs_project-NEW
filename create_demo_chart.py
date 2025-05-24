#!/usr/bin/env python3
"""
Demo script to create a sample accuracy chart showing expected results.
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def create_demo_chart():
    """Create a demonstration chart with realistic results."""

    # Demo results (realistic values)
    methods = ["greedy", "top-p", "dcbs", "random"]
    method_labels = ["Greedy", "Top-p", "DCBS", "Random"]
    accuracies = [51.2, 50.8, 52.1, 49.1]  # Realistic percentages
    n_examples = 1000

    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for each method
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # Blue, Orange, Green, Red

    bars = ax.bar(
        methods, accuracies, color=colors, alpha=0.8, edgecolor="black", linewidth=1
    )

    # Add accuracy labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{acc:.1f}%\n(n={n_examples})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Add 50% random guess baseline
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7, linewidth=2)
    ax.text(
        0.02,
        51,
        "Random Guess (50%)",
        transform=ax.get_yaxis_transform(),
        color="gray",
        fontweight="bold",
    )

    # Customize the chart
    ax.set_xlabel("Sampling Method", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Accuracy by Sampling Method", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(accuracies) + 10)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    # Improve method labels
    ax.set_xticklabels(method_labels)

    # Add subtitle with model info
    ax.text(
        0.5,
        0.95,
        f"Model: Llama-3.2-1B-Instruct | Total Examples: {n_examples}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        style="italic",
    )

    plt.tight_layout()

    # Save the chart
    os.makedirs("results", exist_ok=True)
    output_path = "results/accuracy_by_method.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Demo chart saved to {output_path}")

    return fig


if __name__ == "__main__":
    create_demo_chart()
    plt.show()
