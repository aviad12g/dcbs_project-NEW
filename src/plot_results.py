#!/usr/bin/env python3
"""
Visualize results comparing DCBS, Greedy, Top-p, and Random sampling methods.

This module generates visualization plots from evaluation results:
1. Accuracy by sampling method
2. Processing time by method
3. Accuracy vs latency trade-off
4. DCBS latency scaling with top_n parameter
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import yaml
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional

# Set a better visual style
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")
sns.set_context("talk")


class VisualizationError(Exception):
    """Exception raised for errors during visualization."""
    pass


def load_config(config_path: str = "configs/study_config.yaml") -> Dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        VisualizationError: If configuration file cannot be loaded
    """
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Use default colors if config not found
        return {"visualization": {"method_colors": {
            "greedy": "#1f77b4",  # Blue
            "top-p": "#ff7f0e",   # Orange
            "dcbs": "#2ca02c",    # Green
            "random": "#d62728",  # Red
        }}}
    except Exception as e:
        raise VisualizationError(f"Error loading config: {e}")


def get_method_colors(config: Dict) -> Dict[str, str]:
    """Extract method colors from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary mapping method names to color codes
    """
    try:
        return config.get("visualization", {}).get("method_colors", {})
    except (KeyError, AttributeError):
        # Return default colors if not found in config
        return {
            "greedy": "#1f77b4",  # Blue
            "top-p": "#ff7f0e",   # Orange
            "dcbs": "#2ca02c",    # Green
            "random": "#d62728",  # Red
        }


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics from results.

    Args:
        df: DataFrame with evaluation results

    Returns:
        DataFrame with summarized statistics
    """
    # Group by method and parameters
    summary = (
        df.groupby(["method", "top_n", "k", "p"])
        .agg({"correct": ["mean", "count", "std"], "elapsed_ms": ["mean", "std"]})
        .reset_index()
    )

    # Flatten column names
    summary.columns = ["_".join(col).strip("_") for col in summary.columns.values]

    # Rename for clarity
    summary = summary.rename(
        columns={
            "correct_mean": "accuracy",
            "correct_count": "n_samples",
            "elapsed_ms_mean": "avg_time_ms",
        }
    )

    # Calculate standard error
    summary["accuracy_se"] = summary["correct_std"] / (summary["n_samples"] ** 0.5)

    return summary


def plot_accuracy_by_method(
    summary: pd.DataFrame, out_dir: str, method_colors: Dict[str, str]
) -> None:
    """Plot accuracy by method across parameter combinations.

    Args:
        summary: DataFrame with summarized results
        out_dir: Output directory for saving the plot
        method_colors: Dictionary mapping method names to color codes
    """
    # Get unique parameter combinations
    param_combinations = (
        summary.groupby(["top_n", "k", "p"]).size().reset_index()[["top_n", "k", "p"]]
    )
    best_params = param_combinations.iloc[
        0
    ]  # Use first combination for a clean comparison

    # Filter for just the best parameter combination
    best_summary = summary[
        (summary["top_n"] == best_params["top_n"])
        & (summary["k"] == best_params["k"])
        & (summary["p"] == best_params["p"])
    ]

    fig, ax = plt.figure(figsize=(10, 7)), plt.subplot(111)

    # Order methods in a meaningful way
    method_order = ["greedy", "top-p", "dcbs", "random"]
    method_labels = ["Greedy", "Top-p", "DCBS", "Random"]

    # Prepare data
    data = []
    errors = []
    colors = []

    for method in method_order:
        method_data = best_summary[best_summary["method"] == method]
        if not method_data.empty:
            data.append(
                method_data["accuracy"].values[0] * 100
            )  # Convert to percentage
            errors.append(method_data["accuracy_se"].values[0] * 100)
            colors.append(method_colors.get(method, "#333333"))
        else:
            data.append(0)
            errors.append(0)
            colors.append("#cccccc")

    # Create grouped bars
    bars = ax.bar(method_labels, data, yerr=errors, capsize=5, color=colors, width=0.7)

    # Add value labels on top of bars
    for bar, val in zip(bars, data):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 2,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Customize the plot
    ax.set_title(
        f'Accuracy by Sampling Method (top_n={best_params["top_n"]}, k={best_params["k"]}, p={best_params["p"]})',
        fontsize=16,
        pad=20,
    )
    ax.set_ylim(0, max(data) * 1.2)  # Give some headroom for error bars
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_xlabel("Sampling Method", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add horizontal grid lines
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Add a horizontal line at 50% (random guess)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Random Guess (50%)")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_by_method.png"), dpi=300)
    plt.close()


def plot_time_by_method(
    summary: pd.DataFrame, out_dir: str, method_colors: Dict[str, str]
) -> None:
    """Plot average time by method across parameter combinations.

    Args:
        summary: DataFrame with summarized results
        out_dir: Output directory for saving the plot
        method_colors: Dictionary mapping method names to color codes
    """
    # Get unique parameter combinations
    param_combinations = (
        summary.groupby(["top_n", "k", "p"]).size().reset_index()[["top_n", "k", "p"]]
    )
    best_params = param_combinations.iloc[
        0
    ]  # Use first combination for a clean comparison

    # Filter for just the best parameter combination
    best_summary = summary[
        (summary["top_n"] == best_params["top_n"])
        & (summary["k"] == best_params["k"])
        & (summary["p"] == best_params["p"])
    ]

    fig, ax = plt.figure(figsize=(10, 7)), plt.subplot(111)

    # Order methods in a meaningful way
    method_order = ["greedy", "top-p", "dcbs", "random"]
    method_labels = ["Greedy", "Top-p", "DCBS", "Random"]

    # Prepare data
    data = []
    errors = []
    colors = []

    for method in method_order:
        method_data = best_summary[best_summary["method"] == method]
        if not method_data.empty:
            data.append(method_data["avg_time_ms"].values[0])
            errors.append(method_data["elapsed_ms_std"].values[0])
            colors.append(method_colors.get(method, "#333333"))
        else:
            data.append(0)
            errors.append(0)
            colors.append("#cccccc")

    # Create grouped bars
    bars = ax.bar(method_labels, data, yerr=errors, capsize=5, color=colors, width=0.7)

    # Add value labels on top of bars
    for bar, val in zip(bars, data):
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

    # Customize the plot
    ax.set_title(
        f'Processing Time by Sampling Method (top_n={best_params["top_n"]}, k={best_params["k"]}, p={best_params["p"]})',
        fontsize=16,
        pad=20,
    )
    ax.set_ylim(0, max(data) * 1.2)  # Give some headroom for error bars
    ax.set_ylabel("Average Time (ms)", fontsize=14)
    ax.set_xlabel("Sampling Method", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add horizontal grid lines
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Use a logarithmic scale if data varies widely
    if max(data) / (min([x for x in data if x > 0]) or 1) > 100:
        ax.set_yscale("log")
        ax.set_ylabel("Average Time (ms) - Log Scale", fontsize=14)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_by_method.png"), dpi=300)
    plt.close()


def plot_accuracy_vs_latency(
    summary: pd.DataFrame, out_dir: str, method_colors: Dict[str, str]
) -> None:
    """Plot accuracy vs latency scatter plot for different methods.

    Args:
        summary: DataFrame with summarized results
        out_dir: Output directory for saving the plot
        method_colors: Dictionary mapping method names to color codes
    """
    fig, ax = plt.figure(figsize=(10, 8)), plt.subplot(111)

    # Filter for methods of interest
    methods = ["greedy", "top-p", "dcbs", "random"]
    filtered_summary = summary[summary["method"].isin(methods)]

    # Create scatter plot
    for method in methods:
        method_data = filtered_summary[filtered_summary["method"] == method]
        if not method_data.empty:
            ax.scatter(
                method_data["avg_time_ms"],
                method_data["accuracy"] * 100,  # Convert to percentage
                s=100,
                color=method_colors.get(method, "#333333"),
                label=method.capitalize(),
                alpha=0.7,
            )

            # Add parameter annotations for DCBS points
            if method == "dcbs":
                for _, row in method_data.iterrows():
                    ax.annotate(
                        f"top_n={int(row['top_n'])}, k={int(row['k'])}",
                        (row["avg_time_ms"], row["accuracy"] * 100),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                        alpha=0.8,
                    )

    # Customize the plot
    ax.set_title("Accuracy vs Latency by Sampling Method", fontsize=16, pad=20)
    ax.set_xlabel("Average Latency (ms)", fontsize=14)
    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add a horizontal line at 50% (random guess)
    ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Random Guess (50%)")

    # Use logarithmic scale for x-axis if data varies widely
    if (
        filtered_summary["avg_time_ms"].max() / filtered_summary["avg_time_ms"].min()
        > 100
    ):
        ax.set_xscale("log")
        ax.set_xlabel("Average Latency (ms) - Log Scale", fontsize=14)

    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_vs_latency.png"), dpi=300)
    plt.close()


def plot_dcbs_latency_vs_topn(
    summary: pd.DataFrame, out_dir: str, method_colors: Dict[str, str]
) -> None:
    """Plot DCBS latency vs top_n parameter.

    Args:
        summary: DataFrame with summarized results
        out_dir: Output directory for saving the plot
        method_colors: Dictionary mapping method names to color codes
    """
    # Filter for DCBS method only
    dcbs_summary = summary[summary["method"] == "dcbs"]

    # Check if we have enough data points with varying top_n
    if len(dcbs_summary["top_n"].unique()) < 2:
        print("Not enough top_n variations to plot latency vs top_n")
        return

    fig, ax = plt.figure(figsize=(10, 7)), plt.subplot(111)

    # Group by top_n and k
    grouped = dcbs_summary.groupby(["top_n", "k"])

    # Plot lines for each k value
    for (_, k), group in grouped:
        sorted_group = group.sort_values("top_n")
        ax.plot(
            sorted_group["top_n"],
            sorted_group["avg_time_ms"],
            marker="o",
            linewidth=2,
            label=f"k={k}",
            color=method_colors.get("dcbs", "#2ca02c"),  # Use DCBS color
        )

    # Customize the plot
    ax.set_title("DCBS Latency vs top_n Parameter", fontsize=16, pad=20)
    ax.set_xlabel("Top-n Parameter", fontsize=14)
    ax.set_ylabel("Average Latency (ms)", fontsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.7)

    if len(grouped) > 1:  # Only show legend if we have multiple k values
        ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dcbs_latency_vs_topn.png"), dpi=300)
    plt.close()


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description="Generate plots from evaluation results"
    )
    parser.add_argument(
        "--input_csv", type=str, required=True, help="Path to input CSV file"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/figures",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/study_config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.out_dir, exist_ok=True)

        # Load data
        try:
            df = pd.read_csv(args.input_csv)
        except FileNotFoundError:
            raise VisualizationError(f"Input CSV file not found: {args.input_csv}")
        except pd.errors.EmptyDataError:
            raise VisualizationError(f"Input CSV file is empty: {args.input_csv}")
        except pd.errors.ParserError:
            raise VisualizationError(f"Error parsing CSV file: {args.input_csv}")

        # Load configuration and get method colors
        config = load_config(args.config)
        method_colors = get_method_colors(config)

        # Generate summary statistics
        summary = summarize_results(df)

        # Save summary to CSV
        summary_path = os.path.join(os.path.dirname(args.out_dir), "summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Summary statistics saved to {summary_path}")

        # Generate plots
        plot_accuracy_by_method(summary, args.out_dir, method_colors)
        plot_time_by_method(summary, args.out_dir, method_colors)
        plot_accuracy_vs_latency(summary, args.out_dir, method_colors)
        plot_dcbs_latency_vs_topn(summary, args.out_dir, method_colors)

        print(f"Plots generated and saved to {args.out_dir}")

    except VisualizationError as e:
        print(f"Visualization error: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()
