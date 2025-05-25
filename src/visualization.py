"""
Visualization module for evaluation results with proper statistical analysis.

This module provides functions for creating publication-quality charts with
confidence intervals and statistical significance testing.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


@dataclass
class PlotConfig:
    """Configuration for plot appearance."""

    figsize: Tuple[int, int] = (12, 8)
    title_fontsize: int = 16
    label_fontsize: int = 14
    tick_fontsize: int = 12
    annotation_fontsize: int = 11
    baseline_color: str = "#FF6B6B"
    baseline_style: str = "--"
    baseline_alpha: float = 0.7
    bar_alpha: float = 0.8
    error_bar_color: str = "black"
    error_bar_alpha: float = 0.6
    error_bar_capsize: int = 5


class AccuracyVisualizer:
    """Creates accuracy comparison visualizations with statistical analysis."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()

    def create_accuracy_chart(
        self,
        results: Dict,
        output_path: str,
        show_confidence: bool = True,
        show_significance: bool = True,
    ) -> None:
        """Create accuracy comparison bar chart with statistical analysis.

        Args:
            results: Results dictionary with statistics for each method
            output_path: Path to save the chart
            show_confidence: Whether to show confidence intervals
            show_significance: Whether to show statistical significance
        """
        # Extract data from results
        methods, accuracies, intervals, sample_sizes = self._extract_data(results)

        # Create the plot
        fig, ax = plt.subplots(figsize=self.config.figsize)

        # Create colors for each method
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

        # Create bars
        bars = ax.bar(
            range(len(methods)),
            accuracies,
            color=colors,
            alpha=self.config.bar_alpha,
            edgecolor="black",
            linewidth=1,
        )

        # Add confidence intervals if requested
        if show_confidence and intervals:
            error_bars = [
                [acc - ci[0] for acc, ci in zip(accuracies, intervals)],
                [ci[1] - acc for acc, ci in zip(accuracies, intervals)],
            ]
            ax.errorbar(
                range(len(methods)),
                accuracies,
                yerr=error_bars,
                fmt="none",
                color=self.config.error_bar_color,
                alpha=self.config.error_bar_alpha,
                capsize=self.config.error_bar_capsize,
                capthick=2,
            )

        # Add baseline at 50% (random guess)
        baseline = ax.axhline(
            y=50,
            color=self.config.baseline_color,
            linestyle=self.config.baseline_style,
            alpha=self.config.baseline_alpha,
            linewidth=2,
            label="Random baseline (50%)",
        )

        # Customize axes
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(
            [m.title() for m in methods], fontsize=self.config.tick_fontsize
        )
        ax.set_ylabel("Accuracy (%)", fontsize=self.config.label_fontsize)
        ax.set_title(
            "Accuracy by Sampling Method", fontsize=self.config.title_fontsize, pad=20
        )

        # Set y-axis limits with some padding
        y_min = min(45, min(accuracies) - 2)
        y_max = max(accuracies) + 5
        ax.set_ylim(y_min, y_max)

        # Add value annotations on bars
        for i, (bar, acc, n) in enumerate(zip(bars, accuracies, sample_sizes)):
            height = bar.get_height()

            # Main accuracy annotation
            ax.annotate(
                f"{acc:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),  # 5 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=self.config.annotation_fontsize,
                fontweight="bold",
            )

            # Sample size annotation
            ax.annotate(
                f"(n={n})",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, -15),  # 15 points below bar
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=self.config.annotation_fontsize - 1,
                style="italic",
                alpha=0.8,
            )

        # Add significance testing if requested
        if show_significance and len(methods) > 1:
            self._add_significance_annotations(ax, results, methods, accuracies)

        # Add legend
        legend_elements = [baseline]
        if show_confidence:
            legend_elements.append(
                mpatches.Patch(color="none", label="Error bars: 95% CI")
            )

        ax.legend(
            handles=legend_elements,
            loc="upper right",
            fontsize=self.config.annotation_fontsize,
        )

        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

        # Tight layout and save
        plt.tight_layout()

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save with high DPI for publication quality
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"Chart saved to: {output_path}")

    def create_detailed_comparison(self, results: Dict, output_path: str) -> None:
        """Create a detailed comparison chart with multiple metrics."""
        methods, accuracies, intervals, sample_sizes = self._extract_data(results)

        # Extract timing data
        avg_times = [
            results["statistics"][method].get("avg_time_ms", 0) for method in methods
        ]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Accuracy subplot
        colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
        bars1 = ax1.bar(methods, accuracies, color=colors, alpha=0.8)

        # Add confidence intervals
        if intervals:
            error_bars = [
                [acc - ci[0] for acc, ci in zip(accuracies, intervals)],
                [ci[1] - acc for acc, ci in zip(accuracies, intervals)],
            ]
            ax1.errorbar(
                methods,
                accuracies,
                yerr=error_bars,
                fmt="none",
                color="black",
                alpha=0.6,
                capsize=5,
            )

        ax1.axhline(
            y=50, color="red", linestyle="--", alpha=0.7, label="Random baseline"
        )
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Accuracy Comparison")
        ax1.legend()

        # Timing subplot
        bars2 = ax2.bar(methods, avg_times, color=colors, alpha=0.8)
        ax2.set_ylabel("Average Time (ms)")
        ax2.set_title("Average Inference Time")

        # Add value annotations
        for bars, values in [(bars1, accuracies), (bars2, avg_times)]:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                bar.axes.annotate(
                    f"{val:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()

        print(f"Detailed comparison saved to: {output_path}")

    def _extract_data(
        self, results: Dict
    ) -> Tuple[List[str], List[float], List[Tuple[float, float]], List[int]]:
        """Extract plotting data from results dictionary."""
        statistics = results.get("statistics", {})

        methods = []
        accuracies = []
        intervals = []
        sample_sizes = []

        # Order methods for consistent display
        method_order = ["greedy", "top-p", "dcbs", "random"]

        for method in method_order:
            if method in statistics:
                stats = statistics[method]
                methods.append(method)
                accuracies.append(stats["accuracy"])
                intervals.append(stats.get("confidence_interval", (0, 0)))
                sample_sizes.append(stats["total"])

        return methods, accuracies, intervals, sample_sizes

    def _add_significance_annotations(
        self, ax, results: Dict, methods: List[str], accuracies: List[float]
    ) -> None:
        """Add statistical significance annotations between methods."""
        statistics = results["statistics"]

        # Find the best performing method
        best_idx = np.argmax(accuracies)
        best_method = methods[best_idx]

        # Compare each method against the best
        for i, method in enumerate(methods):
            if method == best_method:
                continue

            # Get statistics for both methods
            method_stats = statistics[method]
            best_stats = statistics[best_method]

            # Perform binomial test
            p_value = self._binomial_test(
                method_stats["correct"],
                method_stats["total"],
                best_stats["correct"],
                best_stats["total"],
            )

            # Add significance annotation
            if p_value < 0.05:
                significance = "**" if p_value < 0.01 else "*"
                ax.annotate(
                    significance,
                    xy=(i, accuracies[i]),
                    xytext=(0, 25),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=14,
                    color="red",
                    fontweight="bold",
                )

    @staticmethod
    def _binomial_test(correct1: int, total1: int, correct2: int, total2: int) -> float:
        """Perform Fisher's Exact Test to compare two proportions.
        
        Fisher's Exact Test is more appropriate than z-score tests for comparing
        proportions, especially with smaller sample sizes, as it provides exact
        p-values rather than asymptotic approximations.
        
        Args:
            correct1: Number of correct predictions for method 1
            total1: Total predictions for method 1
            correct2: Number of correct predictions for method 2
            total2: Total predictions for method 2
            
        Returns:
            Two-tailed p-value from Fisher's Exact Test
        """
        # Handle edge cases
        if total1 == 0 or total2 == 0:
            return 1.0
            
        # Calculate incorrect counts
        incorrect1 = total1 - correct1
        incorrect2 = total2 - correct2
        
        # Create contingency table for Fisher's Exact Test
        # Format: [[correct1, incorrect1], [correct2, incorrect2]]
        contingency_table = [[correct1, incorrect1], [correct2, incorrect2]]
        
        try:
            # Use scipy's Fisher's Exact Test implementation
            # Returns odds ratio and two-tailed p-value
            _, p_value = stats.fisher_exact(contingency_table, alternative='two-sided')
            return p_value
        except (ValueError, ZeroDivisionError):
            # Fallback for edge cases where Fisher's test fails
            return 1.0


def create_summary_table(results: Dict, output_path: str) -> None:
    """Create a summary table of results in markdown format."""
    statistics = results["statistics"]

    # Create markdown table
    table_lines = [
        "# Evaluation Results Summary",
        "",
        "| Method | Accuracy (%) | 95% CI | Correct/Total | Avg Time (ms) |",
        "|--------|--------------|--------|---------------|---------------|",
    ]

    for method in ["greedy", "top-p", "dcbs", "random"]:
        if method in statistics:
            stats = statistics[method]
            ci = stats.get("confidence_interval", (0, 0))
            table_lines.append(
                f"| {method.title()} | {stats['accuracy']:.2f} | "
                f"({ci[0]:.1f}, {ci[1]:.1f}) | "
                f"{stats['correct']}/{stats['total']} | "
                f"{stats.get('avg_time_ms', 0):.2f} |"
            )

    table_lines.extend(
        [
            "",
            f"**Total Examples:** {results['config']['total_examples']}",
            f"**Model:** {results['config']['model']}",
            "",
            "**Notes:**",
            "- CI = Confidence Interval (binomial proportion)",
            "- Baseline random performance is 50%",
            "- * indicates p < 0.05, ** indicates p < 0.01",
        ]
    )

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(table_lines))

    print(f"Summary table saved to: {output_path}")


def generate_all_visualizations(results: Dict, output_dir: str) -> None:
    """Generate all visualizations and save to output directory."""
    visualizer = AccuracyVisualizer()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate main accuracy chart
    main_chart_path = os.path.join(output_dir, "accuracy_by_method.png")
    visualizer.create_accuracy_chart(results, main_chart_path)

    # Generate detailed comparison
    detailed_chart_path = os.path.join(output_dir, "detailed_comparison.png")
    visualizer.create_detailed_comparison(results, detailed_chart_path)

    # Generate summary table
    summary_path = os.path.join(output_dir, "results_summary.md")
    create_summary_table(results, summary_path)

    print(f"All visualizations generated in: {output_dir}")
