import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_results(csv_path):
    methods = ["greedy", "top-p", "dcbs", "random"]
    results = {method: {"correct": [], "times": []} for method in methods}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"]
            if method in methods:
                results[method]["correct"].append(int(row["correct"]))
                results[method]["times"].append(float(row["elapsed_ms"]))

    return results


def plot_accuracy(results, output_path):
    methods = list(results.keys())
    accuracies = [
        sum(results[m]["correct"]) / len(results[m]["correct"]) * 100 for m in methods
    ]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color=["blue", "green", "red", "purple"])

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
        )

    plt.title("Accuracy by Sampling Method")
    plt.xlabel("Method")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, max(accuracies) * 1.2)
    plt.savefig(output_path)
    plt.close()

    print(f"Accuracy chart saved to {output_path}")


def plot_times(results, output_path):
    methods = list(results.keys())
    avg_times = [np.mean(results[m]["times"]) for m in methods]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, avg_times, color=["blue", "green", "red", "purple"])

    for bar, time in zip(bars, avg_times):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{time:.2f}ms",
            ha="center",
            va="bottom",
        )

    plt.title("Average Execution Time by Sampling Method")
    plt.xlabel("Method")
    plt.ylabel("Time (ms)")
    plt.ylim(0, max(avg_times) * 1.2)
    plt.savefig(output_path)
    plt.close()

    print(f"Time chart saved to {output_path}")


def analyze_and_visualize(csv_path):
    print(f"Analyzing results from {csv_path}...")
    results = load_results(csv_path)

    # Calculate summary statistics
    stats = {}
    for method, data in results.items():
        correct_count = sum(data["correct"])
        total = len(data["correct"])
        avg_time = np.mean(data["times"])

        stats[method] = {
            "accuracy": (correct_count / total) * 100,
            "avg_time": avg_time,
            "count": total,
        }

    # Print summary
    print("\nResults Summary:")
    print("=" * 60)
    print(f"{'Method':<10} {'Accuracy':<15} {'Avg Time (ms)':<15} {'Count':<10}")
    print("-" * 60)
    for method, stat in stats.items():
        print(
            f"{method:<10} {stat['accuracy']:<15.2f} {stat['avg_time']:<15.2f} {stat['count']:<10}"
        )
    print("=" * 60)

    # Generate plots
    accuracy_plot = csv_path.replace(".csv", "_accuracy.png")
    plot_accuracy(results, accuracy_plot)

    time_plot = csv_path.replace(".csv", "_times.png")
    plot_times(results, time_plot)

    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    analyze_and_visualize(csv_path)
