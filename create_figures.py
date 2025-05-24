import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from collections import defaultdict

def load_results(csv_path):
    """Load evaluation results from CSV file."""
    methods = ['greedy', 'top-p', 'dcbs', 'random']
    results = defaultdict(lambda: defaultdict(list))
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            if method in methods:
                # Extract data for each row
                top_n = int(row['top_n'])
                k = int(row['k'])
                p = float(row['p'])
                correct = int(row['correct'])
                elapsed_ms = float(row['elapsed_ms'])
                
                # Store by parameter combination
                param_key = f"{top_n}_{k}_{p}"
                results[param_key][method].append({
                    'correct': correct,
                    'elapsed_ms': elapsed_ms,
                    'prompt_id': row['prompt_id']
                })
                
                # Store raw data for each method
                results['all'][method].append({
                    'correct': correct,
                    'elapsed_ms': elapsed_ms,
                    'top_n': top_n,
                    'k': k,
                    'p': p,
                    'prompt_id': row['prompt_id']
                })
    
    # Also store metadata about parameter combinations
    param_combos = [k for k in results.keys() if k != 'all']
    param_details = []
    for combo in param_combos:
        top_n, k, p = combo.split('_')
        param_details.append({
            'top_n': int(top_n),
            'k': int(k), 
            'p': float(p),
            'key': combo
        })
    
    results['metadata'] = {
        'param_combinations': param_details
    }
    
    return results

def plot_accuracy_comparison(results, output_dir):
    """Create bar chart comparing accuracy across methods."""
    methods = ['greedy', 'top-p', 'dcbs', 'random']
    all_data = results['all']
    
    # Calculate overall accuracy
    accuracies = []
    counts = []
    for method in methods:
        correct = sum(item['correct'] for item in all_data[method])
        total = len(all_data[method])
        accuracy = (correct / total) * 100 if total > 0 else 0
        accuracies.append(accuracy)
        counts.append(total)
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = plt.bar(methods, accuracies, color=colors)
    
    # Add accuracy values above bars
    for bar, accuracy, count in zip(bars, accuracies, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{accuracy:.2f}%\n(n={count})', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Accuracy by Sampling Method', fontsize=14)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, max(accuracies) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal line at 50% (random guessing in binary tasks)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
    plt.text(3.5, 50.5, 'Random guess (50%)', ha='right', va='bottom', color='r', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'method_accuracy_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved accuracy comparison to {output_path}")
    
    return accuracies, counts

def plot_time_comparison(results, output_dir):
    """Create bar chart comparing execution time across methods."""
    methods = ['greedy', 'top-p', 'dcbs', 'random']
    all_data = results['all']
    
    # Calculate average time
    avg_times = []
    for method in methods:
        times = [item['elapsed_ms'] for item in all_data[method]]
        avg_time = np.mean(times)
        avg_times.append(avg_time)
    
    # Create the bar chart
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    bars = plt.bar(methods, avg_times, color=colors)
    
    # Add time values above bars
    for bar, time in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{time:.2f} ms', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Average Execution Time by Method', fontsize=14)
    plt.xlabel('Method', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.ylim(0, max(avg_times) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'method_time_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved time comparison to {output_path}")
    
    return avg_times

def plot_parameter_impact(results, output_dir):
    """Plot how different parameters affect accuracy and time."""
    param_combinations = results['metadata']['param_combinations']
    methods = ['greedy', 'top-p', 'dcbs', 'random']
    
    # Sort parameter combinations by top_n
    param_combinations.sort(key=lambda x: (x['top_n'], x['k'], x['p']))
    
    # Prepare data
    param_labels = []
    accuracy_by_method = {method: [] for method in methods}
    time_by_method = {method: [] for method in methods}
    
    for param in param_combinations:
        key = param['key']
        param_labels.append(f"top_n={param['top_n']}, k={param['k']}, p={param['p']:.1f}")
        
        for method in methods:
            data = results[key][method]
            if data:
                accuracy = sum(item['correct'] for item in data) / len(data) * 100
                avg_time = np.mean([item['elapsed_ms'] for item in data])
                accuracy_by_method[method].append(accuracy)
                time_by_method[method].append(avg_time)
            else:
                accuracy_by_method[method].append(0)
                time_by_method[method].append(0)
    
    # Plot accuracy by parameter combination
    if len(param_labels) > 1:  # Only if we have multiple parameter combinations
        plt.figure(figsize=(12, 7))
        markers = ['o', 's', '^', 'D']
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        for i, method in enumerate(methods):
            plt.plot(range(len(param_labels)), accuracy_by_method[method], 
                    marker=markers[i], label=method, color=colors[i], linewidth=2)
        
        plt.xticks(range(len(param_labels)), param_labels, rotation=45, ha='right')
        plt.title('Accuracy by Parameter Combination', fontsize=14)
        plt.xlabel('Parameter Combination', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend(title='Method')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'parameter_accuracy_impact.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved parameter impact on accuracy to {output_path}")
        
        # Plot time by parameter combination
        plt.figure(figsize=(12, 7))
        
        for i, method in enumerate(methods):
            plt.plot(range(len(param_labels)), time_by_method[method], 
                    marker=markers[i], label=method, color=colors[i], linewidth=2)
        
        plt.xticks(range(len(param_labels)), param_labels, rotation=45, ha='right')
        plt.title('Execution Time by Parameter Combination', fontsize=14)
        plt.xlabel('Parameter Combination', fontsize=12)
        plt.ylabel('Time (ms)', fontsize=12)
        plt.grid(linestyle='--', alpha=0.7)
        plt.legend(title='Method')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'parameter_time_impact.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Saved parameter impact on time to {output_path}")

def plot_accuracy_vs_time(results, output_dir):
    """Create scatter plot of accuracy vs time for all methods."""
    methods = ['greedy', 'top-p', 'dcbs', 'random']
    all_data = results['all']
    
    # Calculate accuracy and average time for each method
    accuracies = []
    avg_times = []
    
    for method in methods:
        data = all_data[method]
        accuracy = sum(item['correct'] for item in data) / len(data) * 100 if data else 0
        avg_time = np.mean([item['elapsed_ms'] for item in data]) if data else 0
        
        accuracies.append(accuracy)
        avg_times.append(avg_time)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    for i, method in enumerate(methods):
        plt.scatter(avg_times[i], accuracies[i], s=200, color=colors[i], label=method, 
                   alpha=0.7, edgecolors='black', linewidth=1)
        plt.annotate(f"{method}\n{accuracies[i]:.2f}%, {avg_times[i]:.2f}ms", 
                    (avg_times[i], accuracies[i]), 
                    xytext=(10, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    plt.title('Accuracy vs Execution Time by Method', fontsize=14)
    plt.xlabel('Average Execution Time (ms)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(linestyle='--', alpha=0.7)
    
    # Add horizontal line at 50% (random guessing in binary tasks)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)
    plt.text(max(avg_times), 50.5, 'Random guess (50%)', ha='right', va='bottom', color='r', alpha=0.7)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'accuracy_vs_time.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved accuracy vs time scatter plot to {output_path}")

def generate_report(results, accuracies, times, output_dir):
    """Generate summary report in JSON and Markdown formats."""
    methods = ['greedy', 'top-p', 'dcbs', 'random']
    all_data = results['all']
    
    # Prepare summary data
    summary = {
        'total_examples': len(all_data[methods[0]]),
        'methods': {}
    }
    
    for i, method in enumerate(methods):
        data = all_data[method]
        summary['methods'][method] = {
            'accuracy': accuracies[i],
            'avg_time_ms': times[i],
            'count': len(data)
        }
    
    # Save JSON summary
    json_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON to {json_path}")
    
    # Generate Markdown report
    md_content = [
        "# DCBS Evaluation Results",
        "",
        f"Total examples evaluated: {summary['total_examples']}",
        "",
        "## Accuracy Comparison",
        "",
        "| Method | Accuracy | Avg Time (ms) |",
        "|--------|----------|---------------|"
    ]
    
    for method in methods:
        stats = summary['methods'][method]
        md_content.append(f"| {method} | {stats['accuracy']:.2f}% | {stats['avg_time_ms']:.2f} |")
    
    md_content.extend([
        "",
        "## Key Observations",
        "",
        "1. **DCBS vs Other Methods**: " + 
        ("DCBS outperforms other methods in accuracy." 
         if summary['methods']['dcbs']['accuracy'] > max(summary['methods']['greedy']['accuracy'], 
                                                      summary['methods']['top-p']['accuracy'],
                                                      summary['methods']['random']['accuracy'])
         else "DCBS does not outperform other methods in this evaluation."),
        "",
        f"2. **Performance Trade-off**: DCBS takes approximately " +
        f"{summary['methods']['dcbs']['avg_time_ms'] / summary['methods']['greedy']['avg_time_ms']:.1f}x " +
        "longer than greedy sampling.",
        "",
        "3. **Overall Results**: " + 
        ("All methods perform similarly in terms of accuracy, suggesting the model's knowledge " +
         "is the limiting factor rather than the sampling method." 
         if max(accuracies) - min(accuracies) < 3 else
         "There are significant differences in accuracy between methods, " +
         "indicating the sampling strategy has a meaningful impact."),
        "",
        "## Generated Visualizations",
        "",
        "1. `method_accuracy_comparison.png`: Bar chart comparing accuracy across methods",
        "2. `method_time_comparison.png`: Bar chart comparing execution time across methods",
        "3. `accuracy_vs_time.png`: Scatter plot showing accuracy vs execution time trade-off",
    ])
    
    # Add parameter impact plots if available
    if len(results['metadata']['param_combinations']) > 1:
        md_content.extend([
            "4. `parameter_accuracy_impact.png`: Line chart showing how parameters affect accuracy",
            "5. `parameter_time_impact.png`: Line chart showing how parameters affect execution time"
        ])
    
    md_path = os.path.join(output_dir, 'evaluation_report.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_content))
    print(f"Saved markdown report to {md_path}")

def main(csv_path):
    """Process results and generate all visualizations."""
    print(f"Analyzing results from {csv_path}...")
    
    # Create output directory
    output_dir = os.path.join('results', 'figures')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and process data
    results = load_results(csv_path)
    
    # Generate visualizations
    accuracies, counts = plot_accuracy_comparison(results, output_dir)
    times = plot_time_comparison(results, output_dir)
    plot_parameter_impact(results, output_dir)
    plot_accuracy_vs_time(results, output_dir)
    
    # Generate summary report
    generate_report(results, accuracies, times, output_dir)
    
    print("\nAll visualizations and reports have been generated!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_figures.py <path_to_csv>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    main(csv_path) 