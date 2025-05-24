#!/usr/bin/env python3
"""
Generate simple but effective visualizations from ARC evaluation results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os

def create_arc_charts():
    # Load the ARC results
    with open('results/arc_full_evaluation.json', 'r') as f:
        data = json.load(f)
    
    stats = data['statistics']
    config = data['config']
    
    # Extract data for plotting
    methods = ['Greedy', 'Top-p', 'DCBS', 'Random']
    method_keys = ['greedy', 'top_p', 'dcbs', 'random']
    
    accuracies = [stats[key]['accuracy'] * 100 for key in method_keys]
    correct_counts = [stats[key]['correct'] for key in method_keys]
    total_counts = [stats[key]['total'] for key in method_keys]
    avg_times = [stats[key]['avg_time_ms'] for key in method_keys]
    
    # Create output directory
    output_dir = "results/arc_charts"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plot style
    plt.style.use('default')
    
    # Create main accuracy chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create colors - highlight DCBS as the winner
    colors = ['#3498db', '#e74c3c', '#f39c12', '#95a5a6']  # Blue, Red, Orange, Gray
    dcbs_idx = 2  # DCBS is at index 2
    colors[dcbs_idx] = '#2ecc71'  # Green for the winner
    
    # Create bars
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value annotations on bars
    for bar, acc, correct, total in zip(bars, accuracies, correct_counts, total_counts):
        height = bar.get_height()
        
        # Main accuracy annotation
        ax.annotate(f'{acc:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 5),
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=14, fontweight='bold')
        
        # Count annotation
        ax.annotate(f'{correct}/{total}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, -20),
                   textcoords="offset points",
                   ha='center', va='top',
                   fontsize=10, style='italic')
    
    # Add 50% baseline for random guessing
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.7, linewidth=2, 
               label='Expected Random (50%)')
    
    # Add 25% baseline for true random on 4-choice
    ax.axhline(y=25, color='gray', linestyle=':', alpha=0.7, linewidth=2, 
               label='True Random (25%)')
    
    # Customize the plot
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'ARC Easy Evaluation Results\n{config["num_questions"]:,} Questions', 
                 fontsize=16, pad=20)
    ax.set_ylim(0, max(accuracies) + 10)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12)
    
    # Tight layout and save
    plt.tight_layout()
    
    # Save main chart
    main_chart_path = os.path.join(output_dir, "arc_accuracy_chart.png")
    plt.savefig(main_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create timing comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Accuracy subplot
    bars1 = ax1.bar(methods, accuracies, color=colors, alpha=0.8)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Comparison')
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
    
    # Timing subplot
    bars2 = ax2.bar(methods, avg_times, color=colors, alpha=0.8)
    ax2.set_ylabel('Average Time (ms)')
    ax2.set_title('Average Inference Time')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, time_ms in zip(bars2, avg_times):
        height = bar.get_height()
        ax2.annotate(f'{time_ms:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10)
    
    plt.tight_layout()
    
    # Save detailed chart
    detailed_chart_path = os.path.join(output_dir, "arc_detailed_comparison.png")
    plt.savefig(detailed_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create summary table (markdown)
    summary_path = os.path.join(output_dir, "arc_results_summary.md")
    with open(summary_path, 'w') as f:
        f.write("# ARC Easy Evaluation Results\n\n")
        f.write(f"**Total Questions:** {config['num_questions']:,}\n")
        f.write(f"**Model:** {config['model']}\n")
        f.write(f"**Chain-of-Thought:** {config['use_cot']}\n\n")
        
        f.write("| Method | Accuracy (%) | Correct/Total | Avg Time (ms) |\n")
        f.write("|--------|--------------|---------------|---------------|\n")
        
        # Sort by accuracy for display
        sorted_data = list(zip(methods, method_keys, accuracies, correct_counts, total_counts, avg_times))
        sorted_data.sort(key=lambda x: x[2], reverse=True)
        
        for method, key, acc, correct, total, time_ms in sorted_data:
            f.write(f"| **{method}** | {acc:.2f}% | {correct}/{total} | {time_ms:.1f} |\n")
        
        f.write(f"\n**Random Baseline:** 25.0% (4-choice multiple choice)\n")
        f.write(f"**DCBS Improvement over Random:** +{stats['dcbs']['accuracy']*100 - stats['random']['accuracy']*100:.1f} percentage points\n")
        f.write(f"**DCBS vs Greedy:** +{stats['dcbs']['accuracy']*100 - stats['greedy']['accuracy']*100:.1f} percentage points\n")
    
    print("✅ Charts generated successfully!")
    print(f"📊 Main chart: {main_chart_path}")
    print(f"📊 Detailed comparison: {detailed_chart_path}")
    print(f"📄 Summary table: {summary_path}")
    
    # Print quick summary
    print(f"\n🎯 ARC Easy Results Summary:")
    for method, key, acc, correct, total, time_ms in sorted_data:
        print(f"{method:<8}: {acc:5.1f}% ({correct:4d}/{total}) avg: {time_ms:4.1f}ms")

if __name__ == "__main__":
    create_arc_charts() 