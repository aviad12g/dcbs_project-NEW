import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Load results
with open('results/arc_llama_full_evaluation.json', 'r') as f:
    data = json.load(f)

statistics = data['statistics']
output_dir = Path('results/analysis')
output_dir.mkdir(exist_ok=True)

# Create DataFrame
methods = list(statistics.keys())
accuracies = [statistics[method]['accuracy'] for method in methods]
avg_times = [statistics[method]['avg_time_ms'] for method in methods]
correct_counts = [statistics[method]['correct'] for method in methods]
total_counts = [statistics[method]['total'] for method in methods]

results_df = pd.DataFrame({
    'Method': methods,
    'Accuracy': accuracies,
    'Avg_Time_ms': avg_times,
    'Correct': correct_counts,
    'Total': total_counts
})

colors = ['#2E8B57', '#FF6B35', '#4ECDC4', '#95A5A6']

print("Creating charts...")

# Chart 2: Timing Comparison
plt.figure(figsize=(12, 8))
bars = plt.bar(results_df['Method'], results_df['Avg_Time_ms'], color=colors, alpha=0.8)

for bar, time_ms in zip(bars, results_df['Avg_Time_ms']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{time_ms:.1f}ms',
            ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.title('ARC Easy Evaluation - Processing Time\nAverage Time per Question by Method', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Sampling Method', fontsize=14, fontweight='bold')
plt.ylabel('Average Time (milliseconds)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

chart_path = output_dir / 'timing_comparison.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Timing chart saved: {chart_path}")

# Chart 3: Accuracy vs Speed
plt.figure(figsize=(12, 8))
for i, (method, acc, time_ms) in enumerate(zip(results_df['Method'], 
                                              results_df['Accuracy'], 
                                              results_df['Avg_Time_ms'])):
    plt.scatter(time_ms, acc, c=colors[i], s=200, alpha=0.8, 
               label=method, edgecolors='black', linewidth=2)
    plt.annotate(method, (time_ms, acc), xytext=(5, 5), 
                textcoords='offset points', fontweight='bold', fontsize=12)

plt.title('ARC Easy Evaluation - Accuracy vs Speed Trade-off\nLlama 3.2-1B Performance', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Average Time per Question (milliseconds)', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()

chart_path = output_dir / 'accuracy_vs_speed.png'
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Accuracy vs Speed chart saved: {chart_path}")

print("All charts created successfully! 🎉") 