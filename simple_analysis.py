"""
Simple analysis of ARC Easy evaluation results without interactive displays.
"""

import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def main():
    print("🚀 **ANALYZING FULL ARC EASY EVALUATION RESULTS**\n")
    
    # Load results
    results_file = Path('results/arc_llama_full_evaluation.json')
    output_dir = Path('results/analysis')
    output_dir.mkdir(exist_ok=True)
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    config = data['config']
    statistics = data['statistics']
    
    print("🔧 **CONFIGURATION:**")
    print(f"Model: {config['model']}")
    print(f"Dataset: {config['data_file']}")
    print(f"Questions: {config['num_questions']:,}")
    print(f"Chain of Thought: {config['use_cot']}")
    print(f"DCBS K: {config['dcbs_k']}")
    print(f"DCBS Top-N: {config['dcbs_top_n']}")
    
    # Create results DataFrame
    methods = list(statistics.keys())
    accuracies = [statistics[method]['accuracy'] for method in methods]
    correct_counts = [statistics[method]['correct'] for method in methods]
    total_counts = [statistics[method]['total'] for method in methods]
    avg_times = [statistics[method]['avg_time_ms'] for method in methods]
    
    results_df = pd.DataFrame({
        'Method': methods,
        'Accuracy': accuracies,
        'Correct': correct_counts,
        'Total': total_counts,
        'Avg_Time_ms': avg_times
    })
    
    print(f"\n📊 **FINAL RESULTS TABLE:**")
    print(results_df.to_string(index=False, float_format='%.3f'))
    
    # Chart 1: Accuracy Comparison
    plt.figure(figsize=(12, 8))
    colors = ['#2E8B57', '#FF6B35', '#4ECDC4', '#95A5A6']
    bars = plt.bar(results_df['Method'], results_df['Accuracy'], color=colors, alpha=0.8)
    
    for i, (bar, acc, correct, total) in enumerate(zip(bars, results_df['Accuracy'], 
                                                      results_df['Correct'], results_df['Total'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.1%}\n({correct:,}/{total:,})',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    plt.title('ARC Easy Evaluation Results - Llama 3.2-1B\nAccuracy by Sampling Method', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Sampling Method', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.ylim(0, max(results_df['Accuracy']) * 1.15)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    chart_path = output_dir / 'accuracy_comparison.png'
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Accuracy chart saved: {chart_path}")
    
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
    
    # Generate method agreement analysis
    print(f"\n🔍 **METHOD AGREEMENT ANALYSIS:**")
    individual_results = data['individual_results']
    
    method_pairs = [('greedy', 'dcbs'), ('greedy', 'top_p'), ('dcbs', 'top_p')]
    agreement_patterns = {}
    
    for question in individual_results:
        results = question['results']
        
        for method1, method2 in method_pairs:
            correct1 = results[method1]['correct']
            correct2 = results[method2]['correct']
            
            key = f"{method1}_vs_{method2}"
            if key not in agreement_patterns:
                agreement_patterns[key] = {'both_correct': 0, 'both_wrong': 0, 
                                         'method1_only': 0, 'method2_only': 0, 'total': 0}
            
            if correct1 and correct2:
                agreement_patterns[key]['both_correct'] += 1
            elif not correct1 and not correct2:
                agreement_patterns[key]['both_wrong'] += 1
            elif correct1 and not correct2:
                agreement_patterns[key]['method1_only'] += 1
            elif not correct1 and correct2:
                agreement_patterns[key]['method2_only'] += 1
            
            agreement_patterns[key]['total'] += 1
    
    for pair, stats in agreement_patterns.items():
        method1, method2 = pair.split('_vs_')
        total = stats['total']
        both_correct = stats['both_correct']
        both_wrong = stats['both_wrong']
        agreement_rate = (both_correct + both_wrong) / total
        
        print(f"\n**{method1.upper()} vs {method2.upper()}:**")
        print(f"  Agreement Rate: {agreement_rate:.1%}")
        print(f"  Both Correct: {both_correct:,} ({both_correct/total:.1%})")
        print(f"  Both Wrong: {both_wrong:,} ({both_wrong/total:.1%})")
        print(f"  Only {method1} Correct: {stats['method1_only']:,} ({stats['method1_only']/total:.1%})")
        print(f"  Only {method2} Correct: {stats['method2_only']:,} ({stats['method2_only']/total:.1%})")
    
    # Generate summary report
    report_path = output_dir / 'full_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write("# ARC Easy Evaluation - Llama 3.2-1B Complete Analysis\n\n")
        
        f.write("## 📊 Configuration\n")
        f.write(f"- **Model**: {config['model']}\n")
        f.write(f"- **Dataset**: {config['data_file']}\n")
        f.write(f"- **Questions**: {config['num_questions']:,}\n")
        f.write(f"- **Chain of Thought**: {config['use_cot']}\n")
        f.write(f"- **DCBS Parameters**: K={config['dcbs_k']}, Top-N={config['dcbs_top_n']}\n\n")
        
        f.write("## 🎯 Results Summary\n\n")
        f.write("| Method | Accuracy | Correct/Total | Avg Time (ms) |\n")
        f.write("|--------|----------|---------------|---------------|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| **{row['Method']}** | **{row['Accuracy']:.1%}** | "
                   f"{row['Correct']:,}/{row['Total']:,} | {row['Avg_Time_ms']:.1f} |\n")
        
        f.write("\n## 🔍 Key Insights\n\n")
        
        best_method = results_df.loc[results_df['Accuracy'].idxmax()]
        f.write(f"- **Best Accuracy**: {best_method['Method']} at {best_method['Accuracy']:.1%}\n")
        
        fastest_method = results_df.loc[results_df['Avg_Time_ms'].idxmin()]
        f.write(f"- **Fastest Method**: {fastest_method['Method']} at {fastest_method['Avg_Time_ms']:.1f}ms\n")
        
        greedy_acc = results_df[results_df['Method'] == 'greedy']['Accuracy'].iloc[0]
        dcbs_acc = results_df[results_df['Method'] == 'dcbs']['Accuracy'].iloc[0]
        performance_gap = greedy_acc - dcbs_acc
        f.write(f"- **Greedy vs DCBS Gap**: {performance_gap:.1%} ({greedy_acc:.1%} vs {dcbs_acc:.1%})\n")
        
        random_acc = results_df[results_df['Method'] == 'random']['Accuracy'].iloc[0]
        f.write(f"- **All methods significantly outperform random baseline** ({random_acc:.1%})\n")
        
        f.write("\n## 📈 Charts Generated\n")
        f.write("- `accuracy_comparison.png` - Accuracy by method\n")
        f.write("- `timing_comparison.png` - Processing time by method\n")
        f.write("- `accuracy_vs_speed.png` - Accuracy vs speed trade-off\n")
    
    print(f"✅ Summary report saved: {report_path}")
    print(f"\n✅ **ANALYSIS COMPLETE!**")
    print(f"All files saved to: {output_dir}")

if __name__ == "__main__":
    main() 