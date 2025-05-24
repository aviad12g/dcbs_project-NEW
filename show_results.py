#!/usr/bin/env python3
"""
Extract and display results from the full ARC evaluation.
"""

import json

def main():
    # Load results
    with open('results/arc_full_evaluation.json', 'r') as f:
        data = json.load(f)
    
    config = data['config']
    stats = data['statistics']
    
    print("🎯 === FULL ARC EASY EVALUATION RESULTS ===")
    print(f"📊 Total Questions: {config['num_questions']:,}")
    print(f"🤖 Model: {config['model']}")
    print(f"🧠 Chain-of-Thought: {config['use_cot']}")
    print(f"🔧 DCBS Config: k={config['dcbs_k']}, top_n={config['dcbs_top_n']}")
    print()
    
    print("📈 METHOD PERFORMANCE:")
    print("-" * 60)
    print(f"{'Method':<12} {'Accuracy':<12} {'Correct/Total':<15} {'Avg Time (ms)':<15}")
    print("-" * 60)
    
    # Sort by accuracy for better display
    methods = ['greedy', 'top_p', 'dcbs', 'random']
    method_data = [(method, stats[method]) for method in methods if method in stats]
    method_data.sort(key=lambda x: x[1]['accuracy'], reverse=True)
    
    for method, method_stats in method_data:
        accuracy = method_stats['accuracy'] * 100
        correct = method_stats['correct']
        total = method_stats['total']
        avg_time = method_stats['avg_time_ms']
        
        print(f"{method.title():<12} {accuracy:>8.2f}%   {correct:>6}/{total:<6}   {avg_time:>10.1f}")
    
    print("-" * 60)
    print("🎲 Random baseline: 25.0% (4-way multiple choice)")
    print("✅ Results saved to: results/arc_full_evaluation.json")
    
    # Calculate improvements
    if 'dcbs' in stats and 'random' in stats:
        dcbs_acc = stats['dcbs']['accuracy'] * 100
        random_acc = stats['random']['accuracy'] * 100
        improvement = dcbs_acc - random_acc
        print(f"🚀 DCBS improvement over random: +{improvement:.2f} percentage points")
    
    if 'dcbs' in stats and 'greedy' in stats:
        dcbs_acc = stats['dcbs']['accuracy'] * 100
        greedy_acc = stats['greedy']['accuracy'] * 100
        diff = dcbs_acc - greedy_acc
        if diff > 0:
            print(f"🏆 DCBS outperforms greedy by: +{diff:.2f} percentage points")
        else:
            print(f"📊 Greedy outperforms DCBS by: +{abs(diff):.2f} percentage points")

if __name__ == "__main__":
    main() 