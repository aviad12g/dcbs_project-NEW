#!/usr/bin/env python3
import json

# Load the results  
with open('results/evaluation_results_20250623_214122.json', 'r') as f:
    data = json.load(f)

stats = data['statistics']

sample_count = stats.get('greedy', {}).get('total', 0)
print(f"=== PHI-3.5 MINI INSTRUCT RESULTS ({sample_count} samples) ===")
print(f"Greedy: {stats['greedy']['accuracy']:.1f}% ({stats['greedy']['correct']}/{stats['greedy']['total']})")
print(f"DCBS:   {stats['dcbs']['accuracy']:.1f}% ({stats['dcbs']['correct']}/{stats['dcbs']['total']})")

# Only print if available
if 'top_p' in stats:
    print(f"Top-P:  {stats['top_p']['accuracy']:.1f}% ({stats['top_p']['correct']}/{stats['top_p']['total']})")
if 'random' in stats:
    print(f"Random: {stats['random']['accuracy']:.1f}% ({stats['random']['correct']}/{stats['random']['total']})")

print(f"\nModel: {data['config']['model']}")
print(f"Clustering: {data['config']['clustering_method']}")

print("\n=== CONFIDENCE ANALYSIS ===")
greedy_results = [r for r in data['detailed_results'] if r['sampler'] == 'greedy']
high_conf = sum(1 for r in greedy_results if max(r['answer_probs'].values()) > 0.95)
avg_conf = sum(max(r['answer_probs'].values()) for r in greedy_results) / len(greedy_results)

print(f"Questions with >95% greedy confidence: {high_conf}/{len(greedy_results)}")
print(f"Average greedy confidence: {avg_conf:.3f}")

print("\n=== CLUSTER INFO ===")
dcbs_results = [r for r in data['detailed_results'] if r['sampler'] == 'dcbs']
cluster_null = sum(1 for r in dcbs_results if r['cluster_info'] is None)
print(f"DCBS results with cluster_info=null: {cluster_null}/{len(dcbs_results)}")

print("\n=== INDIVIDUAL QUESTION ANALYSIS ===")
for i, r in enumerate(greedy_results):
    confidence = max(r['answer_probs'].values())
    print(f"Q{i+1}: {confidence:.3f} confidence - {'VERY HIGH' if confidence > 0.95 else 'MEDIUM' if confidence > 0.8 else 'LOW'}") 