import json
import glob
import os
from collections import defaultdict

# Find the most recent evaluation results
result_files = glob.glob("results/clustering_comparison/*.json")
if not result_files:
    print("No evaluation results found!")
    exit(1)

latest_file = max(result_files, key=os.path.getmtime)
print(f"Analyzing: {latest_file}\n")

with open(latest_file, 'r') as f:
    results = json.load(f)

# First, let's analyze what data was actually used
print("="*80)
print("EVALUATION SUMMARY")
print("="*80)
config = results.get('config', {})
print(f"Model: {config.get('model')}")
print(f"Examples evaluated: {config.get('examples')}")
print(f"Parameters: k={config.get('k')}, top_n={config.get('top_n')}")

# Analyze each clustering method
detailed_results = results.get('detailed_results', {})

# Track predictions across all methods
all_predictions = defaultdict(lambda: defaultdict(list))
question_results = defaultdict(lambda: defaultdict(dict))

print("\n" + "="*80)
print("DETAILED PREDICTION ANALYSIS")
print("="*80)

for method_name, method_data in detailed_results.items():
    print(f"\n\nCLUSTERING METHOD: {method_name}")
    print("-"*60)
    
    if 'detailed_results' in method_data:
        # Group by question ID
        questions = defaultdict(list)
        for result in method_data['detailed_results']:
            qid = result.get('id', 'unknown')
            questions[qid].append(result)
        
        # Analyze each question
        for qid, results_list in questions.items():
            print(f"\nQuestion ID: {qid}")
            
            # Get question details from first result
            first = results_list[0]
            correct_answer = first.get('correct_answer')
            correct_option = first.get('correct_option')
            
            # Map correct answer to A/B/C/D
            options = first.get('options', [])
            if correct_answer in options:
                correct_idx = options.index(correct_answer)
                correct_letter = chr(ord('A') + correct_idx)
            else:
                correct_letter = '?'
            
            print(f"Correct answer: {correct_letter} - {correct_answer[:50]}...")
            
            # Check predictions for each sampler
            sampler_results = {}
            for result in results_list:
                sampler = result.get('sampler')
                pred_id = result.get('pred_id')
                answer_ids = result.get('answer_ids', {})
                
                # Find which option was predicted
                pred_answer = None
                for answer, token_id in answer_ids.items():
                    if token_id == pred_id:
                        pred_answer = answer
                        if answer in options:
                            pred_idx = options.index(answer)
                            pred_letter = chr(ord('A') + pred_idx)
                        else:
                            pred_letter = '?'
                        break
                
                is_correct = result.get('correct', False)
                sampler_results[sampler] = {
                    'predicted': pred_letter,
                    'correct': is_correct
                }
                
                # Track for overall analysis
                all_predictions[sampler][pred_letter].append(qid)
                question_results[qid][method_name][sampler] = pred_letter
            
            # Print sampler predictions
            print("Predictions by sampler:")
            for sampler in ['greedy', 'top_p', 'dcbs', 'random']:
                if sampler in sampler_results:
                    res = sampler_results[sampler]
                    status = "✓" if res['correct'] else "✗"
                    print(f"  {sampler:8s}: {res['predicted']} {status}")

# Overall prediction patterns
print("\n\n" + "="*80)
print("PREDICTION PATTERNS ACROSS ALL METHODS")
print("="*80)

for sampler in ['greedy', 'top_p', 'dcbs', 'random']:
    print(f"\n{sampler.upper()} sampler predictions:")
    pred_counts = defaultdict(int)
    for letter, qids in all_predictions[sampler].items():
        pred_counts[letter] = len(qids)
    
    total = sum(pred_counts.values())
    for letter in ['A', 'B', 'C', 'D']:
        count = pred_counts.get(letter, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {letter}: {count:2d} ({pct:5.1f}%)")

# Check if predictions are identical across clustering methods
print("\n\n" + "="*80)
print("CONSISTENCY CHECK: Do all clustering methods predict the same?")
print("="*80)

identical_count = 0
total_questions = 0

for qid, methods_data in question_results.items():
    total_questions += 1
    
    # Check if all methods gave same prediction for each sampler
    all_same = True
    for sampler in ['greedy', 'top_p', 'dcbs']:
        predictions = []
        for method in ['kmeans', 'dbscan', 'hierarchical']:
            if method in methods_data and sampler in methods_data[method]:
                predictions.append(methods_data[method][sampler])
        
        if len(set(predictions)) > 1:
            all_same = False
            break
    
    if all_same:
        identical_count += 1

print(f"\nQuestions where all clustering methods gave identical predictions:")
print(f"{identical_count}/{total_questions} ({identical_count/total_questions*100:.1f}%)")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("If all clustering methods produce identical predictions, it means:")
print("1. The clustering differences don't matter for these examples")
print("2. The model is likely heavily biased towards certain answers")
print("3. The test set may still be imbalanced")
print("\nFor meaningful comparison, try:")
print("- Using a larger, balanced dataset (e.g., full ARC-Easy)")
print("- Testing on harder questions where the model is less certain")
print("- Examining cases where clustering methods disagree") 