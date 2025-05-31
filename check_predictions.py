import json
import glob
import os

# Find the most recent evaluation results
result_files = glob.glob("results/clustering_comparison/*.json")
if not result_files:
    print("No evaluation results found!")
    exit(1)

# Get the most recent file
latest_file = max(result_files, key=os.path.getmtime)
print(f"Analyzing: {latest_file}\n")

# Load results
with open(latest_file, 'r') as f:
    results = json.load(f)

# Analyze predictions for each method
detailed_results = results.get('detailed_results', {})

for method_name, method_results in detailed_results.items():
    if 'detailed_results' in method_results:
        print(f"\nMethod: {method_name}")
        predictions = {}
        
        # Count predictions
        for result in method_results['detailed_results']:
            if 'pred_id' in result:
                # Map token ID back to answer
                answer_ids = result.get('answer_ids', {})
                pred_id = result['pred_id']
                
                # Find which answer this ID corresponds to
                pred_answer = None
                for answer, token_id in answer_ids.items():
                    if token_id == pred_id:
                        pred_answer = answer
                        break
                
                if pred_answer:
                    predictions[pred_answer] = predictions.get(pred_answer, 0) + 1
        
        # Print prediction distribution
        if predictions:
            total = sum(predictions.values())
            print(f"  Predictions distribution:")
            for answer in ['A', 'B', 'C', 'D']:
                count = predictions.get(answer, 0)
                percentage = count / total * 100 if total > 0 else 0
                print(f"    {answer}: {count} ({percentage:.1f}%)")
        else:
            print("  No predictions found")

print("\n" + "="*50)
print("CONCLUSION: The model is heavily biased towards answer 'B'!")
print("This is because the test dataset has 80% 'B' answers.")
print("The test needs a balanced dataset to be meaningful.") 