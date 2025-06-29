#!/usr/bin/env python3

import json

def analyze_disagreements():
    # Load the results
    with open('results/evaluation_results_20250629_143353.json', 'r') as f:
        results = json.load(f)
    
    print('=== DCBS vs GREEDY DISAGREEMENT ANALYSIS ===')
    print()
    
    # Check detailed results
    detailed_results = results.get('detailed_results', [])
    
    dcbs_wins = 0
    greedy_wins = 0
    both_wrong = 0
    total_disagreements = 0
    
    disagreement_examples = []
    
    print(f"Found {len(detailed_results)} detailed results")
    
    # Debug first few examples
    if detailed_results:
        first = detailed_results[0]
        print(f"First example structure: {list(first.keys())}")
        if 'predictions' in first:
            print(f"Prediction structure: {list(first['predictions'].keys())}")
    
    # Group results by example ID since we have 4000 results (2000 examples × 2 samplers)
    examples_by_id = {}
    
    for result in detailed_results:
        example_id = result.get('id')
        sampler = result.get('sampler')
        predicted_answer = result.get('predicted_answer')
        correct_answer = result.get('correct_answer')
        answer_probs = result.get('answer_probs', {})
        
        if example_id not in examples_by_id:
            examples_by_id[example_id] = {'correct_answer': correct_answer}
        
        examples_by_id[example_id][sampler] = {
            'answer': predicted_answer,
            'probs': answer_probs
        }
    
    # Now compare DCBS vs Greedy for each example
    for example_id, data in examples_by_id.items():
        dcbs_data = data.get('dcbs', {})
        greedy_data = data.get('greedy', {})
        correct_answer = data.get('correct_answer')
        
        # Skip if we don't have both predictions
        if not dcbs_data or not greedy_data:
            continue
            
        dcbs_pred = dcbs_data.get('answer')
        greedy_pred = greedy_data.get('answer')
        dcbs_probs = dcbs_data.get('probs', {})
        greedy_probs = greedy_data.get('probs', {})
            
        if dcbs_pred != greedy_pred:
            total_disagreements += 1
            
            dcbs_correct = (dcbs_pred == correct_answer)
            greedy_correct = (greedy_pred == correct_answer)
            
            if dcbs_correct and not greedy_correct:
                dcbs_wins += 1
                winner = "DCBS"
            elif greedy_correct and not dcbs_correct:
                greedy_wins += 1
                winner = "GREEDY"
            else:
                both_wrong += 1
                winner = "BOTH_WRONG"
            
            # Get confidence (probability of selected answer)
            dcbs_confidence = dcbs_probs.get(dcbs_pred, 0.0) if dcbs_probs else 0.0
            greedy_confidence = greedy_probs.get(greedy_pred, 0.0) if greedy_probs else 0.0
            
            disagreement_examples.append({
                'id': example_id,
                'winner': winner,
                'dcbs': dcbs_pred,
                'greedy': greedy_pred,
                'correct': correct_answer,
                'dcbs_confidence': dcbs_confidence,
                'greedy_confidence': greedy_confidence
            })
    
    print(f'Total disagreements: {total_disagreements}')
    print(f'DCBS wins (DCBS right, Greedy wrong): {dcbs_wins}')
    print(f'GREEDY wins (Greedy right, DCBS wrong): {greedy_wins}')
    print(f'Both wrong: {both_wrong}')
    print()
    
    if dcbs_wins + greedy_wins > 0:
        print(f'When one method is right and the other wrong:')
        print(f'  DCBS win rate: {dcbs_wins}/{dcbs_wins + greedy_wins} = {dcbs_wins/(dcbs_wins + greedy_wins)*100:.1f}%')
        print(f'  GREEDY win rate: {greedy_wins}/{dcbs_wins + greedy_wins} = {greedy_wins/(dcbs_wins + greedy_wins)*100:.1f}%')
    print()
    
    # Analyze confidence patterns
    print('=== CONFIDENCE ANALYSIS ===')
    
    # Calculate average confidence by outcome
    dcbs_win_confidences = []
    greedy_win_confidences = []
    both_wrong_dcbs_confidences = []
    both_wrong_greedy_confidences = []
    
    for ex in disagreement_examples:
        if ex['winner'] == 'DCBS':
            dcbs_win_confidences.append((ex['dcbs_confidence'], ex['greedy_confidence']))
        elif ex['winner'] == 'GREEDY':
            greedy_win_confidences.append((ex['dcbs_confidence'], ex['greedy_confidence']))
        else:  # BOTH_WRONG
            both_wrong_dcbs_confidences.append(ex['dcbs_confidence'])
            both_wrong_greedy_confidences.append(ex['greedy_confidence'])
    
    if dcbs_win_confidences:
        avg_dcbs_when_dcbs_wins = sum(conf[0] for conf in dcbs_win_confidences) / len(dcbs_win_confidences)
        avg_greedy_when_dcbs_wins = sum(conf[1] for conf in dcbs_win_confidences) / len(dcbs_win_confidences)
        print(f'When DCBS wins ({len(dcbs_win_confidences)} cases):')
        print(f'  DCBS confidence: {avg_dcbs_when_dcbs_wins:.3f}')
        print(f'  Greedy confidence: {avg_greedy_when_dcbs_wins:.3f}')
        print()
    
    if greedy_win_confidences:
        avg_dcbs_when_greedy_wins = sum(conf[0] for conf in greedy_win_confidences) / len(greedy_win_confidences)
        avg_greedy_when_greedy_wins = sum(conf[1] for conf in greedy_win_confidences) / len(greedy_win_confidences)
        print(f'When GREEDY wins ({len(greedy_win_confidences)} cases):')
        print(f'  DCBS confidence: {avg_dcbs_when_greedy_wins:.3f}')
        print(f'  Greedy confidence: {avg_greedy_when_greedy_wins:.3f}')
        print()
    
    if both_wrong_dcbs_confidences:
        avg_dcbs_both_wrong = sum(both_wrong_dcbs_confidences) / len(both_wrong_dcbs_confidences)
        avg_greedy_both_wrong = sum(both_wrong_greedy_confidences) / len(both_wrong_greedy_confidences)
        print(f'When both wrong ({len(both_wrong_dcbs_confidences)} cases):')
        print(f'  DCBS confidence: {avg_dcbs_both_wrong:.3f}')
        print(f'  Greedy confidence: {avg_greedy_both_wrong:.3f}')
        print()
    
    # Show sample disagreements with confidence
    print('=== SAMPLE DISAGREEMENTS WITH CONFIDENCE ===')
    for i, ex in enumerate(disagreement_examples[:8]):
        dcbs_conf = ex['dcbs_confidence']
        greedy_conf = ex['greedy_confidence']
        print(f'{i+1}. {ex["winner"]}: DCBS={ex["dcbs"]} ({dcbs_conf:.3f}), GREEDY={ex["greedy"]} ({greedy_conf:.3f}), CORRECT={ex["correct"]}')
    
    return {
        'total_disagreements': total_disagreements,
        'dcbs_wins': dcbs_wins,
        'greedy_wins': greedy_wins,
        'both_wrong': both_wrong,
        'examples': disagreement_examples
    }

if __name__ == "__main__":
    analyze_disagreements() 