"""
Verification script to test evaluation methodology and check for data leakage/artifacts.
"""

import json
import random
from collections import Counter
from pathlib import Path

def test_uniform_guess_sanity():
    """Test: Always pick 'A' should give exactly 25% on multiple choice."""
    print("🔍 **TEST 1: Uniform Guess Sanity Check**")
    
    # Load the data
    with open('data/arc_easy_full.json', 'r') as f:
        questions = json.load(f)
    
    # Count correct answers
    answer_distribution = Counter()
    correct_with_A = 0
    
    for q in questions:
        correct_answer = q['answer_key']
        answer_distribution[correct_answer] += 1
        if correct_answer == 'A':
            correct_with_A += 1
    
    always_A_accuracy = correct_with_A / len(questions)
    
    print(f"  Total questions: {len(questions):,}")
    print(f"  Always pick 'A' accuracy: {always_A_accuracy:.1%}")
    print(f"  Expected: ~25% (should be close)")
    print(f"  Answer distribution: {dict(answer_distribution)}")
    
    if abs(always_A_accuracy - 0.25) > 0.05:
        print(f"  ⚠️  WARNING: Always-A accuracy ({always_A_accuracy:.1%}) far from 25%!")
    else:
        print(f"  ✅ Always-A accuracy looks reasonable")
    
    return always_A_accuracy

def analyze_question_patterns():
    """Look for repetitive patterns that could inflate cache hit rates."""
    print(f"\n🔍 **TEST 2: Question Pattern Analysis**")
    
    with open('data/arc_easy_full.json', 'r') as f:
        questions = json.load(f)
    
    # Check for duplicate questions
    question_texts = [q['question'] for q in questions]
    unique_questions = set(question_texts)
    
    print(f"  Total questions: {len(questions):,}")
    print(f"  Unique questions: {len(unique_questions):,}")
    print(f"  Duplicate rate: {(1 - len(unique_questions)/len(questions)):.1%}")
    
    # Check question length distribution
    lengths = [len(q['question'].split()) for q in questions]
    avg_length = sum(lengths) / len(lengths)
    
    print(f"  Average question length: {avg_length:.1f} words")
    print(f"  Min/Max length: {min(lengths)}/{max(lengths)} words")
    
    # Sample a few questions
    print(f"\n  📝 **Sample Questions:**")
    for i, q in enumerate(random.sample(questions, 3)):
        print(f"    {i+1}. Q: {q['question'][:100]}...")
        print(f"       Correct: {q['answer_key']}")

def check_timing_measurement():
    """Verify what our timing is actually measuring."""
    print(f"\n🔍 **TEST 3: Timing Measurement Analysis**")
    
    # Load our results
    with open('results/arc_llama_full_evaluation.json', 'r') as f:
        data = json.load(f)
    
    stats = data['statistics']
    
    print(f"  📊 **Reported Timings:**")
    for method, stat in stats.items():
        avg_time = stat['avg_time_ms']
        print(f"    {method:10s}: {avg_time:6.1f}ms average")
    
    # Check for suspicious patterns
    random_time = stats['random']['avg_time_ms']
    if random_time < 0.001:
        print(f"  ⚠️  Random method shows {random_time:.3f}ms - likely measuring post-processing only!")
    
    # Calculate expected inference time
    total_questions = stats['greedy']['total']
    config = data['config']
    
    print(f"\n  🧮 **Inference Time Reality Check:**")
    print(f"    Total questions: {total_questions:,}")
    print(f"    Reported evaluation time: ~27 minutes (1,625 seconds)")
    print(f"    Wall-clock per question: {1625/total_questions:.2f} seconds")
    print(f"    That's {1625/total_questions*1000:.0f}ms per question - vs reported ~1-6ms!")
    print(f"    ➡️  This confirms we're only timing post-processing, not inference!")

def sample_question_predictions():
    """Sample some questions and show what each method predicted."""
    print(f"\n🔍 **TEST 4: Sample Predictions Analysis**")
    
    with open('results/arc_llama_full_evaluation.json', 'r') as f:
        data = json.load(f)
    
    individual_results = data['individual_results']
    
    print(f"  📝 **Sample Question Predictions:**")
    
    # Get a few random questions
    sample_questions = random.sample(individual_results, 5)
    
    for i, result in enumerate(sample_questions):
        question = result['question'][:80] + "..." if len(result['question']) > 80 else result['question']
        correct = result['correct_answer']
        
        print(f"\n    Question {i+1}: {question}")
        print(f"    Correct Answer: {correct}")
        
        predictions = result['results']
        for method in ['greedy', 'dcbs', 'top_p', 'random']:
            pred = predictions[method]['predicted']
            is_correct = predictions[method]['correct']
            status = "✅" if is_correct else "❌"
            print(f"      {method:8s}: {pred} {status}")

def main():
    """Run all verification tests."""
    print("🚨 **REALITY CHECK: Verifying Evaluation Claims** 🚨\n")
    
    # Run tests
    always_A_acc = test_uniform_guess_sanity()
    analyze_question_patterns()
    check_timing_measurement()
    sample_question_predictions()
    
    print(f"\n📋 **SUMMARY OF VERIFICATION:**")
    print(f"  1. Always-A accuracy: {always_A_acc:.1%} (should be ~25%)")
    print(f"  2. Timing measurement: Post-processing only (not wall-clock inference)")
    print(f"  3. 41% accuracy claim: Needs verification against other benchmarks")
    print(f"  4. DCBS cache efficiency: Could be legitimate but needs timing correction")
    
    print(f"\n🎯 **NEXT STEPS FOR PROPER VERIFICATION:**")
    print(f"  ☐ Test on ARC-Challenge (harder split)")
    print(f"  ☐ Test on WinoGrande (different task)")
    print(f"  ☐ Measure wall-clock inference time properly")
    print(f"  ☐ Compare against published 1B model benchmarks")
    print(f"  ☐ Check if model was fine-tuned on reasoning data")

if __name__ == "__main__":
    main() 