import json

# Load the test data
with open('data/arc_easy_processed.json', 'r') as f:
    data = json.load(f)

# Analyze the answer distribution
correct_answers = [d['answer_key'] for d in data]
answer_counts = {}
for answer in ['A', 'B', 'C', 'D']:
    answer_counts[answer] = correct_answers.count(answer)

print(f"Total questions: {len(data)}")
print(f"Answer distribution:")
for answer, count in sorted(answer_counts.items()):
    percentage = count / len(data) * 100
    print(f"  {answer}: {count} ({percentage:.1f}%)")

print(f"\nThis explains why all methods get 80% accuracy!")
print(f"The model likely defaults to answering 'B' which is correct 80% of the time.")
print(f"\nLet's check individual predictions:")

# Let's also check which questions have non-B answers
non_b_questions = [(i+1, d['id'], d['answer_key']) for i, d in enumerate(data) if d['answer_key'] != 'B']
print(f"\nQuestions with non-B answers:")
for idx, qid, answer in non_b_questions:
    print(f"  Question {idx}: {qid} - Answer: {answer}") 