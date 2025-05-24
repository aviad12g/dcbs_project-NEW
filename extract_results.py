import json

data = json.load(open("results/arc_full_evaluation.json"))
stats = data["statistics"]

print("=== FULL ARC EASY RESULTS (2,946 questions) ===")
print(
    f"DCBS:   {stats['dcbs']['accuracy']*100:.1f}% ({stats['dcbs']['correct']}/{stats['dcbs']['total']})"
)
print(
    f"Greedy: {stats['greedy']['accuracy']*100:.1f}% ({stats['greedy']['correct']}/{stats['greedy']['total']})"
)
print(
    f"Top-p:  {stats['top_p']['accuracy']*100:.1f}% ({stats['top_p']['correct']}/{stats['top_p']['total']})"
)
print(
    f"Random: {stats['random']['accuracy']*100:.1f}% ({stats['random']['correct']}/{stats['random']['total']})"
)

print("\nDCBS vs others:")
dcbs_acc = stats["dcbs"]["accuracy"] * 100
greedy_acc = stats["greedy"]["accuracy"] * 100
random_acc = stats["random"]["accuracy"] * 100

print(f"DCBS vs Greedy: +{dcbs_acc - greedy_acc:.1f} percentage points")
print(f"DCBS vs Random: +{dcbs_acc - random_acc:.1f} percentage points")
