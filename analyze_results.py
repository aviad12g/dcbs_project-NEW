import csv
import sys

def analyze_results(csv_path, output_path=None):
    results = {
        'greedy': {'correct': 0, 'total': 0, 'avg_time': 0},
        'top-p': {'correct': 0, 'total': 0, 'avg_time': 0},
        'dcbs': {'correct': 0, 'total': 0, 'avg_time': 0},
        'random': {'correct': 0, 'total': 0, 'avg_time': 0}
    }
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            correct = int(row['correct'])
            elapsed_ms = float(row['elapsed_ms'])
            
            results[method]['correct'] += correct
            results[method]['total'] += 1
            results[method]['avg_time'] += elapsed_ms
    
    # Prepare output
    output_lines = []
    output_lines.append(f"Results from {csv_path}:")
    output_lines.append("=" * 60)
    output_lines.append(f"{'Method':<10} {'Accuracy':<15} {'Avg Time (ms)':<15}")
    output_lines.append("-" * 60)
    
    for method, stats in results.items():
        accuracy = (stats['correct'] / stats['total']) * 100 if stats['total'] > 0 else 0
        avg_time = stats['avg_time'] / stats['total'] if stats['total'] > 0 else 0
        output_lines.append(f"{method:<10} {accuracy:<15.2f} {avg_time:<15.2f}")
    
    output_lines.append("=" * 60)
    
    output_text = "\n".join(output_lines)
    
    # Write to file if output_path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(output_text)
    
    # Always print to console
    print(output_text)
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <path_to_csv> [output_file]")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyze_results(csv_path, output_path) 