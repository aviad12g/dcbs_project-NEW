#!/usr/bin/env python3
"""
Cache vs Non-Cache Performance Comparison for DCBS

This script runs DCBS evaluation with caching enabled and disabled
to compare performance differences.
"""

import time
import json
from src.evaluation_core import EvaluationConfig, EvaluationRunner, load_benchmark_data

def run_cache_comparison(benchmark_path="data/arc_easy_processed.json", limit=10):
    """Run DCBS evaluation with and without caching to compare performance."""
    
    print("=" * 60)
    print("DCBS CACHE vs NON-CACHE PERFORMANCE COMPARISON")
    print("=" * 60)
    
    # Load benchmark data
    print(f"Loading benchmark data from: {benchmark_path}")
    benchmark_data = load_benchmark_data(benchmark_path)
    
    if limit:
        benchmark_data = benchmark_data[:limit]
    
    print(f"Testing with {len(benchmark_data)} examples")
    print()
    
    results = {}
    
    # Test 1: With caching enabled (default)
    print("Running DCBS with CACHING ENABLED...")
    config_cached = EvaluationConfig(
        model_name="meta-llama/Llama-3.2-1B",
        benchmark_path=benchmark_path,
        output_dir="results/cache_test",
        limit=limit,
        enable_caching=True,
        k=8,
        top_n=50
    )
    
    start_time = time.time()
    runner_cached = EvaluationRunner(config_cached)
    # Only test DCBS
    runner_cached.samplers = {"dcbs": runner_cached.samplers["dcbs"]}
    results_cached = runner_cached.run_evaluation(benchmark_data)
    cached_time = time.time() - start_time
    
    print(f"Cached evaluation completed in {cached_time:.2f} seconds")
    
    # Test 2: With caching disabled
    print("\nRunning DCBS with CACHING DISABLED...")
    config_no_cache = EvaluationConfig(
        model_name="meta-llama/Llama-3.2-1B",
        benchmark_path=benchmark_path,
        output_dir="results/cache_test",
        limit=limit,
        enable_caching=False,
        k=8,
        top_n=50
    )
    
    start_time = time.time()
    runner_no_cache = EvaluationRunner(config_no_cache)
    # Only test DCBS
    runner_no_cache.samplers = {"dcbs": runner_no_cache.samplers["dcbs"]}
    results_no_cache = runner_no_cache.run_evaluation(benchmark_data)
    no_cache_time = time.time() - start_time
    
    print(f"Non-cached evaluation completed in {no_cache_time:.2f} seconds")
    
    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)
    
    cached_stats = results_cached["statistics"]["dcbs"]
    no_cache_stats = results_no_cache["statistics"]["dcbs"]
    
    print(f"Dataset: {len(benchmark_data)} examples")
    print(f"Model: meta-llama/Llama-3.2-1B")
    print(f"DCBS Parameters: k=8, top_n=50")
    print()
    
    print("TIMING COMPARISON:")
    print(f"  With Cache:    {cached_time:.2f}s total, {cached_stats['avg_time_ms']:.2f}ms per example")
    print(f"  Without Cache: {no_cache_time:.2f}s total, {no_cache_stats['avg_time_ms']:.2f}ms per example")
    print(f"  Time Difference: {no_cache_time - cached_time:+.2f}s ({((no_cache_time - cached_time) / cached_time * 100):+.1f}%)")
    print()
    
    print("ACCURACY COMPARISON:")
    print(f"  With Cache:    {cached_stats['accuracy']:.2f}% ({cached_stats['correct']}/{cached_stats['total']})")
    print(f"  Without Cache: {no_cache_stats['accuracy']:.2f}% ({no_cache_stats['correct']}/{no_cache_stats['total']})")
    print(f"  Accuracy Match: {'IDENTICAL' if cached_stats['accuracy'] == no_cache_stats['accuracy'] else 'DIFFERENT'}")
    print()
    
    # Cache effectiveness analysis
    cache_overhead = no_cache_stats['avg_time_ms'] - cached_stats['avg_time_ms']
    print("CACHE EFFECTIVENESS:")
    print(f"  Per-example overhead: {cache_overhead:+.2f}ms")
    if cache_overhead > 0:
        print(f"  Cache provides {cache_overhead:.2f}ms speedup per example")
    else:
        print(f"  Cache adds {abs(cache_overhead):.2f}ms overhead per example")
    print()
    
    # Save detailed results
    comparison_results = {
        "test_config": {
            "benchmark": benchmark_path,
            "examples": len(benchmark_data),
            "model": "meta-llama/Llama-3.2-1B",
            "dcbs_params": {"k": 8, "top_n": 50}
        },
        "cached": {
            "total_time_s": cached_time,
            "stats": cached_stats
        },
        "no_cache": {
            "total_time_s": no_cache_time,
            "stats": no_cache_stats
        },
        "comparison": {
            "time_difference_s": no_cache_time - cached_time,
            "time_difference_percent": ((no_cache_time - cached_time) / cached_time * 100),
            "accuracy_match": cached_stats['accuracy'] == no_cache_stats['accuracy'],
            "cache_overhead_ms": cache_overhead
        }
    }
    
    output_file = "results/cache_comparison_results.json"
    with open(output_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"Detailed results saved to: {output_file}")
    print("=" * 60)
    
    return comparison_results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare DCBS cache vs non-cache performance")
    parser.add_argument("--benchmark", default="data/arc_easy_processed.json", help="Benchmark file path")
    parser.add_argument("--limit", type=int, default=10, help="Number of examples to test")
    
    args = parser.parse_args()
    
    run_cache_comparison(args.benchmark, args.limit) 