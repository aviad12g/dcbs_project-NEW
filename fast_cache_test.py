#!/usr/bin/env python3
"""
Fast DCBS Cache Performance Test

Simulates evaluation performance without loading heavy models.
"""

import time
import torch
import json
from dcbs import DCBSSampler, SamplingContext

def create_mock_evaluation_scenario():
    """Create a realistic but fast evaluation scenario."""
    
    # Simulate ARC Easy-like scenario
    vocab_size = 50000  # Realistic vocab size
    embedding_dim = 4096  # Llama-like embedding dimension
    
    # Create embedding layer (this is fast)
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    
    # Create sampling context
    context = SamplingContext(
        embedding_layer=embedding_layer,
        device=torch.device("cpu")
    )
    
    # Simulate 500 different evaluation scenarios
    torch.manual_seed(42)
    scenarios = []
    for i in range(500):
        # Each scenario has different logits and filter tokens
        logits = torch.randn(vocab_size) * 2.0 + torch.randn(1) * 0.5  # Add variation
        
        # Simulate multiple choice answers (4 options)
        base_tokens = [1000 + i*10, 2000 + i*10, 3000 + i*10, 4000 + i*10]
        filter_tokens = set(base_tokens)
        
        scenarios.append({
            'logits': logits,
            'filter_tokens': filter_tokens,
            'id': f"arc_easy_{i}"
        })
    
    return context, scenarios

def run_fast_cache_comparison():
    """Run fast cache comparison test."""
    
    print("=" * 60)
    print("FAST DCBS CACHE PERFORMANCE TEST")
    print("Simulating 500 ARC Easy Examples")
    print("=" * 60)
    
    # Create test scenario
    context, scenarios = create_mock_evaluation_scenario()
    
    print(f"Created {len(scenarios)} test scenarios")
    print(f"Vocab size: {context.embedding_layer.num_embeddings}")
    print(f"Embedding dim: {context.embedding_layer.embedding_dim}")
    print()
    
    # DCBS parameters
    k = 8
    top_n = 50
    
    # Test 1: With caching enabled
    print("Running DCBS with CACHING ENABLED...")
    sampler_cached = DCBSSampler.create_default(k=k, top_n=top_n, enable_caching=True)
    
    start_time = time.time()
    cached_results = []
    cached_times = []
    
    for i, scenario in enumerate(scenarios):
        iter_start = time.time()
        result = sampler_cached.sample(
            scenario['logits'], 
            context, 
            filter_tokens=scenario['filter_tokens']
        )
        iter_time = (time.time() - iter_start) * 1000
        
        cached_results.append(result)
        cached_times.append(iter_time)
        
        if (i + 1) % 50 == 0:
            print(f"  Completed {i+1}/500 examples, avg time: {sum(cached_times[-50:]) / 50:.2f}ms")
    
    total_cached_time = time.time() - start_time
    avg_cached_time = sum(cached_times) / len(cached_times)
    
    print(f"Cached evaluation completed in {total_cached_time:.2f}s")
    print(f"Average per example: {avg_cached_time:.2f}ms")
    print()
    
    # Get cache stats
    cache_stats = sampler_cached.get_cache_stats()
    print(f"Cache stats: {cache_stats}")
    print()
    
    # Test 2: With caching disabled
    print("Running DCBS with CACHING DISABLED...")
    sampler_no_cache = DCBSSampler.create_no_cache(k=k, top_n=top_n)
    
    start_time = time.time()
    no_cache_results = []
    no_cache_times = []
    
    for i, scenario in enumerate(scenarios):
        iter_start = time.time()
        result = sampler_no_cache.sample(
            scenario['logits'], 
            context, 
            filter_tokens=scenario['filter_tokens']
        )
        iter_time = (time.time() - iter_start) * 1000
        
        no_cache_results.append(result)
        no_cache_times.append(iter_time)
        
        if (i + 1) % 50 == 0:
            print(f"  Completed {i+1}/500 examples, avg time: {sum(no_cache_times[-50:]) / 50:.2f}ms")
    
    total_no_cache_time = time.time() - start_time
    avg_no_cache_time = sum(no_cache_times) / len(no_cache_times)
    
    print(f"Non-cached evaluation completed in {total_no_cache_time:.2f}s")
    print(f"Average per example: {avg_no_cache_time:.2f}ms")
    print()
    
    # Compare results
    print("=" * 60)
    print("PERFORMANCE COMPARISON RESULTS")
    print("=" * 60)
    
    print(f"Dataset: {len(scenarios)} ARC Easy examples (simulated)")
    print(f"DCBS Parameters: k={k}, top_n={top_n}")
    print()
    
    print("TIMING COMPARISON:")
    print(f"  With Cache:    {total_cached_time:.2f}s total, {avg_cached_time:.2f}ms per example")
    print(f"  Without Cache: {total_no_cache_time:.2f}s total, {avg_no_cache_time:.2f}ms per example")
    
    time_diff = total_no_cache_time - total_cached_time
    percent_diff = (time_diff / total_cached_time) * 100
    
    print(f"  Time Difference: {time_diff:+.2f}s ({percent_diff:+.1f}%)")
    print()
    
    # Per-example analysis
    cache_overhead = avg_no_cache_time - avg_cached_time
    print("CACHE EFFECTIVENESS:")
    print(f"  Per-example difference: {cache_overhead:+.2f}ms")
    if cache_overhead > 0:
        print(f"  Cache provides {cache_overhead:.2f}ms speedup per example")
    else:
        print(f"  Cache adds {abs(cache_overhead):.2f}ms overhead per example")
    print()
    
    # Consistency check
    matches = sum(1 for a, b in zip(cached_results, no_cache_results) if a == b)
    print("CONSISTENCY CHECK:")
    print(f"  Results match: {matches}/{len(scenarios)} ({matches/len(scenarios)*100:.1f}%)")
    print()
    
    # Save results
    results = {
        "test_config": {
            "scenarios": len(scenarios),
            "vocab_size": context.embedding_layer.num_embeddings,
            "embedding_dim": context.embedding_layer.embedding_dim,
            "dcbs_params": {"k": k, "top_n": top_n}
        },
        "timing": {
            "cached_total_s": total_cached_time,
            "no_cache_total_s": total_no_cache_time,
            "cached_avg_ms": avg_cached_time,
            "no_cache_avg_ms": avg_no_cache_time,
            "time_difference_s": time_diff,
            "time_difference_percent": percent_diff,
            "cache_overhead_ms": cache_overhead
        },
        "consistency": {
            "matches": matches,
            "total": len(scenarios),
            "match_rate": matches/len(scenarios)
        },
        "cache_stats": cache_stats
    }
    
    with open("results/fast_cache_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Detailed results saved to: results/fast_cache_test_results.json")
    print("=" * 60)
    
    return results

if __name__ == "__main__":
    run_fast_cache_comparison() 