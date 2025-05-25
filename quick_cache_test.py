#!/usr/bin/env python3
"""
Quick DCBS Cache Performance Test

Simple test to demonstrate cache vs no-cache performance difference.
"""

import time
import torch
from dcbs import DCBSSampler, SamplingContext

def quick_cache_test():
    """Quick test of DCBS cache performance."""
    
    print("=" * 50)
    print("QUICK DCBS CACHE PERFORMANCE TEST")
    print("=" * 50)
    
    # Create mock embedding layer
    vocab_size = 1000
    embedding_dim = 768
    
    # Create a simple embedding layer
    embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
    
    # Create sampling context
    context = SamplingContext(
        embedding_layer=embedding_layer,
        device=torch.device("cpu")
    )
    
    # Create test logits
    torch.manual_seed(42)
    test_logits = torch.randn(vocab_size) * 2.0
    
    # Filter tokens (simulating multiple choice)
    filter_tokens = {100, 200, 300, 400}
    
    print(f"Testing with {vocab_size} vocab size")
    print(f"Filter tokens: {filter_tokens}")
    print()
    
    # Test parameters
    num_iterations = 5
    k = 8
    top_n = 50
    
    # Test 1: With caching
    print("Testing DCBS with CACHING ENABLED...")
    sampler_cached = DCBSSampler.create_default(k=k, top_n=top_n, enable_caching=True)
    
    cached_times = []
    for i in range(num_iterations):
        start_time = time.time()
        result_cached = sampler_cached.sample(test_logits, context, filter_tokens=filter_tokens)
        elapsed = (time.time() - start_time) * 1000
        cached_times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f}ms -> token {result_cached}")
    
    avg_cached = sum(cached_times) / len(cached_times)
    print(f"Average with cache: {avg_cached:.2f}ms")
    print()
    
    # Test 2: Without caching
    print("Testing DCBS with CACHING DISABLED...")
    sampler_no_cache = DCBSSampler.create_no_cache(k=k, top_n=top_n)
    
    no_cache_times = []
    for i in range(num_iterations):
        start_time = time.time()
        result_no_cache = sampler_no_cache.sample(test_logits, context, filter_tokens=filter_tokens)
        elapsed = (time.time() - start_time) * 1000
        no_cache_times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.2f}ms -> token {result_no_cache}")
    
    avg_no_cache = sum(no_cache_times) / len(no_cache_times)
    print(f"Average without cache: {avg_no_cache:.2f}ms")
    print()
    
    # Results
    print("=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"With Cache:    {avg_cached:.2f}ms average")
    print(f"Without Cache: {avg_no_cache:.2f}ms average")
    
    diff = avg_no_cache - avg_cached
    percent = (diff / avg_cached) * 100 if avg_cached > 0 else 0
    
    print(f"Difference:    {diff:+.2f}ms ({percent:+.1f}%)")
    
    if diff > 0:
        print(f"Cache provides {diff:.2f}ms speedup per sample")
    else:
        print(f"Cache adds {abs(diff):.2f}ms overhead per sample")
    
    print(f"Results consistent: {result_cached == result_no_cache}")
    print("=" * 50)

if __name__ == "__main__":
    quick_cache_test() 