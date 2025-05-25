"""
Debug script to identify why sampling methods are giving identical results.
"""

import time
import torch
import numpy as np
from dcbs_optimized import create_optimized_samplers, SamplingContext

def create_test_embedding_layer():
    """Create a simple test embedding layer."""
    class TestEmbedding:
        def __init__(self):
            self.embedding_dim = 512
            self.weight = torch.randn(1000, 512)
        
        def __call__(self, token_ids):
            return self.weight[token_ids]
    
    return TestEmbedding()

def debug_sampling_behavior():
    """Debug why all samplers give identical results."""
    
    print("🔍 **DEBUGGING SAMPLING BEHAVIOR**")
    print("=" * 50)
    
    # Setup
    device = torch.device("cpu")
    embedding_layer = create_test_embedding_layer()
    context = SamplingContext(embedding_layer=embedding_layer, device=device)
    
    # Create samplers
    samplers = create_optimized_samplers()
    
    # Test with specific logits that should give different results
    print("\n📊 **Test 1: Clear preference for token 0**")
    logits = torch.tensor([10.0, 1.0, 1.0, 1.0])  # Strong preference for token 0
    filter_tokens = {0, 1, 2, 3}
    
    results = {}
    for name, sampler in samplers.items():
        start_time = time.time()
        result = sampler.sample(logits, filter_tokens, context)
        elapsed = (time.time() - start_time) * 1000
        
        results[name] = {"token": result, "time_ms": elapsed}
        print(f"{name:15} | Token: {result} | Time: {elapsed:.3f}ms")
    
    # Check if all results are identical
    tokens = [r["token"] for r in results.values()]
    if len(set(tokens)) == 1:
        print("❌ **PROBLEM: All samplers returned the same token!**")
    else:
        print("✅ **GOOD: Samplers returned different tokens**")
    
    print("\n📊 **Test 2: Uniform logits (should vary with stochastic methods)**")
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Uniform probabilities
    
    # Run multiple times to see variation
    for run in range(3):
        print(f"\nRun {run + 1}:")
        for name, sampler in samplers.items():
            result = sampler.sample(logits, filter_tokens, context)
            print(f"  {name:15} | Token: {result}")

def debug_timing_measurement():
    """Debug timing measurement accuracy."""
    
    print("\n⏱️  **DEBUGGING TIMING MEASUREMENT**")
    print("=" * 50)
    
    # Test basic timing accuracy
    def dummy_operation():
        time.sleep(0.001)  # 1ms sleep
        return 42
    
    print("Testing 1ms sleep operation:")
    times = []
    for _ in range(10):
        start = time.time()
        result = dummy_operation()
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  Measured: {elapsed:.3f}ms")
    
    avg_time = np.mean(times)
    print(f"Average: {avg_time:.3f}ms (should be ~1ms)")
    
    if avg_time < 0.5:
        print("❌ **PROBLEM: Timer resolution too low!**")
    else:
        print("✅ **GOOD: Timer working correctly**")

def debug_model_inference_time():
    """Debug actual model inference timing."""
    
    print("\n🤖 **DEBUGGING MODEL INFERENCE TIME**")
    print("=" * 50)
    
    # Simulate model inference
    vocab_size = 32000
    
    def simulate_model_forward():
        # Simulate what happens in real model inference
        logits = torch.randn(vocab_size)
        probs = torch.softmax(logits, dim=0)
        return logits
    
    print("Testing simulated model forward pass:")
    times = []
    for _ in range(5):
        start = time.time()
        logits = simulate_model_forward()
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        print(f"  Forward pass: {elapsed:.3f}ms")
    
    avg_time = np.mean(times)
    print(f"Average forward pass: {avg_time:.3f}ms")
    
    if avg_time < 0.1:
        print("⚠️  **WARNING: Simulated inference very fast, real model will be slower**")

def debug_dcbs_clustering():
    """Debug DCBS clustering behavior."""
    
    print("\n🔧 **DEBUGGING DCBS CLUSTERING**")
    print("=" * 50)
    
    from dcbs_optimized import FastPyTorchClusterer
    
    # Test clustering with known data
    embeddings = torch.randn(50, 512)
    clusterer = FastPyTorchClusterer(k=8)
    
    print("Testing clustering:")
    start = time.time()
    labels = clusterer.cluster(embeddings)
    elapsed = (time.time() - start) * 1000
    
    unique_labels = torch.unique(labels)
    print(f"  Clustering time: {elapsed:.3f}ms")
    print(f"  Clusters found: {len(unique_labels)}/8")
    print(f"  Label distribution: {[int((labels == i).sum()) for i in unique_labels]}")
    
    if elapsed < 0.1:
        print("❌ **PROBLEM: Clustering too fast, might not be working**")
    elif elapsed > 100:
        print("❌ **PROBLEM: Clustering too slow**")
    else:
        print("✅ **GOOD: Clustering time reasonable**")

def debug_cache_behavior():
    """Debug cache hit rate behavior."""
    
    print("\n💾 **DEBUGGING CACHE BEHAVIOR**")
    print("=" * 50)
    
    from dcbs_optimized import OptimizedDCBSSampler
    
    # Test cache with fresh sampler
    sampler = OptimizedDCBSSampler(k=8, top_n=50, use_cache=True)
    embedding_layer = create_test_embedding_layer()
    context = SamplingContext(embedding_layer=embedding_layer, device=torch.device("cpu"))
    
    logits = torch.randn(1000)
    filter_tokens = set(range(50))
    
    print("Testing cache behavior over multiple calls:")
    for i in range(5):
        result = sampler.sample(logits, filter_tokens, context)
        stats = sampler.get_cache_stats()
        hit_rate = stats.get("hit_rate", 0) * 100
        print(f"  Call {i+1}: Token {result}, Hit rate: {hit_rate:.1f}%")
    
    final_stats = sampler.get_cache_stats()
    print(f"\nFinal cache stats: {final_stats}")

def main():
    """Run all debugging tests."""
    
    print("🚨 **DEBUGGING NONSENSICAL RESULTS**")
    print("=" * 60)
    
    # Run all debug tests
    debug_timing_measurement()
    debug_model_inference_time()
    debug_sampling_behavior()
    debug_dcbs_clustering()
    debug_cache_behavior()
    
    print("\n🎯 **SUMMARY & NEXT STEPS:**")
    print("=" * 60)
    print("1. Check if timing measurements are accurate")
    print("2. Verify samplers produce different results")
    print("3. Confirm DCBS clustering is working")
    print("4. Validate cache behavior is realistic")
    print("5. Test with real model inference timing")

if __name__ == "__main__":
    main() 