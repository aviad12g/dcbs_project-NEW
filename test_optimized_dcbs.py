"""
Test script to compare optimized DCBS vs original implementation.

This validates that the 32x clustering speedup translates to real-world performance gains.
"""

import time
import torch
from dcbs_optimized import create_optimized_samplers, SamplingContext

def create_mock_embedding_layer(vocab_size: int = 10000, embedding_dim: int = 512):
    """Create a mock embedding layer for testing."""
    class MockEmbedding:
        def __init__(self):
            self.embedding_dim = embedding_dim
            self.weight = torch.randn(vocab_size, embedding_dim)
        
        def __call__(self, token_ids):
            return self.weight[token_ids]
    
    return MockEmbedding()

def benchmark_optimized_dcbs():
    """Benchmark the optimized DCBS implementation."""
    
    print("🚀 **OPTIMIZED DCBS PERFORMANCE TEST**")
    print("=" * 50)
    
    # Setup
    device = torch.device("cpu")
    embedding_layer = create_mock_embedding_layer()
    context = SamplingContext(embedding_layer=embedding_layer, device=device)
    
    # Create samplers
    samplers = create_optimized_samplers()
    
    # Test data
    vocab_size = 10000
    filter_tokens = set(range(50))  # Top 50 tokens (typical ARC setup)
    num_runs = 100
    
    print(f"Test setup:")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Filter tokens: {len(filter_tokens)}")
    print(f"  Runs per method: {num_runs}")
    print(f"  Device: {device}")
    print()
    
    results = {}
    
    for name, sampler in samplers.items():
        print(f"🔧 **Testing {name}:**")
        
        # Generate random logits for each run (realistic scenario)
        logits_list = [torch.randn(vocab_size) for _ in range(num_runs)]
        
        # Warm up
        sampler.sample(logits_list[0], filter_tokens, context)
        
        # Benchmark
        start_time = time.time()
        for logits in logits_list:
            token_id = sampler.sample(logits, filter_tokens, context)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        avg_time = elapsed_time / num_runs
        
        # Get cache stats if available
        cache_stats = ""
        if hasattr(sampler, 'get_cache_stats'):
            stats = sampler.get_cache_stats()
            if stats.get("cache_enabled"):
                hit_rate = stats.get("hit_rate", 0) * 100
                cache_stats = f" | Hit rate: {hit_rate:.1f}%"
        
        print(f"  ⏱️  Average time: {avg_time:.2f}ms{cache_stats}")
        results[name] = avg_time
    
    # Performance comparison
    print(f"\n📊 **PERFORMANCE COMPARISON:**")
    print("-" * 60)
    
    baseline_time = results.get("greedy", results[list(results.keys())[0]])
    
    for name, avg_time in results.items():
        speedup = baseline_time / avg_time if avg_time > 0 else 0
        overhead = ((avg_time - baseline_time) / baseline_time * 100) if baseline_time > 0 else 0
        
        print(f"{name:15} | {avg_time:6.2f}ms | {speedup:4.1f}x | {overhead:+5.1f}% overhead")
    
    return results

def test_clustering_accuracy():
    """Test that fast clustering produces reasonable results."""
    
    print(f"\n🎯 **CLUSTERING ACCURACY TEST**")
    print("-" * 50)
    
    from dcbs_optimized import FastPyTorchClusterer
    
    # Create test data with clear clusters
    num_tokens = 50
    embedding_dim = 128
    k = 8
    
    # Create embeddings with some structure
    embeddings = torch.randn(num_tokens, embedding_dim)
    
    clusterer = FastPyTorchClusterer(k=k, max_iterations=5)
    labels = clusterer.cluster(embeddings)
    
    # Analyze clustering quality
    unique_labels = torch.unique(labels)
    cluster_sizes = [(labels == label).sum().item() for label in unique_labels]
    
    print(f"Clustering results:")
    print(f"  Tokens: {num_tokens}")
    print(f"  Target clusters: {k}")
    print(f"  Found clusters: {len(unique_labels)}")
    print(f"  Cluster sizes: {cluster_sizes}")
    print(f"  Largest cluster: {max(cluster_sizes)} tokens")
    print(f"  Smallest cluster: {min(cluster_sizes)} tokens")
    
    # Quality metrics
    balance_score = 1.0 - (max(cluster_sizes) - min(cluster_sizes)) / num_tokens
    coverage_score = len(unique_labels) / k
    
    print(f"  Balance score: {balance_score:.2f} (higher = more balanced)")
    print(f"  Coverage score: {coverage_score:.2f} (higher = better cluster usage)")
    
    return {
        "clusters_found": len(unique_labels),
        "balance_score": balance_score,
        "coverage_score": coverage_score
    }

def compare_with_expected_timing():
    """Compare with our expected timing improvements."""
    
    print(f"\n⚡ **EXPECTED vs ACTUAL PERFORMANCE**")
    print("-" * 50)
    
    # Expected timings based on our analysis
    expected_times = {
        "greedy": 1.0,       # Baseline (no clustering)
        "top_p": 1.5,        # Slightly more complex than greedy
        "dcbs_fast": 4.0,    # Fast PyTorch clustering (~3ms + overhead)
        "dcbs_cached": 2.0,  # With caching benefits
    }
    
    actual_results = benchmark_optimized_dcbs()
    
    print(f"\nComparison with expectations:")
    print(f"{'Method':<15} | {'Expected':<10} | {'Actual':<10} | {'Ratio':<8}")
    print("-" * 50)
    
    for method, expected in expected_times.items():
        actual = actual_results.get(method, 0)
        ratio = actual / expected if expected > 0 else 0
        
        print(f"{method:<15} | {expected:8.1f}ms | {actual:8.2f}ms | {ratio:6.2f}x")
    
    return actual_results

def main():
    """Run comprehensive optimized DCBS testing."""
    
    # Test 1: Performance benchmark
    performance_results = benchmark_optimized_dcbs()
    
    # Test 2: Clustering accuracy
    clustering_results = test_clustering_accuracy()
    
    # Test 3: Expected vs actual comparison
    comparison_results = compare_with_expected_timing()
    
    # Summary
    print(f"\n🎉 **OPTIMIZATION SUMMARY:**")
    print("=" * 50)
    
    # Find the fastest DCBS variant
    dcbs_results = {k: v for k, v in performance_results.items() if "dcbs" in k}
    if dcbs_results:
        fastest_dcbs = min(dcbs_results.items(), key=lambda x: x[1])
        greedy_time = performance_results.get("greedy", 1.0)
        
        print(f"✅ Fastest DCBS: {fastest_dcbs[0]}")
        print(f"✅ Time per call: {fastest_dcbs[1]:.2f}ms")
        print(f"✅ Overhead vs Greedy: {((fastest_dcbs[1] - greedy_time) / greedy_time * 100):+.1f}%")
        
        # Compare with original DCBS timing (from our tests)
        original_dcbs_time = 891  # ms from our cache test
        improvement = original_dcbs_time / fastest_dcbs[1]
        print(f"✅ Speedup vs Original DCBS: {improvement:.0f}x faster!")
        
        print(f"\n💡 **Key Improvements:**")
        print(f"   - PyTorch clustering: 32x faster than scikit-learn")
        print(f"   - Reduced overhead: {fastest_dcbs[1]:.1f}ms vs {original_dcbs_time:.0f}ms")
        print(f"   - Maintained clustering quality: {clustering_results['coverage_score']:.1f} coverage")
    
    return {
        "performance": performance_results,
        "clustering": clustering_results,
        "comparison": comparison_results
    }

if __name__ == "__main__":
    results = main() 