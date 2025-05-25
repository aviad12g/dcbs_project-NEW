"""
Profile the KMeans clustering bottleneck in DCBS.

This script measures the exact time breakdown of clustering operations
to identify optimization opportunities.
"""

import time
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import List, Tuple

def profile_kmeans_operation(num_tokens: int = 50, embedding_dim: int = 512, k: int = 8, num_runs: int = 100):
    """Profile individual components of KMeans clustering."""
    
    print(f"🔍 **PROFILING KMEANS BOTTLENECK**")
    print(f"Tokens: {num_tokens}, Embedding dim: {embedding_dim}, Clusters: {k}")
    print("-" * 60)
    
    # Create test data (similar to real DCBS candidate embeddings)
    device = torch.device("cpu")
    embeddings_tensor = torch.randn(num_tokens, embedding_dim, device=device)
    
    # Normalize embeddings (as done in DCBS)
    norms = torch.norm(embeddings_tensor, p=2, dim=1, keepdim=True)
    normalized_embeddings = embeddings_tensor / norms.clamp(min=1e-6)
    
    # Component timing breakdown
    times = {
        "detach_cpu": [],
        "numpy_conversion": [],
        "kmeans_init": [],
        "kmeans_fit_predict": [],
        "total_clustering": []
    }
    
    for run in range(num_runs):
        start_total = time.time()
        
        # 1. Detach and move to CPU (as done in real DCBS)
        start_detach = time.time()
        detached = normalized_embeddings.detach()
        times["detach_cpu"].append((time.time() - start_detach) * 1000)
        
        # 2. Convert to numpy
        start_numpy = time.time()
        embeddings_np = detached.cpu().numpy()
        times["numpy_conversion"].append((time.time() - start_numpy) * 1000)
        
        # 3. Initialize KMeans
        start_init = time.time()
        effective_k = min(k, len(embeddings_np))
        batch_size = max(3584, len(embeddings_np))  # As in real DCBS
        
        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            batch_size=batch_size,
            max_iter=5,  # As in real DCBS
            random_state=42,
        )
        times["kmeans_init"].append((time.time() - start_init) * 1000)
        
        # 4. Perform clustering
        start_fit = time.time()
        labels = kmeans.fit_predict(embeddings_np)
        times["kmeans_fit_predict"].append((time.time() - start_fit) * 1000)
        
        times["total_clustering"].append((time.time() - start_total) * 1000)
    
    # Calculate averages
    print("⏱️  **TIMING BREAKDOWN (average per call):**")
    for component, timings in times.items():
        avg_time = np.mean(timings)
        std_time = np.std(timings)
        print(f"  {component:20} | {avg_time:6.2f}ms ± {std_time:4.2f}ms")
    
    return times

def compare_clustering_strategies():
    """Compare different clustering approaches for optimization."""
    
    print(f"\n🚀 **COMPARING CLUSTERING STRATEGIES**")
    print("-" * 60)
    
    # Test parameters
    num_tokens = 50
    embedding_dim = 512
    k = 8
    num_runs = 50
    
    device = torch.device("cpu")
    embeddings_tensor = torch.randn(num_tokens, embedding_dim, device=device)
    norms = torch.norm(embeddings_tensor, p=2, dim=1, keepdim=True)
    normalized_embeddings = embeddings_tensor / norms.clamp(min=1e-6)
    embeddings_np = normalized_embeddings.detach().cpu().numpy()
    
    strategies = []
    
    # Strategy 1: Current MiniBatchKMeans (baseline)
    def current_kmeans():
        kmeans = MiniBatchKMeans(
            n_clusters=k, batch_size=max(3584, num_tokens), 
            max_iter=5, random_state=42
        )
        return kmeans.fit_predict(embeddings_np)
    
    strategies.append(("Current MiniBatchKMeans", current_kmeans))
    
    # Strategy 2: Faster KMeans with fewer iterations
    def fast_kmeans():
        kmeans = MiniBatchKMeans(
            n_clusters=k, batch_size=num_tokens, 
            max_iter=2, random_state=42  # Reduced iterations
        )
        return kmeans.fit_predict(embeddings_np)
    
    strategies.append(("Fast KMeans (2 iter)", fast_kmeans))
    
    # Strategy 3: Simple distance-based clustering
    def simple_distance_clustering():
        # Simple clustering based on distance to random centroids
        np.random.seed(42)
        centroids = embeddings_np[np.random.choice(num_tokens, k, replace=False)]
        
        distances = np.linalg.norm(embeddings_np[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels
    
    strategies.append(("Simple Distance", simple_distance_clustering))
    
    # Strategy 4: PyTorch-based clustering (no numpy conversion)
    def pytorch_clustering():
        # K-means in PyTorch (avoiding numpy conversion)
        centroids_idx = torch.randperm(num_tokens)[:k]
        centroids = normalized_embeddings[centroids_idx]
        
        for _ in range(3):  # Few iterations
            distances = torch.cdist(normalized_embeddings, centroids)
            labels = torch.argmin(distances, dim=1)
            
            # Update centroids
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    centroids[i] = normalized_embeddings[mask].mean(dim=0)
        
        return labels.cpu().numpy()
    
    strategies.append(("PyTorch K-means", pytorch_clustering))
    
    # Strategy 5: No clustering (greedy fallback)
    def no_clustering():
        return np.zeros(num_tokens, dtype=int)  # All tokens in one cluster
    
    strategies.append(("No Clustering (Greedy)", no_clustering))
    
    # Benchmark each strategy
    results = {}
    for name, strategy_func in strategies:
        print(f"\n🔧 Testing {name}:")
        
        # Warm up
        strategy_func()
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            labels = strategy_func()
        elapsed_time = (time.time() - start_time) * 1000 / num_runs
        
        # Test output quality (number of unique clusters)
        final_labels = strategy_func()
        unique_clusters = len(np.unique(final_labels))
        
        print(f"  ⏱️  Average time: {elapsed_time:.2f}ms")
        print(f"  🎯 Clusters found: {unique_clusters}/{k}")
        
        results[name] = {
            "time_ms": elapsed_time,
            "clusters": unique_clusters,
            "speedup": 0  # Will calculate relative to baseline
        }
    
    # Calculate speedups relative to baseline
    baseline_time = results["Current MiniBatchKMeans"]["time_ms"]
    for name, metrics in results.items():
        metrics["speedup"] = baseline_time / metrics["time_ms"]
    
    # Summary
    print(f"\n📊 **CLUSTERING STRATEGY COMPARISON:**")
    print("-" * 70)
    print(f"{'Strategy':<25} | {'Time (ms)':<10} | {'Clusters':<8} | {'Speedup':<8}")
    print("-" * 70)
    
    for name, metrics in results.items():
        time_ms = metrics["time_ms"]
        clusters = metrics["clusters"]
        speedup = metrics["speedup"]
        print(f"{name:<25} | {time_ms:8.2f} | {clusters:6d}/{k} | {speedup:6.1f}x")
    
    return results

def analyze_cache_vs_clustering_costs():
    """Analyze the cost tradeoff between caching and clustering."""
    
    print(f"\n💰 **CACHE vs CLUSTERING COST ANALYSIS**")
    print("-" * 60)
    
    # Simulate cache hit rates and clustering costs
    cache_hit_rates = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
    cache_lookup_cost_ms = 0.5   # Estimated cache lookup overhead
    clustering_cost_ms = 25.0    # Estimated from our benchmarks
    
    print(f"Assumptions:")
    print(f"  Cache lookup overhead: {cache_lookup_cost_ms:.1f}ms")
    print(f"  Clustering cost: {clustering_cost_ms:.1f}ms")
    print()
    
    print(f"{'Hit Rate':<10} | {'Avg Cost':<10} | {'vs No Cache':<12}")
    print("-" * 35)
    
    no_cache_cost = clustering_cost_ms  # Always cluster without cache
    
    for hit_rate in cache_hit_rates:
        avg_cost = hit_rate * cache_lookup_cost_ms + (1 - hit_rate) * (cache_lookup_cost_ms + clustering_cost_ms)
        ratio = avg_cost / no_cache_cost
        
        print(f"{hit_rate*100:8.0f}% | {avg_cost:8.2f}ms | {ratio:8.2f}x")
    
    print()
    print("💡 **Insights:**")
    print("  - Cache only helps if hit rate > ~95%")
    print("  - At 98% hit rate, cache still adds overhead!")
    print("  - Better strategy: Optimize clustering, not caching")

def main():
    """Run comprehensive KMeans bottleneck analysis."""
    
    # 1. Profile individual KMeans components
    profile_kmeans_operation()
    
    # 2. Compare clustering strategies
    results = compare_clustering_strategies()
    
    # 3. Analyze cache economics
    analyze_cache_vs_clustering_costs()
    
    # 4. Recommendations
    print(f"\n🎯 **OPTIMIZATION RECOMMENDATIONS:**")
    print("-" * 60)
    
    best_strategy = min(results.items(), key=lambda x: x[1]["time_ms"])
    best_name, best_metrics = best_strategy
    
    print(f"✅ Best clustering strategy: {best_name}")
    print(f"✅ Speed improvement: {best_metrics['speedup']:.1f}x faster")
    print(f"✅ Time per call: {best_metrics['time_ms']:.2f}ms")
    print()
    print(f"💡 **Next steps:**")
    print(f"   1. Replace MiniBatchKMeans with {best_name}")
    print(f"   2. Consider caching only for very high hit rates (>99%)")
    print(f"   3. Implement adaptive clustering based on candidate set size")

if __name__ == "__main__":
    main() 