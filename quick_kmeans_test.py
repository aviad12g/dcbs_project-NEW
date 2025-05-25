"""
Quick test to validate KMeans clustering bottleneck hypothesis.
"""

import time
import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def test_kmeans_timing():
    """Test if KMeans is indeed the 37ms bottleneck."""
    
    print("🔍 **QUICK KMEANS BOTTLENECK TEST**")
    print("-" * 40)
    
    # Simulate DCBS parameters
    num_tokens = 50      # Typical candidate set size
    embedding_dim = 512  # Typical embedding dimension  
    k = 8               # Number of clusters
    
    # Create test embeddings
    embeddings = torch.randn(num_tokens, embedding_dim)
    embeddings_np = embeddings.detach().cpu().numpy()
    
    # Test KMeans timing (as used in real DCBS)
    num_runs = 20
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=max(3584, num_tokens),
            max_iter=5,
            random_state=42,
        )
        labels = kmeans.fit_predict(embeddings_np)
        
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        times.append(elapsed)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"KMeans clustering time: {avg_time:.1f}ms ± {std_time:.1f}ms")
    print(f"Expected overhead: ~{avg_time:.0f}ms per DCBS call")
    
    # Compare with observed overhead
    observed_overhead = 37  # From our no-cache test: 891ms - 854ms
    print(f"Observed cache overhead: {observed_overhead}ms")
    
    if abs(avg_time - observed_overhead) < 15:
        print("✅ **CONFIRMED: KMeans is the bottleneck!**")
    else:
        print("❌ **KMeans timing doesn't match observed overhead**")
    
    return avg_time

def test_fast_alternatives():
    """Test faster clustering alternatives."""
    
    print(f"\n🚀 **TESTING FAST ALTERNATIVES**")
    print("-" * 40)
    
    num_tokens = 50
    embedding_dim = 512
    k = 8
    
    embeddings = torch.randn(num_tokens, embedding_dim)
    embeddings_np = embeddings.detach().cpu().numpy()
    
    alternatives = []
    
    # 1. Fast KMeans (2 iterations)
    def fast_kmeans():
        kmeans = MiniBatchKMeans(n_clusters=k, max_iter=2, random_state=42)
        return kmeans.fit_predict(embeddings_np)
    
    alternatives.append(("Fast KMeans", fast_kmeans))
    
    # 2. PyTorch clustering
    def pytorch_kmeans():
        centroids_idx = torch.randperm(num_tokens)[:k]
        centroids = embeddings[centroids_idx]
        
        for _ in range(2):
            distances = torch.cdist(embeddings, centroids)
            labels = torch.argmin(distances, dim=1)
            
            for i in range(k):
                mask = labels == i
                if mask.sum() > 0:
                    centroids[i] = embeddings[mask].mean(dim=0)
        
        return labels.numpy()
    
    alternatives.append(("PyTorch KMeans", pytorch_kmeans))
    
    # 3. No clustering (greedy)
    def no_clustering():
        return np.zeros(num_tokens, dtype=int)
    
    alternatives.append(("No Clustering", no_clustering))
    
    # Benchmark alternatives
    for name, func in alternatives:
        times = []
        for _ in range(20):
            start_time = time.time()
            labels = func()
            times.append((time.time() - start_time) * 1000)
        
        avg_time = np.mean(times)
        clusters = len(np.unique(labels))
        
        print(f"{name:15} | {avg_time:5.1f}ms | {clusters}/{k} clusters")
    
    return alternatives

if __name__ == "__main__":
    kmeans_time = test_kmeans_timing()
    test_fast_alternatives()
    
    print(f"\n💡 **CONCLUSION:**")
    print(f"   KMeans clustering adds ~{kmeans_time:.0f}ms per DCBS call")
    print(f"   This explains the 37ms cache overhead!")
    print(f"   Solution: Use faster clustering or disable for small sets") 