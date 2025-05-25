"""
Cache optimization tests - comparing different strategies to fix the 37ms overhead.

This script tests various cache optimization approaches:
1. Baseline (current implementation)
2. Batch-optimized cache
3. Lock-free cache
4. Device-aware cache
5. Hybrid approaches
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import threading
from dataclasses import dataclass

# Mock the DCBS components for testing
@dataclass
class MockCacheConfig:
    embedding_cache_size: int = 1000
    cluster_cache_size: int = 200
    enable_metrics: bool = True

class BaselineCacheManager:
    """Current cache implementation (with bottlenecks)."""
    
    def __init__(self, config: MockCacheConfig):
        self.config = config
        self._cache = OrderedDict()
        self._global_lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def get_batch_embeddings(self, token_ids: list, embedding_layer, device: torch.device):
        """Current implementation - has bottlenecks."""
        with self._global_lock:  # BOTTLENECK 1: Global lock
            batch_size = len(token_ids)
            embedding_dim = embedding_layer.embedding_dim
            
            result = torch.zeros((batch_size, embedding_dim), device=device)
            uncached_indices = []
            
            # BOTTLENECK 2: Per-token lookup
            for i, token_id in enumerate(token_ids):
                cached_embedding = self._cache.get(token_id)
                if cached_embedding is not None:
                    self.hits += 1
                    # BOTTLENECK 3: Device transfer per hit
                    if cached_embedding.device != device:
                        cached_embedding = cached_embedding.to(device)
                    result[i] = cached_embedding
                else:
                    self.misses += 1
                    uncached_indices.append(i)
            
            return result, uncached_indices

class OptimizedCacheManager:
    """Optimized cache implementation."""
    
    def __init__(self, config: MockCacheConfig):
        self.config = config
        self._cache = OrderedDict()
        self._lock = threading.RLock()  # Lighter lock
        self.hits = 0
        self.misses = 0
    
    def get_batch_embeddings(self, token_ids: list, embedding_layer, device: torch.device):
        """Optimized implementation."""
        batch_size = len(token_ids)
        embedding_dim = embedding_layer.embedding_dim
        
        # OPTIMIZATION 1: Batch cache lookup without full lock
        cached_embeddings = []
        uncached_indices = []
        
        with self._lock:
            for i, token_id in enumerate(token_ids):
                cached_embedding = self._cache.get(token_id)
                if cached_embedding is not None:
                    self.hits += 1
                    cached_embeddings.append((i, cached_embedding))
                else:
                    self.misses += 1
                    uncached_indices.append(i)
        
        # OPTIMIZATION 2: Batch device transfer
        result = torch.zeros((batch_size, embedding_dim), device=device)
        if cached_embeddings:
            indices, embeddings = zip(*cached_embeddings)
            embeddings_tensor = torch.stack(embeddings)
            if embeddings_tensor.device != device:
                embeddings_tensor = embeddings_tensor.to(device)
            
            for idx, embedding in zip(indices, embeddings_tensor):
                result[idx] = embedding
        
        return result, uncached_indices

class LockFreeCacheManager:
    """Lock-free cache using thread-local storage."""
    
    def __init__(self, config: MockCacheConfig):
        self.config = config
        self._cache = OrderedDict()
        self.local = threading.local()
        self.hits = 0
        self.misses = 0
    
    def get_batch_embeddings(self, token_ids: list, embedding_layer, device: torch.device):
        """Lock-free implementation."""
        batch_size = len(token_ids)
        embedding_dim = embedding_layer.embedding_dim
        
        result = torch.zeros((batch_size, embedding_dim), device=device)
        uncached_indices = []
        
        # OPTIMIZATION: No locks for read-only cache access
        for i, token_id in enumerate(token_ids):
            cached_embedding = self._cache.get(token_id)
            if cached_embedding is not None:
                self.hits += 1
                if cached_embedding.device != device:
                    cached_embedding = cached_embedding.to(device)
                result[i] = cached_embedding
            else:
                self.misses += 1
                uncached_indices.append(i)
        
        return result, uncached_indices

class DeviceAwareCacheManager:
    """Device-aware cache that pre-stores embeddings on target device."""
    
    def __init__(self, config: MockCacheConfig, device: torch.device):
        self.config = config
        self.device = device
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def put_embedding(self, token_id: int, embedding: torch.Tensor):
        """Store embedding on target device."""
        with self._lock:
            # OPTIMIZATION: Store on target device to avoid transfers
            device_embedding = embedding.detach().to(self.device)
            self._cache[token_id] = device_embedding
    
    def get_batch_embeddings(self, token_ids: list, embedding_layer, device: torch.device):
        """Device-aware implementation."""
        batch_size = len(token_ids)
        embedding_dim = embedding_layer.embedding_dim
        
        result = torch.zeros((batch_size, embedding_dim), device=device)
        uncached_indices = []
        
        with self._lock:
            for i, token_id in enumerate(token_ids):
                cached_embedding = self._cache.get(token_id)
                if cached_embedding is not None:
                    self.hits += 1
                    # No device transfer needed!
                    result[i] = cached_embedding
                else:
                    self.misses += 1
                    uncached_indices.append(i)
        
        return result, uncached_indices

def create_mock_embedding_layer(vocab_size: int, embedding_dim: int, device: torch.device):
    """Create mock embedding layer for testing."""
    class MockEmbedding:
        def __init__(self):
            self.embedding_dim = embedding_dim
            self.weight = torch.randn(vocab_size, embedding_dim, device=device)
        
        def __call__(self, token_ids):
            return self.weight[token_ids]
    
    return MockEmbedding()

def benchmark_cache_strategy(cache_manager, name: str, num_runs: int = 100):
    """Benchmark a cache strategy."""
    print(f"\n🔧 **Testing {name}:**")
    
    device = torch.device("cpu")
    embedding_layer = create_mock_embedding_layer(10000, 512, device)
    
    # Pre-populate cache with some embeddings
    if hasattr(cache_manager, 'put_embedding'):
        for i in range(500):
            embedding = torch.randn(512, device=device)
            cache_manager.put_embedding(i, embedding)
    elif hasattr(cache_manager, '_cache'):
        for i in range(500):
            embedding = torch.randn(512, device=device)
            cache_manager._cache[i] = embedding
    
    # Test data: 50 tokens (typical candidate set size)
    token_ids = list(range(50))  # All should be cache hits
    
    # Warm up
    cache_manager.get_batch_embeddings(token_ids, embedding_layer, device)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        result, uncached = cache_manager.get_batch_embeddings(token_ids, embedding_layer, device)
    
    elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
    avg_time = elapsed_time / num_runs
    
    hit_rate = cache_manager.hits / (cache_manager.hits + cache_manager.misses) * 100
    
    print(f"  ⏱️  Average time: {avg_time:.2f}ms")
    print(f"  🎯 Hit rate: {hit_rate:.1f}%")
    print(f"  📊 Total hits: {cache_manager.hits}, misses: {cache_manager.misses}")
    
    return avg_time, hit_rate

def main():
    """Run cache optimization benchmarks."""
    print("🚀 **CACHE OPTIMIZATION BENCHMARK**")
    print("=" * 50)
    
    config = MockCacheConfig()
    device = torch.device("cpu")
    
    # Test all strategies
    strategies = [
        (BaselineCacheManager(config), "Baseline (Current)"),
        (OptimizedCacheManager(config), "Batch Optimized"),
        (LockFreeCacheManager(config), "Lock-Free"),
        (DeviceAwareCacheManager(config, device), "Device-Aware"),
    ]
    
    results = {}
    for cache_manager, name in strategies:
        avg_time, hit_rate = benchmark_cache_strategy(cache_manager, name)
        results[name] = {"time": avg_time, "hit_rate": hit_rate}
    
    # Summary
    print(f"\n📊 **OPTIMIZATION RESULTS SUMMARY:**")
    print("-" * 60)
    
    baseline_time = results["Baseline (Current)"]["time"]
    
    for name, metrics in results.items():
        time_ms = metrics["time"]
        hit_rate = metrics["hit_rate"]
        speedup = baseline_time / time_ms if time_ms > 0 else 0
        
        print(f"{name:20} | {time_ms:6.2f}ms | {hit_rate:5.1f}% | {speedup:4.1f}x speedup")
    
    # Recommendations
    print(f"\n🎯 **OPTIMIZATION RECOMMENDATIONS:**")
    best_strategy = min(results.items(), key=lambda x: x[1]["time"])
    improvement = baseline_time / best_strategy[1]["time"]
    
    print(f"✅ Best strategy: {best_strategy[0]}")
    print(f"✅ Performance improvement: {improvement:.1f}x faster")
    print(f"✅ Time saved per call: {baseline_time - best_strategy[1]['time']:.2f}ms")

if __name__ == "__main__":
    main() 