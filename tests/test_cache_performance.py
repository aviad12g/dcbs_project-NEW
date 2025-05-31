"""
Cache performance tests for DCBS sampling.

Tests caching efficiency, thread safety, and performance characteristics
of the embedding and clustering caches.
"""

import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

# Remove sys.path.append line for clean imports

try:
    from src.dcbs import CacheConfig, DCBSSampler, SamplingContext
    from src.dcbs.cache_manager import get_cache_manager, reset_cache_manager
    from src.dcbs.clustering import KMeansClusterer, TopNCandidateSelector
except ImportError as e:
    pytest.skip(f"DCBS modules not available: {e}", allow_module_level=True)


class TestCachePerformance:
    """Test that caching provides measurable performance improvements."""

    def setup_method(self):
        """Setup test fixtures."""
        # Reset cache manager for clean tests
        reset_cache_manager()

        # Create mock embedding layer
        self.vocab_size = 1000
        self.embedding_dim = 512
        self.mock_embedding = Mock()
        self.mock_embedding.weight = torch.randn(self.vocab_size, self.embedding_dim)
        self.mock_embedding.embedding_dim = self.embedding_dim

        def mock_forward(token_ids):
            if isinstance(token_ids, torch.Tensor):
                indices = token_ids.cpu().numpy()
            else:
                indices = np.array(token_ids)
            return self.mock_embedding.weight[indices]

        self.mock_embedding.side_effect = mock_forward

        # Create context
        self.context = SamplingContext(
            embedding_layer=self.mock_embedding, device=torch.device("cpu")
        )

        # Test data
        self.logits = torch.randn(self.vocab_size)
        self.filter_tokens = set(range(50))  # Top 50 tokens

    def test_embedding_cache_performance(self):
        """Test that embedding cache improves performance."""
        # Create samplers with and without caching
        cache_config = CacheConfig(embedding_cache_size=1000, cluster_cache_size=200)
        no_cache_config = CacheConfig(embedding_cache_size=0, cluster_cache_size=0)

        clusterer = KMeansClusterer(k=8)
        candidate_selector = TopNCandidateSelector(top_n=50)

        cached_sampler = DCBSSampler(
            clusterer, candidate_selector, cache_config.__dict__
        )
        uncached_sampler = DCBSSampler(
            clusterer, candidate_selector, no_cache_config.__dict__
        )

        # Warm up (first run to establish baseline)
        cached_sampler.sample(self.logits, self.filter_tokens, self.context)
        uncached_sampler.sample(self.logits, self.filter_tokens, self.context)

        # Measure cached performance (repeated operations)
        num_runs = 20

        start_time = time.time()
        for _ in range(num_runs):
            cached_sampler.sample(self.logits, self.filter_tokens, self.context)
        cached_time = time.time() - start_time

        # Clear cache and measure uncached performance
        reset_cache_manager()

        start_time = time.time()
        for _ in range(num_runs):
            uncached_sampler.sample(self.logits, self.filter_tokens, self.context)
        uncached_time = time.time() - start_time

        # Caching should provide speedup for repeated operations
        speedup = uncached_time / cached_time

        print(f"Cached time: {cached_time:.4f}s")
        print(f"Uncached time: {uncached_time:.4f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Cache should provide at least 1.2x speedup for repeated operations
        assert speedup > 1.2, f"Cache speedup {speedup:.2f}x is less than expected 1.2x"

    def test_cache_hit_rates(self):
        """Test that cache achieves reasonable hit rates."""
        cache_config = CacheConfig(
            embedding_cache_size=1000, cluster_cache_size=200, enable_metrics=True
        )

        clusterer = KMeansClusterer(k=8)
        candidate_selector = TopNCandidateSelector(top_n=50)
        sampler = DCBSSampler(clusterer, candidate_selector, cache_config.__dict__)

        # Run multiple samples with overlapping token sets
        for _ in range(10):
            sampler.sample(self.logits, self.filter_tokens, self.context)

        # Check cache statistics
        stats = sampler.get_cache_stats()

        embedding_metrics = stats["embedding_cache"]["metrics"]
        cluster_metrics = stats["cluster_cache"]["metrics"]

        if embedding_metrics:
            embedding_hit_rate = embedding_metrics.hit_rate
            print(f"Embedding cache hit rate: {embedding_hit_rate:.2%}")
            # Should achieve reasonable hit rate for repeated operations
            assert (
                embedding_hit_rate > 0.5
            ), f"Embedding hit rate {embedding_hit_rate:.2%} too low"

        if cluster_metrics:
            cluster_hit_rate = cluster_metrics.hit_rate
            print(f"Cluster cache hit rate: {cluster_hit_rate:.2%}")
            # Cluster cache should have high hit rate for same parameters
            assert (
                cluster_hit_rate > 0.7
            ), f"Cluster hit rate {cluster_hit_rate:.2%} too low"

    def test_memory_usage_with_cache(self):
        """Test that cache doesn't cause excessive memory usage."""
        cache_config = CacheConfig(
            embedding_cache_size=100,  # Small cache
            cluster_cache_size=50,
            enable_metrics=True,
        )

        clusterer = KMeansClusterer(k=8)
        candidate_selector = TopNCandidateSelector(top_n=50)
        sampler = DCBSSampler(clusterer, candidate_selector, cache_config.__dict__)

        # Fill cache beyond capacity
        for i in range(200):  # More than cache size
            # Use different logits to force cache misses
            varied_logits = self.logits + torch.randn_like(self.logits) * 0.1
            sampler.sample(varied_logits, self.filter_tokens, self.context)

        stats = sampler.get_cache_stats()

        # Cache should not exceed configured size
        embedding_size = stats["embedding_cache"]["size"]
        cluster_size = stats["cluster_cache"]["size"]

        assert embedding_size <= cache_config.embedding_cache_size
        assert cluster_size <= cache_config.cluster_cache_size

        # Should have some evictions due to size limit
        if stats["embedding_cache"]["metrics"]:
            evictions = stats["embedding_cache"]["metrics"].evictions
            assert evictions > 0, "Expected cache evictions due to size limit"

    def test_cache_vs_no_cache_correctness(self):
        """Test that cached and uncached results are identical."""
        cache_config = CacheConfig(embedding_cache_size=1000, cluster_cache_size=200)
        no_cache_config = CacheConfig(embedding_cache_size=0, cluster_cache_size=0)

        clusterer = KMeansClusterer(k=8, random_seed=42)  # Fixed seed for determinism
        candidate_selector = TopNCandidateSelector(top_n=50)

        cached_sampler = DCBSSampler(
            clusterer, candidate_selector, cache_config.__dict__
        )
        uncached_sampler = DCBSSampler(
            clusterer, candidate_selector, no_cache_config.__dict__
        )

        # Test multiple samples
        for i in range(5):
            test_logits = self.logits + i * 0.1  # Slightly different each time

            cached_result = cached_sampler.sample(
                test_logits, self.filter_tokens, self.context
            )
            uncached_result = uncached_sampler.sample(
                test_logits, self.filter_tokens, self.context
            )

            assert cached_result == uncached_result, f"Results differ at iteration {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
