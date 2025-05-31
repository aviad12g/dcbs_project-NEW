"""
Thread-safe cache management for DCBS sampling.

This module provides thread-safe caching mechanisms for token embeddings and
clustering results to support concurrent evaluation scenarios.
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    embedding_cache_size: int = 1000
    cluster_cache_size: int = 200
    cache_ttl_seconds: Optional[float] = None  # Time-to-live for cache entries
    enable_metrics: bool = False


@dataclass
class CacheMetrics:
    """Cache performance metrics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class ThreadSafeCache:
    """Thread-safe LRU cache with optional TTL support."""

    def __init__(
        self,
        max_size: int,
        ttl_seconds: Optional[float] = None,
        enable_metrics: bool = False,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.enable_metrics = enable_metrics

        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[Any, float] = {} if ttl_seconds else {}
        self._lock = threading.RLock()  # Reentrant lock for nested operations
        self._metrics = CacheMetrics() if enable_metrics else None

    def get(self, key: Any) -> Optional[Any]:
        """Thread-safe cache retrieval with TTL support."""
        with self._lock:
            # Check if key exists
            if key not in self._cache:
                if self._metrics:
                    self._metrics.misses += 1
                return None

            # Check TTL if enabled
            if self.ttl_seconds and key in self._timestamps:
                if time.time() - self._timestamps[key] > self.ttl_seconds:
                    # Entry expired, remove it
                    del self._cache[key]
                    del self._timestamps[key]
                    if self._metrics:
                        self._metrics.misses += 1
                        self._metrics.evictions += 1
                    return None

            # Move to end (mark as recently used)
            value = self._cache[key]
            self._cache.move_to_end(key)

            if self._metrics:
                self._metrics.hits += 1

            return value

    def put(self, key: Any, value: Any) -> None:
        """Thread-safe cache insertion with size management."""
        with self._lock:
            current_time = time.time()

            # Update existing key
            if key in self._cache:
                self._cache[key] = value
                self._cache.move_to_end(key)
                if self.ttl_seconds:
                    self._timestamps[key] = current_time
                return

            # Add new key
            self._cache[key] = value
            if self.ttl_seconds:
                self._timestamps[key] = current_time

            # Evict if necessary
            while len(self._cache) > self.max_size:
                evicted_key = next(iter(self._cache))
                del self._cache[evicted_key]
                if evicted_key in self._timestamps:
                    del self._timestamps[evicted_key]
                if self._metrics:
                    self._metrics.evictions += 1

    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            if self._metrics:
                self._metrics.hits = 0
                self._metrics.misses = 0
                self._metrics.evictions = 0

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)

    def get_metrics(self) -> Optional[CacheMetrics]:
        """Get cache performance metrics."""
        return self._metrics


class DCBSCacheManager:
    """Centralized cache manager for DCBS operations."""

    # Singleton instance for backward compatibility
    _instance: Optional['DCBSCacheManager'] = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls, config: Optional[CacheConfig] = None) -> 'DCBSCacheManager':
        """Get the singleton cache manager instance (for backward compatibility).
        
        This method is provided for backward compatibility. New code should
        use dependency injection by creating and passing DCBSCacheManager instances.
        
        Args:
            config: Optional cache configuration
            
        Returns:
            Singleton cache manager instance
        """
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cache_config = config or CacheConfig()
                    cls._instance = cls(cache_config)
        
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton cache manager instance (for testing)."""
        with cls._instance_lock:
            if cls._instance:
                cls._instance.clear_all_caches()
            cls._instance = None

    def __init__(self, config: CacheConfig):
        self.config = config

        # Separate caches for different data types
        self.embedding_cache = ThreadSafeCache(
            max_size=config.embedding_cache_size,
            ttl_seconds=config.cache_ttl_seconds,
            enable_metrics=config.enable_metrics,
        )

        self.cluster_cache = ThreadSafeCache(
            max_size=config.cluster_cache_size,
            ttl_seconds=config.cache_ttl_seconds,
            enable_metrics=config.enable_metrics,
        )

        # Global lock for cross-cache operations
        self._global_lock = threading.Lock()

    def get_embedding(self, token_id: int) -> Optional[torch.Tensor]:
        """Retrieve cached token embedding."""
        return self.embedding_cache.get(token_id)

    def put_embedding(self, token_id: int, embedding: torch.Tensor) -> None:
        """Cache token embedding (detached from computation graph)."""
        # Always detach to prevent memory leaks
        detached_embedding = embedding.detach().clone()
        self.embedding_cache.put(token_id, detached_embedding)

    def get_clustering(self, cache_key: Tuple) -> Optional[np.ndarray]:
        """Retrieve cached clustering results."""
        return self.cluster_cache.get(cache_key)

    def put_clustering(self, cache_key: Tuple, labels: np.ndarray) -> None:
        """Cache clustering results."""
        # Make a copy to ensure thread safety
        labels_copy = labels.copy()
        self.cluster_cache.put(cache_key, labels_copy)

    def get_batch_embeddings(
        self, token_ids: list, embedding_layer: torch.nn.Embedding, device: torch.device
    ) -> Tuple[torch.Tensor, list]:
        """
        Thread-safe batch embedding retrieval with cache optimization.

        Returns:
            (embeddings_tensor, uncached_indices)
        """
        with self._global_lock:
            batch_size = len(token_ids)
            embedding_dim = embedding_layer.embedding_dim

            # Pre-allocate result tensor
            result = torch.zeros((batch_size, embedding_dim), device=device)
            uncached_indices = []

            # Check cache for each token
            for i, token_id in enumerate(token_ids):
                cached_embedding = self.get_embedding(token_id)
                if cached_embedding is not None:
                    # Move to correct device if necessary
                    if cached_embedding.device != device:
                        cached_embedding = cached_embedding.to(device)
                    result[i] = cached_embedding
                else:
                    uncached_indices.append(i)

            return result, uncached_indices

    def cache_batch_embeddings(
        self, token_ids: list, uncached_indices: list, embeddings: torch.Tensor
    ) -> None:
        """Cache a batch of embeddings efficiently."""
        for i, batch_idx in enumerate(uncached_indices):
            token_id = token_ids[batch_idx]
            embedding = embeddings[i]
            self.put_embedding(token_id, embedding)

    def clear_all_caches(self) -> None:
        """Clear all caches."""
        with self._global_lock:
            self.embedding_cache.clear()
            self.cluster_cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            "embedding_cache": {
                "size": self.embedding_cache.size(),
                "max_size": self.config.embedding_cache_size,
                "metrics": self.embedding_cache.get_metrics(),
            },
            "cluster_cache": {
                "size": self.cluster_cache.size(),
                "max_size": self.config.cluster_cache_size,
                "metrics": self.cluster_cache.get_metrics(),
            },
        }

        # Calculate overall hit rates
        if self.config.enable_metrics:
            total_hits = 0
            total_requests = 0

            for cache_stats in stats.values():
                if cache_stats["metrics"]:
                    total_hits += cache_stats["metrics"].hits
                    total_requests += (
                        cache_stats["metrics"].hits + cache_stats["metrics"].misses
                    )

            stats["overall"] = {
                "hit_rate": total_hits / total_requests if total_requests > 0 else 0.0,
                "total_requests": total_requests,
            }

        return stats


# Backward compatibility functions using the singleton pattern
def get_cache_manager(config: Optional[CacheConfig] = None) -> DCBSCacheManager:
    """Get the cache manager singleton instance (for backward compatibility).
    
    New code should use dependency injection by creating and passing DCBSCacheManager instances.
    
    Args:
        config: Optional cache configuration
        
    Returns:
        Cache manager instance
    """
    return DCBSCacheManager.get_instance(config)


def reset_cache_manager() -> None:
    """Reset the cache manager singleton instance (for testing)."""
    DCBSCacheManager.reset_instance()
