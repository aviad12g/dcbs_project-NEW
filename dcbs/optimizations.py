"""
Performance optimizations for DCBS in high-throughput scenarios.

This module provides optimized implementations for batch processing,
GPU acceleration, and memory-efficient operations.
"""

import concurrent.futures
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from .cache_manager import CacheConfig, DCBSCacheManager
from .sampler import SamplingContext


@dataclass
class OptimizationConfig:
    """Configuration for performance optimizations."""

    batch_size: int = 32
    use_gpu_clustering: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    precompute_embeddings: bool = True
    use_mixed_precision: bool = True
    memory_efficient_mode: bool = False
    prefetch_factor: int = 2


class BatchDCBSProcessor:
    """High-performance batch processor for DCBS operations."""

    def __init__(self, config: OptimizationConfig, cache_manager: DCBSCacheManager):
        self.config = config
        self.cache_manager = cache_manager
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Thread pool for parallel processing
        if config.enable_parallel_processing:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=config.max_workers
            )
        else:
            self.executor = None

    def batch_sample(
        self,
        logits_batch: torch.Tensor,
        filter_tokens_batch: List[Optional[set]],
        context: SamplingContext,
        k: int = 8,
        top_n: int = 50,
    ) -> List[int]:
        """
        Process a batch of logits simultaneously for improved throughput.

        Args:
            logits_batch: Tensor of shape (batch_size, vocab_size)
            filter_tokens_batch: List of filter token sets for each example
            context: Sampling context with embedding layer
            k: Number of clusters
            top_n: Number of top tokens to consider

        Returns:
            List of selected token IDs
        """
        batch_size = logits_batch.shape[0]

        if batch_size == 1:
            # Single example, use standard processing
            return [
                self._single_sample(
                    logits_batch[0], filter_tokens_batch[0], context, k, top_n
                )
            ]

        # Batch processing optimizations
        if self.config.enable_parallel_processing and batch_size > 4:
            return self._parallel_batch_sample(
                logits_batch, filter_tokens_batch, context, k, top_n
            )
        else:
            return self._sequential_batch_sample(
                logits_batch, filter_tokens_batch, context, k, top_n
            )

    def _parallel_batch_sample(
        self,
        logits_batch: torch.Tensor,
        filter_tokens_batch: List[Optional[set]],
        context: SamplingContext,
        k: int,
        top_n: int,
    ) -> List[int]:
        """Process batch using parallel workers."""
        # Split batch into chunks for parallel processing
        chunk_size = max(1, len(logits_batch) // self.config.max_workers)
        chunks = []

        for i in range(0, len(logits_batch), chunk_size):
            end_idx = min(i + chunk_size, len(logits_batch))
            chunks.append(
                (
                    logits_batch[i:end_idx],
                    filter_tokens_batch[i:end_idx],
                    context,
                    k,
                    top_n,
                )
            )

        # Process chunks in parallel
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._process_chunk, *chunk)
            futures.append(future)

        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            chunk_results = future.result()
            results.extend(chunk_results)

        return results

    def _process_chunk(
        self,
        logits_chunk: torch.Tensor,
        filter_tokens_chunk: List[Optional[set]],
        context: SamplingContext,
        k: int,
        top_n: int,
    ) -> List[int]:
        """Process a chunk of the batch."""
        results = []
        for i in range(len(logits_chunk)):
            result = self._single_sample(
                logits_chunk[i], filter_tokens_chunk[i], context, k, top_n
            )
            results.append(result)
        return results

    def _sequential_batch_sample(
        self,
        logits_batch: torch.Tensor,
        filter_tokens_batch: List[Optional[set]],
        context: SamplingContext,
        k: int,
        top_n: int,
    ) -> List[int]:
        """Process batch sequentially with optimizations."""
        results = []

        # Pre-extract all candidate tokens and embeddings
        all_candidates = []
        all_embeddings = []

        with torch.no_grad():
            for i, (logits, filter_tokens) in enumerate(
                zip(logits_batch, filter_tokens_batch)
            ):
                candidates = self._get_candidates(logits, filter_tokens, top_n)
                all_candidates.append(candidates)

                if len(candidates) > 3:  # Only compute if clustering will be used
                    embeddings = self._batch_get_embeddings(
                        candidates, context.embedding_layer
                    )
                    all_embeddings.append(embeddings)
                else:
                    all_embeddings.append(None)

        # Process each example with pre-computed data
        for i, (logits, candidates, embeddings) in enumerate(
            zip(logits_batch, all_candidates, all_embeddings)
        ):
            if embeddings is not None:
                result = self._optimized_dcbs_selection(
                    logits, candidates, embeddings, filter_tokens_batch[i], k
                )
            else:
                # Simple selection for small candidate sets
                result = self._simple_selection(logits, candidates)

            results.append(result)

        return results

    def _single_sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[set],
        context: SamplingContext,
        k: int,
        top_n: int,
    ) -> int:
        """Standard single-example processing."""
        candidates = self._get_candidates(logits, filter_tokens, top_n)

        if len(candidates) <= 3:
            return self._simple_selection(logits, candidates)

        embeddings = self._batch_get_embeddings(candidates, context.embedding_layer)
        return self._optimized_dcbs_selection(
            logits, candidates, embeddings, filter_tokens, k
        )

    def _get_candidates(
        self, logits: torch.Tensor, filter_tokens: Optional[set], top_n: int
    ) -> List[int]:
        """Get candidate tokens efficiently."""
        if filter_tokens:
            return list(filter_tokens)
        else:
            # Use torch.topk for efficiency
            _, indices = torch.topk(logits, min(top_n, len(logits)))
            return indices.cpu().tolist()

    def _batch_get_embeddings(
        self, candidates: List[int], embedding_layer: torch.nn.Embedding
    ) -> torch.Tensor:
        """Efficiently retrieve embeddings for candidates."""
        # Convert to tensor for batch processing
        candidate_tensor = torch.tensor(candidates, device=self.device)

        # Check cache first
        cached_embeddings = []
        uncached_indices = []

        for i, token_id in enumerate(candidates):
            cached = self.cache_manager.get_embedding(token_id)
            if cached is not None:
                if cached.device != self.device:
                    cached = cached.to(self.device)
                cached_embeddings.append((i, cached))
            else:
                uncached_indices.append(i)

        # Get embeddings for uncached tokens
        if uncached_indices:
            uncached_tokens = candidate_tensor[uncached_indices]
            with torch.no_grad():
                if self.config.use_mixed_precision:
                    with torch.amp.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu"
                    ):
                        uncached_embeddings = embedding_layer(uncached_tokens)
                else:
                    uncached_embeddings = embedding_layer(uncached_tokens)

            # Cache the new embeddings
            for i, embedding in enumerate(uncached_embeddings):
                token_id = candidates[uncached_indices[i]]
                self.cache_manager.put_embedding(token_id, embedding)

        # Reconstruct full embedding tensor
        result = torch.zeros(
            (len(candidates), embedding_layer.embedding_dim),
            device=self.device,
            dtype=torch.float32,
        )

        # Fill cached embeddings
        for idx, embedding in cached_embeddings:
            result[idx] = embedding

        # Fill uncached embeddings
        if uncached_indices:
            for i, idx in enumerate(uncached_indices):
                result[idx] = uncached_embeddings[i]

        return result

    def _optimized_dcbs_selection(
        self,
        logits: torch.Tensor,
        candidates: List[int],
        embeddings: torch.Tensor,
        filter_tokens: Optional[set],
        k: int,
    ) -> int:
        """Optimized DCBS selection with GPU acceleration."""
        # Compute probabilities
        candidate_logits = logits[candidates]
        candidate_probs = torch.softmax(candidate_logits, dim=0)

        # Normalize embeddings
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = embeddings / norms.clamp(min=1e-6)

        # Clustering
        effective_k = min(k, len(candidates))

        if self.config.use_gpu_clustering and self.device.type == "cuda":
            labels = self._gpu_clustering(normalized_embeddings, effective_k)
        else:
            labels = self._cpu_clustering(normalized_embeddings, effective_k)

        # Group tokens by cluster
        clusters = [[] for _ in range(effective_k)]
        for i, label in enumerate(labels):
            clusters[label].append(i)

        # Calculate cluster probabilities
        cluster_probs = []
        for cluster in clusters:
            if cluster:
                cluster_prob = candidate_probs[cluster].sum().item()
                cluster_probs.append(cluster_prob)
            else:
                cluster_probs.append(0.0)

        if sum(cluster_probs) == 0:
            return candidates[torch.argmax(candidate_probs).item()]

        # Select best cluster and token
        best_cluster_idx = np.argmax(cluster_probs)
        cluster_indices = clusters[best_cluster_idx]

        # Apply filtering within cluster
        if filter_tokens:
            valid_indices = [
                i for i in cluster_indices if candidates[i] in filter_tokens
            ]
            if not valid_indices:
                # Fallback to global best
                return candidates[torch.argmax(candidate_probs).item()]
            cluster_indices = valid_indices

        # Select best token from cluster
        cluster_probs_tensor = candidate_probs[cluster_indices]
        best_in_cluster = torch.argmax(cluster_probs_tensor).item()
        return candidates[cluster_indices[best_in_cluster]]

    def _gpu_clustering(self, embeddings: torch.Tensor, k: int) -> np.ndarray:
        """GPU-accelerated clustering using PyTorch operations."""
        try:
            # Simple k-means implementation using torch operations
            n_samples, n_features = embeddings.shape

            # Initialize centroids randomly
            torch.manual_seed(42)  # For reproducibility
            centroids = embeddings[torch.randperm(n_samples)[:k]]

            # K-means iterations
            for _ in range(10):  # Fixed iterations for speed
                # Compute distances
                distances = torch.cdist(embeddings, centroids)

                # Assign to closest centroid
                assignments = torch.argmin(distances, dim=1)

                # Update centroids
                new_centroids = torch.zeros_like(centroids)
                for i in range(k):
                    mask = assignments == i
                    if mask.sum() > 0:
                        new_centroids[i] = embeddings[mask].mean(dim=0)
                    else:
                        new_centroids[i] = centroids[i]  # Keep old centroid

                centroids = new_centroids

            return assignments.cpu().numpy()

        except Exception:
            # Fallback to CPU clustering
            return self._cpu_clustering(embeddings, k)

    def _cpu_clustering(self, embeddings: torch.Tensor, k: int) -> np.ndarray:
        """CPU clustering using scikit-learn."""
        embeddings_np = embeddings.detach().cpu().numpy()

        # Use MiniBatchKMeans for efficiency
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=max(32, len(embeddings_np)),
            max_iter=10,
            random_state=42,
            n_init=1,  # Single initialization for speed
        )

        labels = kmeans.fit_predict(embeddings_np)
        return labels

    def _simple_selection(self, logits: torch.Tensor, candidates: List[int]) -> int:
        """Simple greedy selection for small candidate sets."""
        candidate_logits = logits[candidates]
        best_idx = torch.argmax(candidate_logits).item()
        return candidates[best_idx]

    def precompute_embeddings(
        self, token_ids: List[int], embedding_layer: torch.nn.Embedding
    ) -> None:
        """Precompute and cache embeddings for frequently used tokens."""
        # Process in batches to avoid memory issues
        batch_size = 1000

        for i in range(0, len(token_ids), batch_size):
            batch_tokens = token_ids[i : i + batch_size]

            # Check which tokens are not already cached
            uncached_tokens = [
                token_id
                for token_id in batch_tokens
                if self.cache_manager.get_embedding(token_id) is None
            ]

            if uncached_tokens:
                # Batch compute embeddings
                token_tensor = torch.tensor(uncached_tokens, device=self.device)

                with torch.no_grad():
                    if self.config.use_mixed_precision:
                        with torch.amp.autocast(
                            device_type="cuda" if torch.cuda.is_available() else "cpu"
                        ):
                            embeddings = embedding_layer(token_tensor)
                    else:
                        embeddings = embedding_layer(token_tensor)

                # Cache embeddings
                for token_id, embedding in zip(uncached_tokens, embeddings):
                    self.cache_manager.put_embedding(token_id, embedding)

    def get_optimization_stats(self) -> Dict[str, any]:
        """Get performance optimization statistics."""
        cache_stats = self.cache_manager.get_cache_stats()

        return {
            "cache_stats": cache_stats,
            "device": str(self.device),
            "config": {
                "batch_size": self.config.batch_size,
                "use_gpu_clustering": self.config.use_gpu_clustering,
                "parallel_processing": self.config.enable_parallel_processing,
                "max_workers": self.config.max_workers,
                "mixed_precision": self.config.use_mixed_precision,
            },
        }

    def cleanup(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)


class MemoryEfficientDCBS:
    """Memory-efficient DCBS implementation for large-scale processing."""

    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_with_memory_limit(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[set],
        context: SamplingContext,
        k: int = 8,
        top_n: int = 50,
    ) -> int:
        """Sample with memory usage constraints."""
        # Monitor memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

        # Adaptive top_n based on available memory
        adaptive_top_n = self._compute_adaptive_top_n(logits, top_n)

        # Get candidates
        if filter_tokens:
            candidates = list(filter_tokens)
        else:
            candidates = self._get_top_n_memory_efficient(logits, adaptive_top_n)

        if len(candidates) <= 3:
            candidate_logits = logits[candidates]
            best_idx = torch.argmax(candidate_logits).item()
            return candidates[best_idx]

        # Memory-efficient embedding retrieval
        embeddings = self._get_embeddings_memory_efficient(
            candidates, context.embedding_layer
        )

        # Clustering with memory constraints
        labels = self._cluster_memory_efficient(embeddings, k)

        # Standard DCBS selection
        return self._select_token_from_clusters(
            logits, candidates, labels, filter_tokens
        )

    def _compute_adaptive_top_n(self, logits: torch.Tensor, target_top_n: int) -> int:
        """Compute adaptive top_n based on available memory."""
        vocab_size = logits.shape[0]

        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory
            used_memory = torch.cuda.memory_allocated()
            free_memory = available_memory - used_memory

            # Estimate memory needed for embeddings (assuming 512-dim embeddings)
            memory_per_token = 512 * 4  # 4 bytes per float32
            max_tokens_by_memory = free_memory // (
                memory_per_token * 4
            )  # Safety factor

            return min(target_top_n, max_tokens_by_memory, vocab_size)

        return min(target_top_n, vocab_size)

    def _get_top_n_memory_efficient(
        self, logits: torch.Tensor, top_n: int
    ) -> List[int]:
        """Get top-n tokens with minimal memory overhead."""
        # Use partial sorting for memory efficiency
        values, indices = torch.topk(logits, top_n, largest=True, sorted=False)
        return indices.cpu().tolist()

    def _get_embeddings_memory_efficient(
        self, candidates: List[int], embedding_layer: torch.nn.Embedding
    ) -> torch.Tensor:
        """Get embeddings with memory constraints."""
        # Process in smaller batches if necessary
        max_batch_size = min(100, len(candidates))

        embeddings_list = []
        for i in range(0, len(candidates), max_batch_size):
            batch_candidates = candidates[i : i + max_batch_size]
            batch_tensor = torch.tensor(batch_candidates, device=self.device)

            with torch.no_grad():
                batch_embeddings = embedding_layer(batch_tensor)
                embeddings_list.append(
                    batch_embeddings.cpu()
                )  # Move to CPU to save GPU memory

        return torch.cat(embeddings_list, dim=0).to(self.device)

    def _cluster_memory_efficient(self, embeddings: torch.Tensor, k: int) -> np.ndarray:
        """Memory-efficient clustering."""
        # Move to CPU for clustering to save GPU memory
        embeddings_cpu = embeddings.cpu().numpy()

        # Use MiniBatchKMeans with small batch size
        effective_k = min(k, len(embeddings_cpu))
        batch_size = min(32, len(embeddings_cpu))

        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            batch_size=batch_size,
            max_iter=5,  # Reduced iterations for speed
            random_state=42,
        )

        labels = kmeans.fit_predict(embeddings_cpu)
        return labels

    def _select_token_from_clusters(
        self,
        logits: torch.Tensor,
        candidates: List[int],
        labels: np.ndarray,
        filter_tokens: Optional[set],
    ) -> int:
        """Standard token selection from clusters."""
        candidate_logits = logits[candidates]
        candidate_probs = torch.softmax(candidate_logits, dim=0)

        # Group by clusters
        k = len(np.unique(labels))
        clusters = [[] for _ in range(k)]
        for i, label in enumerate(labels):
            clusters[label].append(i)

        # Calculate cluster probabilities
        cluster_probs = [
            candidate_probs[cluster].sum().item() if cluster else 0.0
            for cluster in clusters
        ]

        if sum(cluster_probs) == 0:
            return candidates[torch.argmax(candidate_probs).item()]

        # Select best cluster and token
        best_cluster_idx = np.argmax(cluster_probs)
        cluster_indices = clusters[best_cluster_idx]

        # Apply filtering
        if filter_tokens:
            valid_indices = [
                i for i in cluster_indices if candidates[i] in filter_tokens
            ]
            if valid_indices:
                cluster_indices = valid_indices

        # Select best token from cluster
        cluster_probs_tensor = candidate_probs[cluster_indices]
        best_in_cluster = torch.argmax(cluster_probs_tensor).item()
        return candidates[cluster_indices[best_in_cluster]]
