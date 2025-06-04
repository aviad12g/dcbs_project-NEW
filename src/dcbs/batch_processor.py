"""Batch processing optimizations for DCBS."""

import concurrent.futures
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from .cache_manager import DCBSCacheManager
from .samplers.base import SamplingContext


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
        """Process a batch of logits simultaneously for improved throughput."""
        batch_size = logits_batch.shape[0]

        if batch_size == 1:
            return [
                self._single_sample(logits_batch[0], filter_tokens_batch[0], context, k, top_n)
            ]

        if self.config.enable_parallel_processing and batch_size > 4:
            return self._parallel_batch_sample(logits_batch, filter_tokens_batch, context, k, top_n)
        else:
            return self._sequential_batch_sample(logits_batch, filter_tokens_batch, context, k, top_n)

    def _parallel_batch_sample(
        self,
        logits_batch: torch.Tensor,
        filter_tokens_batch: List[Optional[set]],
        context: SamplingContext,
        k: int,
        top_n: int,
    ) -> List[int]:
        """Process batch using parallel workers."""
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

        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._process_chunk, *chunk)
            futures.append(future)

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
        all_candidates = []
        all_embeddings = []
        with torch.no_grad():
            for logits, filter_tokens in zip(logits_batch, filter_tokens_batch):
                candidates = self._get_candidates(logits, filter_tokens, top_n)
                all_candidates.append(candidates)
                if len(candidates) > 3:
                    embeddings = self._batch_get_embeddings(candidates, context.embedding_layer)
                    all_embeddings.append(embeddings)
                else:
                    all_embeddings.append(None)

        for logits, candidates, embeddings, filter_tokens in zip(
            logits_batch, all_candidates, all_embeddings, filter_tokens_batch
        ):
            if embeddings is not None:
                result = self._optimized_dcbs_selection(logits, candidates, embeddings, filter_tokens, k)
            else:
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
        return self._optimized_dcbs_selection(logits, candidates, embeddings, filter_tokens, k)

    def _get_candidates(self, logits: torch.Tensor, filter_tokens: Optional[set], top_n: int) -> List[int]:
        """Get candidate tokens efficiently."""
        if filter_tokens:
            return list(filter_tokens)
        _, indices = torch.topk(logits, min(top_n, len(logits)))
        return indices.cpu().tolist()

    def _batch_get_embeddings(self, candidates: List[int], embedding_layer: torch.nn.Embedding) -> torch.Tensor:
        """Efficiently retrieve embeddings for candidates."""
        candidate_tensor = torch.tensor(candidates, device=self.device)
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

        if uncached_indices:
            uncached_tokens = candidate_tensor[uncached_indices]
            with torch.no_grad():
                if self.config.use_mixed_precision:
                    with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                        uncached_embeddings = embedding_layer(uncached_tokens)
                else:
                    uncached_embeddings = embedding_layer(uncached_tokens)
            for i, embedding in enumerate(uncached_embeddings):
                token_id = candidates[uncached_indices[i]]
                self.cache_manager.put_embedding(token_id, embedding)

        result = torch.zeros(
            (len(candidates), embedding_layer.embedding_dim), device=self.device, dtype=torch.float32
        )
        for idx, embedding in cached_embeddings:
            result[idx] = embedding
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
        candidate_logits = logits[candidates]
        candidate_probs = torch.softmax(candidate_logits, dim=-1)
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = embeddings / norms.clamp(min=1e-6)
        effective_k = min(k, len(candidates))
        if self.config.use_gpu_clustering and self.device.type == "cuda":
            labels = self._gpu_clustering(normalized_embeddings, effective_k)
        else:
            labels = self._cpu_clustering(normalized_embeddings, effective_k)
        clusters = [[] for _ in range(effective_k)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        cluster_probs = []
        for cluster in clusters:
            if cluster:
                cluster_prob = candidate_probs[cluster].sum().item()
                cluster_probs.append(cluster_prob)
            else:
                cluster_probs.append(0.0)
        if sum(cluster_probs) == 0:
            return candidates[torch.argmax(candidate_probs).item()]
        best_cluster_idx = np.argmax(cluster_probs)
        cluster_indices = clusters[best_cluster_idx]
        if filter_tokens:
            valid_indices = [i for i in cluster_indices if candidates[i] in filter_tokens]
            if not valid_indices:
                return candidates[torch.argmax(candidate_probs).item()]
            cluster_indices = valid_indices
        cluster_probs_tensor = candidate_probs[cluster_indices]
        best_in_cluster = torch.argmax(cluster_probs_tensor).item()
        return candidates[cluster_indices[best_in_cluster]]

    def _gpu_clustering(self, embeddings: torch.Tensor, k: int) -> np.ndarray:
        """GPU-accelerated clustering using PyTorch operations."""
        try:
            n_samples, _ = embeddings.shape
            torch.manual_seed(42)
            centroids = embeddings[torch.randperm(n_samples)[:k]]
            for _ in range(10):
                distances = torch.cdist(embeddings, centroids)
                assignments = torch.argmin(distances, dim=1)
                new_centroids = torch.zeros_like(centroids)
                for i in range(k):
                    mask = assignments == i
                    if mask.sum() > 0:
                        new_centroids[i] = embeddings[mask].mean(dim=0)
                    else:
                        new_centroids[i] = centroids[i]
                centroids = new_centroids
            return assignments.cpu().numpy()
        except Exception:
            return self._cpu_clustering(embeddings, k)

    def _cpu_clustering(self, embeddings: torch.Tensor, k: int) -> np.ndarray:
        """CPU clustering using scikit-learn."""
        embeddings_np = embeddings.detach().cpu().numpy()
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=max(32, len(embeddings_np)),
            max_iter=10,
            random_state=42,
            n_init=1,
        )
        labels = kmeans.fit_predict(embeddings_np)
        return labels

    def _simple_selection(self, logits: torch.Tensor, candidates: List[int]) -> int:
        """Simple greedy selection for small candidate sets."""
        candidate_logits = logits[candidates]
        best_idx = torch.argmax(candidate_logits).item()
        return candidates[best_idx]

    def precompute_embeddings(self, token_ids: List[int], embedding_layer: torch.nn.Embedding) -> None:
        """Precompute and cache embeddings for frequently used tokens."""
        batch_size = 1000
        for i in range(0, len(token_ids), batch_size):
            batch_tokens = token_ids[i : i + batch_size]
            uncached_tokens = [
                token_id for token_id in batch_tokens if self.cache_manager.get_embedding(token_id) is None
            ]
            if uncached_tokens:
                token_tensor = torch.tensor(uncached_tokens, device=self.device)
                with torch.no_grad():
                    if self.config.use_mixed_precision:
                        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                            embeddings = embedding_layer(token_tensor)
                    else:
                        embeddings = embedding_layer(token_tensor)
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
