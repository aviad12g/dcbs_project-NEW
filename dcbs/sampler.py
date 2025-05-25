"""
Sampler interface and implementations for different token sampling strategies.

This module provides a unified interface for various sampling methods including
greedy, top-p, random, and DCBS (Deterministic Category Based Sampling).
"""

import random
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Set, Union

import numpy as np
import torch

from .cache_manager import CacheConfig, get_cache_manager
from .clustering import (
    CandidateSelector,
    KMeansClusterer,
    TokenClusterer,
    TopNCandidateSelector,
)

# Small value to prevent division by zero
PROB_EPSILON = 1e-6


@dataclass
class SamplingContext:
    """Context object containing model-specific information for sampling."""

    embedding_layer: Optional[torch.nn.Embedding] = None
    tokenizer: Optional[object] = None
    device: Optional[torch.device] = None


class Sampler(ABC):
    """Abstract base class for token sampling strategies."""

    @abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """Sample a token from the given logits.

        Args:
            logits: Token logits from model
            filter_tokens: Optional set of allowed token IDs
            context: Optional context containing model information

        Returns:
            Selected token ID
        """
        pass


class GreedySampler(Sampler):
    """Greedy sampling - always selects the highest probability token."""

    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """Sample using greedy selection (argmax).

        Args:
            logits: Token logits
            filter_tokens: Optional set of allowed token IDs
            context: Unused for greedy sampling

        Returns:
            Token ID with highest probability
        """
        if filter_tokens is not None and len(filter_tokens) > 0:
            # Create mask for allowed tokens (more intuitive logic)
            allowed_mask = torch.full_like(logits, float("-inf"))
            allowed_indices = list(filter_tokens)
            allowed_mask[allowed_indices] = logits[allowed_indices]
            return allowed_mask.argmax().item()

        return logits.argmax().item()


class TopPSampler(Sampler):
    """Top-p (nucleus) sampling - samples from top tokens until cumulative probability >= p."""

    def __init__(self, p: float = 0.9):
        """Initialize top-p sampler.

        Args:
            p: Probability threshold for nucleus sampling
        """
        self.p = p

    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """Sample using top-p (nucleus) sampling.

        Args:
            logits: Token logits
            filter_tokens: Optional set of allowed token IDs
            context: Unused for top-p sampling

        Returns:
            Sampled token ID
        """
        # Apply filtering first if provided
        if filter_tokens is not None and len(filter_tokens) > 0:
            filtered_logits = torch.full_like(logits, float("-inf"))
            allowed_indices = list(filter_tokens)
            filtered_logits[allowed_indices] = logits[allowed_indices]
            logits = filtered_logits

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float("-inf")

        # Sample from filtered distribution
        token_id = torch.multinomial(torch.softmax(filtered_logits, dim=-1), 1).item()
        return token_id


class RandomSampler(Sampler):
    """Random sampling - uniformly samples from allowed tokens."""

    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """Sample randomly from allowed tokens.

        Args:
            logits: Token logits (used only for shape/device info)
            filter_tokens: Optional set of allowed token IDs
            context: Unused for random sampling

        Returns:
            Randomly selected token ID
        """
        if filter_tokens is not None and len(filter_tokens) > 0:
            return random.choice(list(filter_tokens))
        else:
            return random.randint(0, logits.shape[-1] - 1)


class DCBSSampler(Sampler):
    """Deterministic Category Based Sampling using clustering abstractions."""

    def __init__(
        self,
        clusterer: TokenClusterer,
        candidate_selector: CandidateSelector,
        cache_config: Optional[dict] = None,
        debug_mode: Optional[bool] = None,
        enable_cluster_history: Optional[bool] = None,
        debug_output_file: Optional[str] = None,
    ):
        """
        Initialize DCBS sampler with clustering and candidate selection strategies.

        Args:
            clusterer: Token clustering strategy
            candidate_selector: Candidate token selection strategy
            cache_config: Configuration for embedding and clustering caches
            debug_mode: Enable debug mode (overrides environment variable)
            enable_cluster_history: Track clustering decisions for analysis
            debug_output_file: File path for debug output (optional)
        """
        self.clusterer = clusterer
        self.candidate_selector = candidate_selector

        # Initialize thread-safe cache manager
        if cache_config:
            config = CacheConfig(**cache_config)
        else:
            config = CacheConfig(
                embedding_cache_size=1000, cluster_cache_size=200, enable_metrics=True
            )

        self.cache_manager = get_cache_manager(config)

        # DCBS algorithm parameters
        self.min_tokens_for_clustering = 3

        # Configurable debugging features
        self._debug_mode = self._resolve_debug_mode(debug_mode)
        self._enable_cluster_history = self._resolve_cluster_history(enable_cluster_history)
        self._debug_output_file = debug_output_file or os.environ.get("DCBS_DEBUG_OUTPUT_FILE")
        
        # Initialize debug tracking
        self._cluster_history = [] if self._enable_cluster_history else None
        self._debug_stats = {"total_samples": 0, "clustering_calls": 0, "cache_hits": 0}

    def _resolve_debug_mode(self, debug_mode: Optional[bool]) -> bool:
        """Resolve debug mode from parameter, environment variable, or default."""
        if debug_mode is not None:
            return debug_mode
        
        # Check environment variable
        env_debug = os.environ.get("DCBS_DEBUG_MODE", "").lower()
        if env_debug in ("true", "1", "yes", "on"):
            return True
        elif env_debug in ("false", "0", "no", "off"):
            return False
        
        # Default to False
        return False
    
    def _resolve_cluster_history(self, enable_cluster_history: Optional[bool]) -> bool:
        """Resolve cluster history tracking from parameter, environment variable, or default."""
        if enable_cluster_history is not None:
            return enable_cluster_history
        
        # Check environment variable
        env_history = os.environ.get("DCBS_ENABLE_CLUSTER_HISTORY", "").lower()
        if env_history in ("true", "1", "yes", "on"):
            return True
        elif env_history in ("false", "0", "no", "off"):
            return False
        
        # Default to False (only enable if debug mode is on)
        return self._debug_mode

    @classmethod
    def create_default(
        cls, k: int = 8, top_n: int = 50, cache_config: Optional[dict] = None,
        debug_mode: Optional[bool] = None, enable_cluster_history: Optional[bool] = None
    ):
        """Create DCBS sampler with default k-means clustering and top-n candidate selection."""
        clusterer = KMeansClusterer(k=k)
        candidate_selector = TopNCandidateSelector(top_n=top_n)
        return cls(clusterer, candidate_selector, cache_config, debug_mode, enable_cluster_history)

    def sample(
        self,
        logits: torch.Tensor,
        context: SamplingContext,
        filter_tokens: Optional[Set[int]] = None,
    ) -> int:
        """Sample using DCBS algorithm.

        Args:
            logits: Token logits
            context: Required context containing embedding layer
            filter_tokens: Optional set of allowed token IDs

        Returns:
            Selected token ID

        Raises:
            ValueError: If context or embedding layer is not provided
        """
        if context is None or context.embedding_layer is None:
            raise ValueError("DCBS requires a SamplingContext with embedding_layer")

        embedding = context.embedding_layer

        # Handle simple cases first
        if filter_tokens and len(filter_tokens) == 1:
            return list(filter_tokens)[0]

        # Get candidate tokens using strategy
        candidate_ids = self.candidate_selector.select_candidates(logits, filter_tokens)

        # Handle insufficient candidates for clustering
        if len(candidate_ids) <= self.min_tokens_for_clustering:
            return self._simple_selection(logits, candidate_ids)

        # Handle edge cases with invalid logits
        if self._has_invalid_logits(logits, candidate_ids):
            return self._fallback_selection(logits, filter_tokens)

        # Main DCBS algorithm
        return self._dcbs_selection(logits, candidate_ids, embedding, filter_tokens)

    def _simple_selection(self, logits: torch.Tensor, candidate_ids: list) -> int:
        """Handle simple selection when too few candidates for clustering."""
        candidate_logits = logits[candidate_ids]
        probs = torch.softmax(candidate_logits, dim=0)
        selected_idx = torch.argmax(probs).item()
        return candidate_ids[selected_idx]

    def _has_invalid_logits(self, logits: torch.Tensor, candidate_ids: list) -> bool:
        """Check if candidate logits contain invalid values."""
        candidate_logits = logits[candidate_ids]
        return (
            torch.isinf(candidate_logits).all() or torch.isnan(candidate_logits).any()
        )

    def _fallback_selection(
        self, logits: torch.Tensor, filter_tokens: Optional[Set[int]]
    ) -> int:
        """Fallback selection for edge cases."""
        if filter_tokens:
            filter_list = list(filter_tokens)
            filter_logits = logits[filter_list]
            best_idx = torch.argmax(filter_logits).item()
            return filter_list[best_idx]
        else:
            return logits.argmax().item()

    def _dcbs_selection(
        self,
        logits: torch.Tensor,
        candidate_ids: list,
        embedding: torch.nn.Embedding,
        filter_tokens: Optional[Set[int]],
    ) -> int:
        """Main DCBS selection algorithm."""
        self._debug_stats["total_samples"] += 1
        
        candidate_ids_tensor = torch.tensor(candidate_ids, device=logits.device)
        candidate_logits = logits[candidate_ids_tensor]
        candidate_probs = torch.softmax(candidate_logits, dim=0)

        self._log_debug(f"Starting DCBS selection with {len(candidate_ids)} candidates")

        # Get embeddings using thread-safe cache
        candidate_embeddings = self._get_cached_embeddings(
            candidate_ids_tensor, embedding
        )

        # Normalize embeddings
        norm_embeddings = self._normalize_embeddings(candidate_embeddings)

        # Perform clustering using strategy
        self._debug_stats["clustering_calls"] += 1
        labels = self._get_cached_clustering(norm_embeddings)
        
        self._log_debug(f"Clustering produced {len(set(labels))} clusters from {self.clusterer.num_clusters} requested")

        # Select token from clusters
        selected_token = self._select_from_clusters(
            candidate_ids, candidate_probs, labels, filter_tokens
        )
        
        # Record decision for analysis
        if self._enable_cluster_history:
            # Find which cluster was selected
            selected_idx = candidate_ids.index(selected_token)
            selected_cluster = labels[selected_idx]
            self._record_cluster_decision(candidate_ids, labels, selected_cluster, selected_token)
        
        self._log_debug(f"Selected token {selected_token}")
        
        return selected_token

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Normalize embeddings for clustering."""
        norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        return embeddings / norm.clamp(min=PROB_EPSILON)

    def _select_from_clusters(
        self,
        candidate_ids: list,
        candidate_probs: torch.Tensor,
        labels: np.ndarray,
        filter_tokens: Optional[Set[int]],
    ) -> int:
        """Select token from clustered candidates using greedy selection."""
        # Group tokens by cluster and calculate probabilities
        clusters = self._group_by_clusters(labels, self.clusterer.num_clusters)
        cluster_probs = self._calculate_cluster_probabilities(clusters, candidate_probs)

        if sum(cluster_probs) == 0:
            return self._fallback_selection(None, filter_tokens)

        # GREEDY selection of cluster (deterministic argmax)
        selected_cluster_idx = np.argmax(cluster_probs)
        cluster_token_indices = clusters[selected_cluster_idx]

        # Apply filtering and select best token using GREEDY selection
        return self._select_best_from_cluster(
            candidate_ids, cluster_token_indices, candidate_probs, filter_tokens
        )

    def _group_by_clusters(
        self, labels: np.ndarray, num_clusters: int
    ) -> List[List[int]]:
        """Group token indices by cluster labels."""
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        return clusters

    def _calculate_cluster_probabilities(
        self, clusters: List[List[int]], candidate_probs: torch.Tensor
    ) -> List[float]:
        """Calculate total probability mass for each cluster."""
        return [
            candidate_probs[cluster].sum().item() if cluster else 0.0
            for cluster in clusters
        ]

    def _select_best_from_cluster(
        self,
        candidate_ids: list,
        cluster_token_indices: List[int],
        candidate_probs: torch.Tensor,
        filter_tokens: Optional[Set[int]],
    ) -> int:
        """Select the best token from within a cluster using GREEDY selection."""
        cluster_token_probs = candidate_probs[cluster_token_indices]

        # Apply filtering within cluster if needed
        if filter_tokens:
            valid_indices = [
                i
                for i, token_idx in enumerate(cluster_token_indices)
                if candidate_ids[token_idx] in filter_tokens
            ]

            if not valid_indices:
                return self._fallback_selection(None, filter_tokens)

            cluster_token_indices = [cluster_token_indices[i] for i in valid_indices]
            cluster_token_probs = cluster_token_probs[valid_indices]

        # GREEDY selection of highest probability token from cluster (deterministic argmax)
        selected_in_cluster_idx = torch.argmax(cluster_token_probs).item()
        selected_token_idx = cluster_token_indices[selected_in_cluster_idx]

        return candidate_ids[selected_token_idx]

    def _get_cached_embeddings(
        self, token_ids: torch.Tensor, embedding: torch.nn.Embedding
    ) -> torch.Tensor:
        """Get embeddings using thread-safe cache manager."""
        device = token_ids.device
        token_ids_list = token_ids.cpu().tolist()

        # Use optimized batch retrieval
        result, uncached_indices = self.cache_manager.get_batch_embeddings(
            token_ids_list, embedding, device
        )

        # Fetch uncached embeddings
        if uncached_indices:
            uncached_ids = [token_ids_list[i] for i in uncached_indices]

            # Bounds checking
            max_token_id = max(uncached_ids)
            if embedding.weight.shape[0] > max_token_id:
                with torch.no_grad():
                    uncached_embeds = embedding(
                        torch.tensor(uncached_ids, device=device)
                    )

                    # Update result tensor and cache
                    for i, batch_idx in enumerate(uncached_indices):
                        result[batch_idx] = uncached_embeds[i]

                    # Cache the new embeddings
                    self.cache_manager.cache_batch_embeddings(
                        token_ids_list, uncached_indices, uncached_embeds
                    )

        return result

    def _get_cached_clustering(self, embeddings: torch.Tensor) -> np.ndarray:
        """Get clustering results using strategy and cache."""
        device_str = str(embeddings.device)
        cache_key = (embeddings.shape[0], self.clusterer.num_clusters, device_str)

        # Try to get from cache
        cached_labels = self.cache_manager.get_clustering(cache_key)
        if cached_labels is not None:
            return cached_labels

        # Compute clustering using strategy
        labels = self.clusterer.cluster(embeddings)

        # Cache the result
        self.cache_manager.put_clustering(cache_key, labels)

        return labels

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        return self.cache_manager.get_cache_stats()

    def clear_caches(self) -> None:
        """Clear all caches."""
        self.cache_manager.clear_all_caches()

    def _log_debug(self, message: str) -> None:
        """Log debug message if debug mode is enabled."""
        if not self._debug_mode:
            return
        
        debug_msg = f"[DCBS DEBUG] {message}"
        
        if self._debug_output_file:
            try:
                with open(self._debug_output_file, "a") as f:
                    f.write(f"{debug_msg}\n")
            except IOError:
                # Fallback to print if file writing fails
                print(debug_msg)
        else:
            print(debug_msg)
    
    def _record_cluster_decision(self, candidate_ids: list, labels: np.ndarray, 
                               selected_cluster: int, selected_token: int) -> None:
        """Record clustering decision for analysis if enabled."""
        if not self._enable_cluster_history or self._cluster_history is None:
            return
        
        decision = {
            "candidate_count": len(candidate_ids),
            "num_clusters": len(set(labels)),
            "selected_cluster": selected_cluster,
            "selected_token": selected_token,
            "cluster_sizes": [np.sum(labels == i) for i in range(self.clusterer.num_clusters)]
        }
        
        self._cluster_history.append(decision)
        
        # Log if debug mode is also enabled
        if self._debug_mode:
            self._log_debug(f"Cluster decision: {decision}")
    
    def get_debug_stats(self) -> dict:
        """Get debugging statistics."""
        stats = self._debug_stats.copy()
        stats["debug_mode"] = self._debug_mode
        stats["cluster_history_enabled"] = self._enable_cluster_history
        stats["cluster_history_count"] = len(self._cluster_history) if self._cluster_history else 0
        return stats
    
    def get_cluster_history(self) -> Optional[List[dict]]:
        """Get cluster decision history if enabled."""
        return self._cluster_history.copy() if self._cluster_history else None
    
    def clear_debug_data(self) -> None:
        """Clear debug data and statistics."""
        if self._cluster_history:
            self._cluster_history.clear()
        self._debug_stats = {"total_samples": 0, "clustering_calls": 0, "cache_hits": 0}
