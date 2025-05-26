"""
Deterministic Category Based Sampling implementation.

This module provides DCBS which combines semantic clustering of token embeddings
with deterministic selection for reproducible yet diverse sampling.
"""

import os
from typing import List, Optional, Set

import numpy as np
import torch

from .base import Sampler, SamplingContext, PROB_EPSILON
from ..cache_manager import CacheConfig, get_cache_manager
from ..clustering import (
    CandidateSelector,
    TokenClusterer,
)
from ..category_sampling import CategorySampler, greedy_category_sampler


class DCBSSampler(Sampler):
    """
    Deterministic Category Based Sampling using clustering abstractions.
    
    This sampling method clusters token embeddings into semantic categories,
    selects the cluster with highest probability mass, then chooses the
    highest probability token from that cluster. All selections are deterministic.
    """

    def __init__(
        self,
        clusterer: TokenClusterer,
        candidate_selector: CandidateSelector,
        category_sampler: Optional[CategorySampler] = None,
        cache_config: Optional[dict] = None,
        enable_caching: bool = True,
        debug_mode: Optional[bool] = None,
        enable_cluster_history: Optional[bool] = None,
        debug_output_file: Optional[str] = None,
    ):
        """
        Initialize the DCBS sampler.
        
        Args:
            clusterer: Token clustering strategy
            candidate_selector: Candidate token selection strategy
            category_sampler: Strategy for sampling from categories (default: greedy)
            cache_config: Optional cache configuration
            enable_caching: Whether to enable caching (default: True)
            debug_mode: Enable debug logging (default: False)
            enable_cluster_history: Track cluster decisions (default: False)
            debug_output_file: File path for debug output
        """
        self.clusterer = clusterer
        self.candidate_selector = candidate_selector
        self.category_sampler = category_sampler or greedy_category_sampler
        self.enable_caching = enable_caching

        # Initialize cache manager only if caching is enabled
        if self.enable_caching:
            if cache_config:
                config = CacheConfig(**cache_config)
            else:
                # Default cache configuration with reasonable limits
                config = CacheConfig(
                    embedding_cache_size=1000, cluster_cache_size=200, enable_metrics=True
                )
            self.cache_manager = get_cache_manager(config)
        else:
            self.cache_manager = None

        # DCBS algorithm parameters
        self.min_tokens_for_clustering = 3  # Minimum tokens needed for clustering

        # Configurable debugging features
        self._debug_mode = self._resolve_debug_mode(debug_mode)
        self._enable_cluster_history = self._resolve_cluster_history(enable_cluster_history)
        self._debug_output_file = debug_output_file or os.environ.get("DCBS_DEBUG_OUTPUT_FILE")
        
        # Initialize debug tracking
        self._cluster_history = [] if self._enable_cluster_history else None
        self._debug_stats = {"total_samples": 0, "clustering_calls": 0, "cache_hits": 0}

    def _resolve_debug_mode(self, debug_mode: Optional[bool]) -> bool:
        """Resolve debug mode from parameter or environment variable."""
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
        """Resolve cluster history tracking from parameter or environment variable."""
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
        enable_caching: bool = True, debug_mode: Optional[bool] = None, 
        enable_cluster_history: Optional[bool] = None
    ):
        """
        Create a DCBS sampler with default clustering and candidate selection.
        
        Note: This method imports specific implementations to maintain backward 
        compatibility. For new code, prefer dependency injection by directly 
        passing clusterer and candidate_selector instances.
        
        Args:
            k: Number of clusters for K-means (default: 8)
            top_n: Number of top tokens to consider (default: 50)
            cache_config: Optional cache configuration
            enable_caching: Whether to enable caching
            debug_mode: Enable debug logging
            enable_cluster_history: Track cluster decisions
            
        Returns:
            Configured DCBSSampler instance
        """
        # Import here to avoid tight coupling at module level
        from ..clustering import KMeansClusterer, TopNCandidateSelector
        from ..category_sampling import CategorySampler, GreedyCategorySelector, GreedyTokenSelector
        
        clusterer = KMeansClusterer(k=k)
        candidate_selector = TopNCandidateSelector(top_n=top_n)
        category_sampler = CategorySampler(
            category_selector=GreedyCategorySelector(),
            token_selector=GreedyTokenSelector()
        )
        
        return cls(clusterer, candidate_selector, category_sampler, cache_config, 
                  enable_caching, debug_mode, enable_cluster_history)

    @classmethod
    def create_no_cache(cls, k: int = 8, top_n: int = 50, **kwargs):
        """Create a DCBS sampler with caching disabled."""
        return cls.create_default(k=k, top_n=top_n, enable_caching=False, **kwargs)

    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """
        Sample a token using DCBS algorithm.
        
        Args:
            logits: Token logits from the model
            filter_tokens: Optional set of allowed token IDs
            context: Sampling context with embedding layer (required for DCBS)
            
        Returns:
            Selected token ID
            
        Raises:
            ValueError: If context or embedding_layer is missing
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
        """Select best token when clustering is not applicable."""
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
        """Fallback selection when DCBS cannot be applied."""
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
        """Main DCBS algorithm implementation."""
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

        # Group tokens by cluster
        clusters = self._group_by_clusters(labels, self.clusterer.num_clusters)
        
        # Use category sampler to select token
        selected_token = self.category_sampler.sample_from_clusters(
            candidate_ids, candidate_probs, clusters, filter_tokens
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
        """Normalize embeddings to unit vectors."""
        norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        return embeddings / norm.clamp(min=PROB_EPSILON)

    def _group_by_clusters(
        self, labels: np.ndarray, num_clusters: int
    ) -> List[List[int]]:
        """Group token indices by their cluster labels."""
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        return clusters

    def _get_cached_embeddings(
        self, token_ids: torch.Tensor, embedding: torch.nn.Embedding
    ) -> torch.Tensor:
        """Get token embeddings with optional caching."""
        if not self.enable_caching or self.cache_manager is None:
            # Direct embedding lookup without caching
            with torch.no_grad():
                return embedding(token_ids)
        
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
        """Get clustering results with optional caching."""
        if not self.enable_caching or self.cache_manager is None:
            return self.clusterer.cluster(embeddings)
        
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
        if not self.enable_caching or self.cache_manager is None:
            return {"caching_enabled": False, "message": "Caching is disabled"}
        return self.cache_manager.get_cache_stats()

    def clear_caches(self) -> None:
        """Clear all caches."""
        if self.cache_manager:
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