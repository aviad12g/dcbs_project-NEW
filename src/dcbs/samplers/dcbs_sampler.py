"""
Deterministic Category Based Sampling implementation.

This module provides DCBS which combines semantic clustering of token embeddings
with deterministic selection for reproducible yet diverse sampling.
"""

from typing import List, Optional, Set

import numpy as np
import torch

from .base import Sampler, SamplingContext
from ..cache_manager import CacheConfig, get_cache_manager
from ..clustering import (
    CandidateSelector,
    TokenClusterer,
    KMeansClusterer,
    TopNCandidateSelector,
)
from ..category_sampling import (
    CategorySampler, 
    greedy_category_sampler,
    GreedyCategorySelector,
    GreedyTokenSelector,
)
from ..constants import (
    MIN_TOKENS_FOR_CLUSTERING,
    DEFAULT_K_CLUSTERS,
    DEFAULT_TOP_N,
    DEFAULT_EMBEDDING_CACHE_SIZE,
    DEFAULT_CLUSTER_CACHE_SIZE,
)
from ..debug import DCBSDebugger
from ..embedding_ops import EmbeddingOperations


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
        context: Optional[SamplingContext] = None,
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
            context: Sampling context with embedding layer (required for DCBS)
            cache_config: Optional cache configuration
            enable_caching: Whether to enable caching (default: True)
            debug_mode: Enable debug logging (default: False)
            enable_cluster_history: Track cluster decisions (default: False)
            debug_output_file: File path for debug output
        """
        self.clusterer = clusterer
        self.candidate_selector = candidate_selector
        self.category_sampler = category_sampler or greedy_category_sampler
        self.context = context
        self.enable_caching = enable_caching

        # Initialize cache manager only if caching is enabled
        if self.enable_caching:
            if cache_config:
                config = CacheConfig(**cache_config)
            else:
                # Default cache configuration with reasonable limits
                config = CacheConfig(
                    embedding_cache_size=DEFAULT_EMBEDDING_CACHE_SIZE, 
                    cluster_cache_size=DEFAULT_CLUSTER_CACHE_SIZE, 
                    enable_metrics=True
                )
            self.cache_manager = get_cache_manager(config)
        else:
            self.cache_manager = None

        # Initialize debugging
        self.debugger = DCBSDebugger(debug_mode, enable_cluster_history, debug_output_file)
        
        # Initialize embedding operations
        self.embedding_ops = EmbeddingOperations(self.cache_manager)

    @classmethod
    def create_default(
        cls, k: int = DEFAULT_K_CLUSTERS, top_n: int = DEFAULT_TOP_N, 
        context: Optional[SamplingContext] = None,
        cache_config: Optional[dict] = None,
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
            context: Sampling context with embedding layer
            cache_config: Optional cache configuration
            enable_caching: Whether to enable caching
            debug_mode: Enable debug logging
            enable_cluster_history: Track cluster decisions
            
        Returns:
            Configured DCBSSampler instance
        """
        clusterer = KMeansClusterer(k=k)
        candidate_selector = TopNCandidateSelector(top_n=top_n)
        category_sampler = CategorySampler(
            category_selector=GreedyCategorySelector(),
            token_selector=GreedyTokenSelector()
        )
        
        return cls(clusterer, candidate_selector, category_sampler, context, cache_config, 
                  enable_caching, debug_mode, enable_cluster_history)

    @classmethod
    def create_no_cache(cls, k: int = DEFAULT_K_CLUSTERS, top_n: int = DEFAULT_TOP_N, 
                       context: Optional[SamplingContext] = None, **kwargs):
        """Create a DCBS sampler with caching disabled."""
        return cls.create_default(k=k, top_n=top_n, context=context, enable_caching=False, **kwargs)

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
            context: Sampling context (optional, uses instance context if not provided)
            
        Returns:
            Selected token ID
            
        Raises:
            ValueError: If context or embedding_layer is missing
        """
        # Use provided context or instance context
        effective_context = context or self.context
        
        if effective_context is None or effective_context.embedding_layer is None:
            raise ValueError("DCBS requires a SamplingContext with embedding_layer")

        embedding = effective_context.embedding_layer

        # Handle simple cases first
        if filter_tokens and len(filter_tokens) == 1:
            return list(filter_tokens)[0]

        # Get candidate tokens using strategy
        candidate_ids = self.candidate_selector.select_candidates(logits, filter_tokens)

        # Handle insufficient candidates for clustering
        if len(candidate_ids) <= MIN_TOKENS_FOR_CLUSTERING:
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
        self.debugger.increment_stat("total_samples")
        
        # Prepare candidate data
        candidate_ids_tensor = torch.tensor(candidate_ids, device=logits.device)
        candidate_logits = logits[candidate_ids_tensor]
        candidate_probs = torch.softmax(candidate_logits, dim=0)

        self.debugger.log_debug(f"Starting DCBS selection with {len(candidate_ids)} candidates")

        # Get normalized embeddings
        norm_embeddings = self.embedding_ops.get_normalized_embeddings(
            candidate_ids_tensor, embedding
        )

        # Perform clustering
        labels = self._perform_clustering(norm_embeddings)
        
        # Group tokens by cluster
        clusters = self._group_by_clusters(labels, self.clusterer.num_clusters)
        
        # Select token using category sampler
        selected_token = self.category_sampler.sample_from_clusters(
            candidate_ids, candidate_probs, clusters, filter_tokens
        )
        
        # Record decision for analysis
        self._record_selection_decision(
            candidate_ids, labels, selected_token
        )
        
        self.debugger.log_debug(f"Selected token {selected_token}")
        
        return selected_token

    def _perform_clustering(self, embeddings: torch.Tensor) -> np.ndarray:
        """Perform clustering on embeddings with optional caching."""
        self.debugger.increment_stat("clustering_calls")
        
        if not self.enable_caching or self.cache_manager is None:
            return self.clusterer.cluster(embeddings)
        
        # Try cached clustering
        device_str = str(embeddings.device)
        cache_key = (embeddings.shape[0], self.clusterer.num_clusters, device_str)
        
        cached_labels = self.cache_manager.get_clustering(cache_key)
        if cached_labels is not None:
            self.debugger.increment_stat("cache_hits")
            return cached_labels
        
        # Compute and cache
        labels = self.clusterer.cluster(embeddings)
        self.cache_manager.put_clustering(cache_key, labels)
        
        self.debugger.log_debug(
            f"Clustering produced {len(set(labels))} clusters from {self.clusterer.num_clusters} requested"
        )
        
        return labels

    def _group_by_clusters(
        self, labels: np.ndarray, num_clusters: int
    ) -> List[List[int]]:
        """Group token indices by their cluster labels."""
        clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        return clusters

    def _record_selection_decision(
        self, candidate_ids: list, labels: np.ndarray, selected_token: int
    ) -> None:
        """Record clustering decision for debugging if enabled."""
        if self.debugger.cluster_history_enabled:
            # Find which cluster was selected
            selected_idx = candidate_ids.index(selected_token)
            selected_cluster = labels[selected_idx]
            self.debugger.record_cluster_decision(
                candidate_ids, labels, selected_cluster, selected_token, 
                self.clusterer.num_clusters
            )

    def get_cache_stats(self) -> dict:
        """Get cache performance statistics."""
        if not self.enable_caching or self.cache_manager is None:
            return {"caching_enabled": False, "message": "Caching is disabled"}
        return self.cache_manager.get_cache_stats()

    def clear_caches(self) -> None:
        """Clear all caches."""
        if self.cache_manager:
            self.cache_manager.clear_all_caches()

    def get_debug_stats(self) -> dict:
        """Get debugging statistics."""
        return self.debugger.get_stats()
    
    def get_cluster_history(self) -> Optional[List[dict]]:
        """Get cluster decision history if enabled."""
        return self.debugger.get_cluster_history()
    
    def clear_debug_data(self) -> None:
        """Clear debug data and statistics."""
        self.debugger.clear_debug_data() 