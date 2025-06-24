"""
Factory module for creating configured DCBS samplers.

This module provides factory methods to create pre-configured samplers
without circular dependency issues. It imports concrete implementations
to avoid the main sampler classes having to import their dependencies.
"""

from typing import Optional

from .cache_manager import CacheConfig
from .clustering import KMeansClusterer, TopNCandidateSelector
from .category_sampling import CategorySampler, ConfidenceAwareCategorySelector, GreedyTokenSelector
from .constants import DEFAULT_K_CLUSTERS, DEFAULT_TOP_N, DEFAULT_EMBEDDING_CACHE_SIZE, DEFAULT_CLUSTER_CACHE_SIZE
from .samplers.base import SamplingContext
from .samplers.dcbs_sampler import DCBSSampler


class DCBSSamplerFactory:
    """Factory for creating pre-configured DCBS samplers."""
    
    @staticmethod
    def create_default(
        k: int = DEFAULT_K_CLUSTERS, 
        top_n: int = DEFAULT_TOP_N, 
        context: Optional[SamplingContext] = None,
        cache_config: Optional[dict] = None,
        enable_caching: bool = True, 
        debug_mode: Optional[bool] = None, 
        enable_cluster_history: Optional[bool] = None
    ) -> DCBSSampler:
        """
        Create a DCBS sampler with default clustering and candidate selection.
        
        This factory method encapsulates the creation of default implementations
        to avoid circular dependencies in the main DCBSSampler class.
        
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
            category_selector=ConfidenceAwareCategorySelector(),
            token_selector=GreedyTokenSelector()
        )
        
        return DCBSSampler(
            clusterer=clusterer, 
            candidate_selector=candidate_selector, 
            category_sampler=category_sampler, 
            context=context, 
            cache_config=cache_config, 
            enable_caching=enable_caching, 
            debug_mode=debug_mode, 
            enable_cluster_history=enable_cluster_history
        )

    @staticmethod
    def create_no_cache(
        k: int = DEFAULT_K_CLUSTERS, 
        top_n: int = DEFAULT_TOP_N, 
        context: Optional[SamplingContext] = None, 
        **kwargs
    ) -> DCBSSampler:
        """Create a DCBS sampler with caching disabled."""
        return DCBSSamplerFactory.create_default(
            k=k, top_n=top_n, context=context, enable_caching=False, **kwargs
        )
    
    @staticmethod
    def create_lightweight(
        k: int = 4, 
        top_n: int = 20, 
        context: Optional[SamplingContext] = None
    ) -> DCBSSampler:
        """Create a lightweight DCBS sampler for resource-constrained environments."""
        cache_config = {
            "embedding_cache_size": 100,
            "cluster_cache_size": 50,
            "enable_metrics": False
        }
        
        return DCBSSamplerFactory.create_default(
            k=k, 
            top_n=top_n, 
            context=context, 
            cache_config=cache_config,
            debug_mode=False,
            enable_cluster_history=False
        ) 