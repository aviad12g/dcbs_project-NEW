"""
Factory for creating and managing sampler instances.

This module provides a factory class for creating different types
of samplers based on evaluation configuration.
"""

from typing import Dict, Optional

from src.dcbs import (
    DCBSSampler, 
    GreedySampler, 
    RandomSampler, 
    TopPSampler, 
    SamplingContext,
    KMeansClusterer,
    DBSCANClusterer,
    HierarchicalClusterer,
    TopNCandidateSelector,
    CategorySampler,
    GreedyCategorySelector,
    GreedyTokenSelector,
    TemperatureSampler,
    TopKSampler,
)
from .config import EvaluationConfig


class SamplerFactory:
    """Factory for creating and managing sampler instances."""

    @staticmethod
    def create_dcbs_sampler(
        config: EvaluationConfig, 
        context: Optional[SamplingContext] = None,
        clustering_method: str = "dbscan",
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 2,
        hierarchical_linkage: str = "average",
        debug_mode: bool = False,
        enable_cluster_history: bool = False,
    ) -> DCBSSampler:
        """
        Create a DCBS sampler with the specified clustering method.
        
        Args:
            config: Evaluation configuration
            context: Sampling context for DCBS sampler
            clustering_method: Clustering method to use ("kmeans", "dbscan", "hierarchical")
            dbscan_eps: DBSCAN epsilon parameter
            dbscan_min_samples: DBSCAN minimum samples parameter
            hierarchical_linkage: Linkage criterion for hierarchical clustering
            
        Returns:
            Configured DCBS sampler instance
        """
        # Create candidate selector
        candidate_selector = TopNCandidateSelector(top_n=config.top_n)
        
        # Create category sampler
        category_sampler = CategorySampler(
            category_selector=GreedyCategorySelector(),
            token_selector=GreedyTokenSelector()
        )
        
        # Create clusterer based on method
        if clustering_method == "kmeans":
            clusterer = KMeansClusterer(k=config.k)
        elif clustering_method == "dbscan":
            clusterer = DBSCANClusterer(
                eps=dbscan_eps,
                min_samples=dbscan_min_samples
            )
        elif clustering_method == "hierarchical":
            clusterer = HierarchicalClusterer(
                k=config.k,
                linkage=hierarchical_linkage
            )
        else:
            raise ValueError(f"Unknown clustering method: {clustering_method}")
        
        return DCBSSampler(
            clusterer=clusterer,
            candidate_selector=candidate_selector,
            category_sampler=category_sampler,
            context=context,
            enable_caching=config.enable_caching,
            debug_mode=debug_mode,
            enable_cluster_history=enable_cluster_history,
        )

    @staticmethod
    def create_samplers(
        config: EvaluationConfig,
        context: Optional[SamplingContext] = None,
        clustering_method: Optional[str] = None,
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 2,
        hierarchical_linkage: str = "average",
        debug_mode: bool = False,
        enable_cluster_history: bool = False,
    ) -> Dict[str, object]:
        """
        Create all sampler instances based on configuration.
        
        Args:
            config: Evaluation configuration containing sampler parameters
            context: Sampling context for DCBS sampler
            clustering_method: Clustering method for DCBS
            dbscan_eps: DBSCAN epsilon parameter
            dbscan_min_samples: DBSCAN minimum samples parameter
            hierarchical_linkage: Linkage criterion for hierarchical clustering
            debug_mode: Enable verbose debug logging for DCBS
            enable_cluster_history: Record cluster assignments and probabilities
            
        Returns:
            Dictionary mapping sampler names to sampler instances
        """
        # Use config's clustering method if none specified
        effective_clustering_method = clustering_method or getattr(config, 'clustering_method', 'dbscan')
        
        samplers = {
            "greedy": GreedySampler(),
            "top_p": TopPSampler(p=config.top_p),
            "dcbs": SamplerFactory.create_dcbs_sampler(
                config,
                context,
                effective_clustering_method,
                dbscan_eps,
                dbscan_min_samples,
                hierarchical_linkage,
                debug_mode=debug_mode,
                enable_cluster_history=enable_cluster_history,
            ),
            "random": RandomSampler(),
        }

        # Add TemperatureSampler if temperature is specified
        if config.temperature is not None:
            samplers["temperature"] = TemperatureSampler(temperature=config.temperature)

        # Add TopKSampler if top_k is specified
        if config.top_k is not None:
            samplers["top_k"] = TopKSampler(k=config.top_k)

        return samplers 