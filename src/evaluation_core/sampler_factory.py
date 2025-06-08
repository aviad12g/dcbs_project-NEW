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
    InformationGainCategorySelector,
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
        enable_cluster_history: bool = True,
        use_information_gain: bool = True,
        kl_threshold: float = 0.1,
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
            debug_mode: Enable verbose debug logging for DCBS
            enable_cluster_history: Record cluster assignments and probabilities
            use_information_gain: Use information-gain pruning to avoid low-probability clusters
            kl_threshold: KL divergence threshold for cluster pruning
            
        Returns:
            Configured DCBS sampler instance
        """
        # Create candidate selector
        candidate_selector = TopNCandidateSelector(top_n=config.top_n)
        
        # Create category sampler with optional information-gain pruning
        if use_information_gain:
            category_selector = InformationGainCategorySelector(
                kl_threshold=kl_threshold,
                min_clusters=1
            )
        else:
            category_selector = GreedyCategorySelector()
        
        category_sampler = CategorySampler(
            category_selector=category_selector,
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
        enable_cluster_history: bool = True,
        requested_samplers: Optional[list] = None,
        use_information_gain: bool = True,
        kl_threshold: float = 0.1,
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
            requested_samplers: List of sampler names to create (if None, creates all)
            use_information_gain: Use information-gain pruning for DCBS clusters
            kl_threshold: KL divergence threshold for cluster pruning
            
        Returns:
            Dictionary mapping sampler names to sampler instances
        """
        # Use config's clustering method if none specified
        effective_clustering_method = clustering_method or getattr(config, 'clustering_method', 'dbscan')
        
        # Define all available samplers
        all_samplers = {
            "greedy": lambda: GreedySampler(),
            "top_p": lambda: TopPSampler(p=config.top_p),
            "dcbs": lambda: SamplerFactory.create_dcbs_sampler(
                config,
                context,
                effective_clustering_method,
                dbscan_eps,
                dbscan_min_samples,
                hierarchical_linkage,
                debug_mode=debug_mode,
                enable_cluster_history=enable_cluster_history,
                use_information_gain=use_information_gain,
                kl_threshold=kl_threshold,
            ),
            "random": lambda: RandomSampler(),
        }

        # Add TemperatureSampler if temperature is specified
        if config.temperature is not None:
            all_samplers["temperature"] = lambda: TemperatureSampler(temperature=config.temperature)

        # Add TopKSampler if top_k is specified
        if config.top_k is not None:
            all_samplers["top_k"] = lambda: TopKSampler(k=config.top_k)

        # Create only requested samplers, or all if none specified
        if requested_samplers is None:
            requested_samplers = list(all_samplers.keys())
        
        samplers = {}
        for name in requested_samplers:
            if name in all_samplers:
                samplers[name] = all_samplers[name]()
            else:
                raise ValueError(f"Unknown sampler: {name}. Available: {list(all_samplers.keys())}")

        return samplers 