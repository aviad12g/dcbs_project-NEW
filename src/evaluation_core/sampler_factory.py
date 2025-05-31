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
)
from .config import EvaluationConfig


class SamplerFactory:
    """Factory for creating and managing sampler instances."""

    @staticmethod
    def create_dcbs_sampler(
        config: EvaluationConfig, 
        context: Optional[SamplingContext] = None,
        clustering_method: str = "kmeans",
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 2,
        hierarchical_linkage: str = "average"
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
        )

    @staticmethod
    def create_samplers(
        config: EvaluationConfig, 
        context: Optional[SamplingContext] = None,
        clustering_method: str = "kmeans",
        dbscan_eps: float = 0.3,
        dbscan_min_samples: int = 2,
        hierarchical_linkage: str = "average"
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
            
        Returns:
            Dictionary mapping sampler names to sampler instances
        """
        return {
            "greedy": GreedySampler(),
            "top_p": TopPSampler(p=config.top_p),
            "dcbs": SamplerFactory.create_dcbs_sampler(
                config, 
                context, 
                clustering_method,
                dbscan_eps,
                dbscan_min_samples,
                hierarchical_linkage
            ),
            "random": RandomSampler(),
        } 