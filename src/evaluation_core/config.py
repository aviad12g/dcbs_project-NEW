"""
Configuration classes for evaluation runs.

This module defines the configuration dataclass used to specify
evaluation parameters and settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""

    model_name: str
    benchmark_path: str
    output_dir: str = "results"
    limit: Optional[int] = None
    top_p: float = 0.9
    k: int = 8
    top_n: int = 50
    include_cot: bool = True
    log_level: str = "INFO"
    load_in_4bit: bool = False
    enable_caching: bool = True  # Control DCBS caching
    clustering_method: str = "kmeans"  # Clustering method for DCBS (default to K-means)
    dbscan_eps: float = 0.3  # DBSCAN epsilon parameter
    dbscan_min_samples: int = 2  # DBSCAN minimum samples
    hierarchical_linkage: str = "average"  # Hierarchical clustering linkage
    debug_mode: bool = False  # Enable verbose debug logging
    enable_cluster_history: bool = True  # Track cluster assignments
    temperature: Optional[float] = None # Temperature for sampling
    top_k: Optional[int] = None # Top-K value for sampling
    batch_size: Optional[int] = None  # Batch size for GPU processing (None = auto-detect)
    use_elbow_method: bool = False  # Use elbow method for dynamic k-means k selection
    # Parallelism control for clustering workers (None = auto)
    max_cluster_workers: Optional[int] = None
    
    # Cluster weighting strategy for category selection
    weighting_strategy: str = "prob_mass"  # Options: prob_mass, size, sqrt_size, uniform

    # Clustering weighting mode (how to weight the clustering step itself)
    # none: unweighted clustering (current default behaviour)
    # prob: weight samples by their token probabilities during clustering
    cluster_weighting: str = "none"  # Options: none, prob, uniform

    # Batched clustering controls
    batched_clustering: str = "auto"  # auto | on | off
    kmeans_iters: Optional[int] = None  # Lloyd iterations override for batched KMeans

    # Multi-dataset evaluation
    datasets: list = None  # List of datasets to evaluate
    
    # Disagreement tracking
    enable_disagreement_tracking: bool = False  # Enable disagreement tracking
    run_id: Optional[str] = None  # Run identifier for disagreement logs