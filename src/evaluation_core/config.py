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
    output_dir: str
    limit: Optional[int] = None
    top_p: float = 0.9
    k: int = 8
    top_n: int = 50
    include_cot: bool = True
    log_level: str = "INFO"
    load_in_4bit: bool = False
    enable_caching: bool = True  # Control DCBS caching
    clustering_method: str = "dbscan"  # Clustering method for DCBS (default to DBSCAN)
    dbscan_eps: float = 0.3  # DBSCAN epsilon parameter
    dbscan_min_samples: int = 2  # DBSCAN minimum samples
    hierarchical_linkage: str = "average"  # Hierarchical clustering linkage
    debug_mode: bool = False  # Enable verbose debug logging
    enable_cluster_history: bool = True  # Track cluster assignments
    temperature: Optional[float] = None # Temperature for sampling
    top_k: Optional[int] = None # Top-K value for sampling
    batch_size: Optional[int] = None  # Batch size for GPU processing (None = auto-detect)

    # Multi-dataset evaluation
    datasets: list = None  # List of datasets to evaluate
    
    # Disagreement tracking
    enable_disagreement_tracking: bool = False  # Enable disagreement tracking
    run_id: Optional[str] = None  # Run identifier for disagreement logs