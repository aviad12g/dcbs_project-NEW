"""
DCBS (Deterministic Category Based Sampling) package.

This package provides a unified interface for various token sampling strategies
including greedy, top-p, random, and DCBS sampling methods.
"""

from .cache_manager import (
    CacheConfig,
    CacheMetrics,
    DCBSCacheManager,
    get_cache_manager,
    reset_cache_manager,
)
from .clustering import (
    CandidateSelector,
    FilteredCandidateSelector,
    KMeansClusterer,
    SingleCluster,
    TokenClusterer,
    TopNCandidateSelector,
)
from .optimizations import BatchDCBSProcessor, MemoryEfficientDCBS, OptimizationConfig
from .samplers import (
    DCBSSampler,
    GreedySampler,
    RandomSampler,
    Sampler,
    SamplingContext,
    TopPSampler,
)

__all__ = [
    "Sampler",
    "GreedySampler",
    "TopPSampler",
    "RandomSampler",
    "DCBSSampler",
    "SamplingContext",
    "TokenClusterer",
    "CandidateSelector",
    "KMeansClusterer",
    "SingleCluster",
    "TopNCandidateSelector",
    "FilteredCandidateSelector",
    "CacheConfig",
    "CacheMetrics",
    "DCBSCacheManager",
    "get_cache_manager",
    "reset_cache_manager",
    "OptimizationConfig",
    "BatchDCBSProcessor",
    "MemoryEfficientDCBS",
]

__version__ = "1.0.0"
