"""
DCBS - Deterministic Category Based Sampling

A novel token sampling method for language models that combines deterministic selection
with semantic clustering of token embeddings.
"""

from .cache_manager import CacheConfig, DCBSCacheManager, get_cache_manager
from .category_sampling import (
    CategorySampler,
    CategorySelector,
    TokenSelector,
    GreedyCategorySelector,
    GreedyTokenSelector,
    greedy_category_sampler,
)
from .clustering import (
    CandidateSelector,
    DBSCANClusterer,
    FilteredCandidateSelector,
    HierarchicalClusterer,
    KMeansClusterer,
    SingleCluster,
    TokenClusterer,
    TopNCandidateSelector,
)
from .constants import (
    DEFAULT_K_CLUSTERS,
    DEFAULT_TOP_N,
    MIN_TOKENS_FOR_CLUSTERING,
    PROB_EPSILON,
)
from .debug import DCBSDebugger
from .embedding_ops import EmbeddingOperations
from .optimizations import (
    OptimizationConfig,
    BatchDCBSProcessor,
    MemoryEfficientDCBS,
)
from .samplers import (
    DCBSSampler,
    GreedySampler,
    RandomSampler,
    Sampler,
    SamplingContext,
    TopPSampler,
)
from .factory import DCBSSamplerFactory

__version__ = "0.1.0"

__all__ = [
    # Main sampler classes
    "DCBSSampler",
    "GreedySampler",
    "TopPSampler",
    "RandomSampler",
    # Factory classes
    "DCBSSamplerFactory",
    # Base classes and interfaces
    "Sampler",
    "SamplingContext",
    "TokenClusterer",
    "CandidateSelector",
    "CategorySampler",
    "CategorySelector",
    "TokenSelector",
    # Clustering implementations
    "KMeansClusterer",
    "DBSCANClusterer",
    "HierarchicalClusterer",
    "SingleCluster",
    # Candidate selection implementations
    "TopNCandidateSelector",
    "FilteredCandidateSelector",
    # Category sampling implementations
    "GreedyCategorySelector",
    "GreedyTokenSelector",
    "greedy_category_sampler",
    # Cache management
    "CacheConfig",
    "DCBSCacheManager",
    "get_cache_manager",
    # Debug utilities
    "DCBSDebugger",
    # Embedding operations
    "EmbeddingOperations",
    # Optimizations
    "OptimizationConfig",
    "BatchDCBSProcessor",
    "MemoryEfficientDCBS",
    # Constants
    "DEFAULT_K_CLUSTERS",
    "DEFAULT_TOP_N",
    "MIN_TOKENS_FOR_CLUSTERING",
    "PROB_EPSILON",
]
