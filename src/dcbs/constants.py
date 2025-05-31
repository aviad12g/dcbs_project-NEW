"""
Constants for DCBS implementation.

This module contains all constants and magic numbers used throughout the DCBS implementation
to improve maintainability and clarity.
"""

# Algorithm Parameters
MIN_TOKENS_FOR_CLUSTERING = 3  # Minimum tokens needed for clustering
DEFAULT_K_CLUSTERS = 8  # Default number of clusters for K-means
DEFAULT_TOP_N = 50  # Default number of top tokens to consider

# Cache Configuration
DEFAULT_EMBEDDING_CACHE_SIZE = 1000  # Default size for embedding cache
DEFAULT_CLUSTER_CACHE_SIZE = 200  # Default size for clustering cache

# K-means Parameters
KMEANS_RANDOM_SEED = 42  # Standard seed for reproducibility across ML libraries
KMEANS_MIN_ITERATIONS = 5  # Minimum iterations for convergence, benchmarked for token embeddings
KMEANS_MIN_BATCH_SIZE = 3584  # Optimal batch size determined by GPU memory constraints (7 * 512)

# DBSCAN Parameters
DBSCAN_DEFAULT_EPS = 0.3  # Maximum distance between samples in a cluster
DBSCAN_MIN_SAMPLES = 2  # Minimum samples in a neighborhood to form a cluster
DBSCAN_DEFAULT_METRIC = "cosine"  # Distance metric for token embeddings

# Hierarchical Clustering Parameters
HIERARCHICAL_DEFAULT_LINKAGE = "average"  # Linkage criterion
HIERARCHICAL_DEFAULT_METRIC = "cosine"  # Distance metric

# Environment Variable Names
ENV_DCBS_DEBUG_MODE = "DCBS_DEBUG_MODE"
ENV_DCBS_ENABLE_CLUSTER_HISTORY = "DCBS_ENABLE_CLUSTER_HISTORY"
ENV_DCBS_DEBUG_OUTPUT_FILE = "DCBS_DEBUG_OUTPUT_FILE"

# Debug Configuration
DEBUG_ENV_TRUE_VALUES = ("true", "1", "yes", "on")
DEBUG_ENV_FALSE_VALUES = ("false", "0", "no", "off")

# Probability Constants
PROB_EPSILON = 1e-6  # Small value to prevent division by zero

# Performance Thresholds
CACHING_BENEFIT_THRESHOLD_EXAMPLES = 1000  # Number of examples where caching becomes beneficial
CACHING_OVERHEAD_SMALL_DATASET = 100  # Dataset size below which caching adds overhead 