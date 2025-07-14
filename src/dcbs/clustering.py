"""
Clustering interfaces and implementations for DCBS.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from .constants import (
    KMEANS_RANDOM_SEED,
    KMEANS_MIN_BATCH_SIZE,
    KMEANS_DEFAULT_METRIC,
    DBSCAN_DEFAULT_EPS,
    DBSCAN_MIN_SAMPLES,
    DBSCAN_DEFAULT_METRIC,
    HIERARCHICAL_DEFAULT_LINKAGE,
    HIERARCHICAL_DEFAULT_METRIC,
    DEFAULT_TOP_N,
    KMEANS_DEFAULT_MAX_ITER,
)


class TokenClusterer(ABC):
    """Interface for clustering token embeddings."""

    @abstractmethod
    def cluster(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Cluster token embeddings.

        Args:
            embeddings: Normalized token embeddings of shape (n_tokens, embedding_dim)

        Returns:
            Cluster labels array of shape (n_tokens,)
        """
        pass

    @property
    @abstractmethod
    def num_clusters(self) -> int:
        """Return the number of clusters this clusterer produces."""
        pass


class KMeansClusterer(TokenClusterer):
    """K-means clustering implementation with adaptive k selection."""

    def __init__(
        self,
        k: int,
        random_seed: int = KMEANS_RANDOM_SEED,
        max_iterations: int = KMEANS_DEFAULT_MAX_ITER,
        min_batch_size: int = KMEANS_MIN_BATCH_SIZE,
        enable_adaptive_k: bool = True,
        min_k: int = 2,
        max_k: int = 16,
        use_elbow_method: bool = False,
    ):
        """
        Initialize K-means clusterer with adaptive k selection.

        Args:
            k: Base number of clusters (used as fallback or max when adaptive disabled)
            random_seed: Random seed for reproducibility (42 is ML convention)
            max_iterations: Maximum number of k-means iterations
            min_batch_size: Minimum batch size for MiniBatchKMeans (3584 optimized for 11GB GPU memory)
            enable_adaptive_k: Whether to adapt k based on candidate set size
            min_k: Minimum number of clusters to use
            max_k: Maximum number of clusters to use
            use_elbow_method: Whether to use elbow method for k selection (more accurate but slower)
        """
        self.k = k
        self.random_seed = random_seed
        self.max_iterations = max_iterations
        self.min_batch_size = min_batch_size
        self.enable_adaptive_k = enable_adaptive_k
        self.min_k = min_k
        self.max_k = max_k
        self.use_elbow_method = use_elbow_method
        self._last_effective_k = k  # Track the actual k used in last clustering

    def _calculate_elbow_method_k(self, embeddings_np: np.ndarray, max_k: int = None) -> int:
        """Calculate optimal k using elbow method with Within-Cluster Sum of Squares (WCSS)."""
        if max_k is None:
            max_k = min(self.max_k, len(embeddings_np) // 2)
        
        if max_k < 2:
            return 2
        
        # Calculate WCSS for different k values
        wcss_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            # Use simple k-means for elbow calculation
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=self.random_seed, n_init=1)
            kmeans.fit(embeddings_np)
            wcss_scores.append(kmeans.inertia_)
        
        # Find elbow point using second derivative
        if len(wcss_scores) < 3:
            return 2
        
        # Calculate second differences to find elbow
        second_diffs = []
        for i in range(1, len(wcss_scores) - 1):
            second_diff = wcss_scores[i-1] - 2*wcss_scores[i] + wcss_scores[i+1]
            second_diffs.append(second_diff)
        
        # Find the k with maximum second difference (sharpest bend)
        elbow_idx = np.argmax(second_diffs)
        elbow_k = k_range[elbow_idx + 1]  # +1 because second_diffs is offset
        
        return elbow_k

    def _calculate_adaptive_k(self, n_candidates: int, embeddings_np: np.ndarray = None) -> int:
        """Calculate optimal k based on candidate set size and optionally elbow method."""
        if not self.enable_adaptive_k:
            return self.k
        
        # If elbow method is enabled and we have embeddings, use it
        if embeddings_np is not None and hasattr(self, 'use_elbow_method') and self.use_elbow_method:
            elbow_k = self._calculate_elbow_method_k(embeddings_np)
            # Combine elbow method with heuristic bounds
            return max(self.min_k, min(self.max_k, elbow_k))
        
        # Original square root heuristic with bounds
        # For small sets: k = 2
        # For medium sets: k = sqrt(n)  
        # For large sets: k = min(max_k, n//3)
        if n_candidates <= 4:
            adaptive_k = 2
        elif n_candidates <= 50:
            adaptive_k = max(self.min_k, int(np.sqrt(n_candidates)))
        else:
            adaptive_k = min(self.max_k, n_candidates // 3)
        
        # Ensure k doesn't exceed configured maximum
        return min(adaptive_k, self.k)

    def cluster(self, embeddings: torch.Tensor) -> np.ndarray:
        """Cluster embeddings using k-means with cosine similarity and adaptive k selection."""
        embeddings_np = embeddings.detach().cpu().numpy()
        n_candidates = len(embeddings_np)
        
        # Calculate adaptive k (pass embeddings for elbow method if enabled)
        adaptive_k = self._calculate_adaptive_k(n_candidates, embeddings_np if self.use_elbow_method else None)
        effective_k = min(adaptive_k, n_candidates)
        
        # Store for debugging/metrics
        self._last_effective_k = effective_k

        # Ensure embeddings are L2 normalized for cosine similarity
        # (already normalized in embedding_ops, but ensure here)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        embeddings_normalized = embeddings_np / norms

        batch_size = max(self.min_batch_size, len(embeddings_np))

        # Use spherical k-means approach for cosine similarity
        # Initialize centroids
        np.random.seed(self.random_seed)
        centroid_indices = np.random.choice(n_candidates, effective_k, replace=False)
        centroids = embeddings_normalized[centroid_indices].copy()
        
        # Iterative clustering with cosine similarity
        labels = np.zeros(n_candidates, dtype=int)
        
        for iteration in range(self.max_iterations):
            # Assign points to closest centroid using cosine similarity
            # Cosine similarity = 1 - cosine_distance
            distances = cosine_distances(embeddings_normalized, centroids)
            new_labels = np.argmin(distances, axis=1)
            
            # Check for convergence
            if np.array_equal(labels, new_labels):
                break
                
            labels = new_labels
            
            # Update centroids (mean of assigned points, then normalize)
            for k in range(effective_k):
                cluster_mask = labels == k
                if np.any(cluster_mask):
                    centroid = np.mean(embeddings_normalized[cluster_mask], axis=0)
                    # Normalize centroid
                    norm = np.linalg.norm(centroid)
                    if norm > 0:
                        centroids[k] = centroid / norm
                    else:
                        # If norm is 0, keep previous centroid
                        pass
        
        return labels

    @property
    def num_clusters(self) -> int:
        """Return the last effective k used (accounts for adaptive selection)."""
        return self._last_effective_k
    
    def get_adaptive_info(self) -> dict:
        """Get information about adaptive k selection for debugging."""
        return {
            "base_k": self.k,
            "last_effective_k": self._last_effective_k,
            "adaptive_enabled": self.enable_adaptive_k,
            "elbow_method_enabled": self.use_elbow_method,
            "min_k": self.min_k,
            "max_k": self.max_k,
        }


class DBSCANClusterer(TokenClusterer):
    """DBSCAN clustering implementation for token embeddings."""

    def __init__(
        self,
        eps: float = DBSCAN_DEFAULT_EPS,
        min_samples: int = DBSCAN_MIN_SAMPLES,
        metric: str = DBSCAN_DEFAULT_METRIC,
        n_jobs: int = 1,
    ):
        """
        Initialize DBSCAN clusterer.

        Args:
            eps: Maximum distance between two samples for them to be in the same neighborhood
            min_samples: Minimum number of samples in a neighborhood for a core point
            metric: Distance metric (cosine recommended for normalized embeddings)
            n_jobs: Number of parallel jobs (-1 uses all processors)
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs
        self._last_n_clusters = 1  # Track actual number of clusters found

    def cluster(self, embeddings: torch.Tensor) -> np.ndarray:
        """Cluster embeddings using DBSCAN."""
        embeddings_np = embeddings.detach().cpu().numpy()
        
        # DBSCAN with cosine distance
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        
        labels = dbscan.fit_predict(embeddings_np)
        
        # DBSCAN may produce -1 for noise points, map them to their own cluster
        unique_labels = np.unique(labels)
        if -1 in unique_labels:
            # Map noise points to separate clusters
            noise_mask = labels == -1
            max_label = labels.max()
            noise_indices = np.where(noise_mask)[0]
            for i, idx in enumerate(noise_indices):
                labels[idx] = max_label + 1 + i
        
        # Update tracked cluster count
        self._last_n_clusters = len(np.unique(labels))
        
        return labels

    @property
    def num_clusters(self) -> int:
        """Return the last observed number of clusters (DBSCAN finds this dynamically)."""
        return self._last_n_clusters


class HierarchicalClusterer(TokenClusterer):
    """Hierarchical (Agglomerative) clustering implementation for token embeddings."""

    def __init__(
        self,
        k: int,
        linkage: str = HIERARCHICAL_DEFAULT_LINKAGE,
        metric: str = HIERARCHICAL_DEFAULT_METRIC,
    ):
        """
        Initialize Hierarchical clusterer.

        Args:
            k: Number of clusters to find
            linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
            metric: Distance metric (cosine recommended for embeddings)
        """
        self.k = k
        self.linkage = linkage
        self.metric = metric

    def cluster(self, embeddings: torch.Tensor) -> np.ndarray:
        """Cluster embeddings using hierarchical clustering."""
        embeddings_np = embeddings.detach().cpu().numpy()
        effective_k = min(self.k, len(embeddings_np))
        
        # Use 'euclidean' metric if linkage is 'ward'
        metric = 'euclidean' if self.linkage == 'ward' else self.metric
        
        clustering = AgglomerativeClustering(
            n_clusters=effective_k,
            linkage=self.linkage,
            metric=metric,
        )
        
        return clustering.fit_predict(embeddings_np)

    @property
    def num_clusters(self) -> int:
        return self.k


class SingleCluster(TokenClusterer):
    """Trivial clusterer that puts all tokens in one cluster (equivalent to greedy)."""

    def cluster(self, embeddings: torch.Tensor) -> np.ndarray:
        """Put all tokens in cluster 0."""
        return np.zeros(embeddings.shape[0], dtype=int)

    @property
    def num_clusters(self) -> int:
        return 1


class CandidateSelector(ABC):
    """Interface for selecting candidate tokens."""

    @abstractmethod
    def select_candidates(
        self, logits: torch.Tensor, filter_tokens: Optional[set]
    ) -> List[int]:
        """
        Select candidate tokens for clustering.

        Args:
            logits: Token logits
            filter_tokens: Optional set of allowed token IDs

        Returns:
            List of candidate token IDs
        """
        pass


class TopNCandidateSelector(CandidateSelector):
    """Select top-n tokens by probability with dynamic adaptation."""

    def __init__(self, top_n: int = DEFAULT_TOP_N, enable_dynamic: bool = True, 
                 min_candidates: int = 20, max_candidates: int = 100, 
                 prob_threshold: float = 0.001):
        """
        Initialize TopNCandidateSelector with dynamic adaptation.
        
        Args:
            top_n: Base number of top tokens to select (used as fallback)
            enable_dynamic: Whether to use dynamic selection based on probability distribution
            min_candidates: Minimum number of candidates to select
            max_candidates: Maximum number of candidates to select  
            prob_threshold: Probability threshold below which tokens are excluded
        """
        self.top_n = top_n
        self.enable_dynamic = enable_dynamic
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.prob_threshold = prob_threshold

    def select_candidates(
        self, logits: torch.Tensor, filter_tokens: Optional[set]
    ) -> List[int]:
        """Select candidate tokens with dynamic top-n adaptation."""
        if filter_tokens:
            return list(filter_tokens)
        
        if not self.enable_dynamic:
            # Fall back to original fixed behavior
            sorted_indices = torch.argsort(logits, descending=True)
            return sorted_indices[: self.top_n].cpu().tolist()
        
        # Dynamic selection based on probability distribution
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # Find dynamic cutoff: stop when probability drops below threshold
        above_threshold = (sorted_probs >= self.prob_threshold).sum().item()
        
        # Apply bounds: at least min_candidates, at most max_candidates
        dynamic_n = max(self.min_candidates, min(self.max_candidates, above_threshold))
        
        # Fallback to configured top_n if dynamic calculation seems unreasonable
        if dynamic_n < self.min_candidates or dynamic_n > len(sorted_indices):
            dynamic_n = min(self.top_n, len(sorted_indices))
        
        return sorted_indices[:dynamic_n].cpu().tolist()


class FilteredCandidateSelector(CandidateSelector):
    """Select only from filtered tokens."""

    def select_candidates(
        self, logits: torch.Tensor, filter_tokens: Optional[set]
    ) -> List[int]:
        """Select from filtered tokens only."""
        if filter_tokens:
            return list(filter_tokens)
        else:
            # If no filter provided, use all tokens (not recommended for large vocabularies)
            return list(range(logits.shape[0]))
