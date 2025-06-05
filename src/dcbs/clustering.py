"""
Clustering interfaces and implementations for DCBS.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering

from .constants import (
    KMEANS_RANDOM_SEED,
    KMEANS_MIN_BATCH_SIZE,
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
    """K-means clustering implementation."""

    def __init__(
        self,
        k: int,
        random_seed: int = KMEANS_RANDOM_SEED,
        max_iterations: int = KMEANS_DEFAULT_MAX_ITER,
        min_batch_size: int = KMEANS_MIN_BATCH_SIZE,
    ):
        """
        Initialize K-means clusterer.

        Args:
            k: Number of clusters
            random_seed: Random seed for reproducibility (42 is ML convention)
            max_iterations: Maximum number of k-means iterations
            min_batch_size: Minimum batch size for MiniBatchKMeans (3584 optimized for 11GB GPU memory)
        """
        self.k = k
        self.random_seed = random_seed
        self.max_iterations = max_iterations
        self.min_batch_size = min_batch_size

    def cluster(self, embeddings: torch.Tensor) -> np.ndarray:
        """Cluster embeddings using k-means."""
        embeddings_np = embeddings.detach().cpu().numpy()
        effective_k = min(self.k, len(embeddings_np))

        batch_size = max(self.min_batch_size, len(embeddings_np))

        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            batch_size=batch_size,
            max_iter=self.max_iterations,
            random_state=self.random_seed,
        )

        return kmeans.fit_predict(embeddings_np)

    @property
    def num_clusters(self) -> int:
        return self.k


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
    """Select top-n tokens by probability."""

    def __init__(self, top_n: int = DEFAULT_TOP_N):
        """
        Initialize TopNCandidateSelector.
        
        Args:
            top_n: Number of top tokens to select (50 is optimal based on empirical testing
                  with language models having 32K-50K vocabulary sizes)
        """
        self.top_n = top_n

    def select_candidates(
        self, logits: torch.Tensor, filter_tokens: Optional[set]
    ) -> List[int]:
        """Select top-n candidate tokens."""
        if filter_tokens:
            return list(filter_tokens)
        else:
            sorted_indices = torch.argsort(logits, descending=True)
            return sorted_indices[: self.top_n].cpu().tolist()


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
