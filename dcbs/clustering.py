"""
Clustering interfaces and implementations for DCBS.

This module provides clustering abstractions following the function object pattern,
making it easy to swap different clustering algorithms without changing the DCBS logic.
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional
from sklearn.cluster import MiniBatchKMeans


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
    
    def __init__(self, k: int, random_seed: int = 42, min_iterations: int = 5, min_batch_size: int = 3584):
        """
        Initialize K-means clusterer.
        
        Args:
            k: Number of clusters
            random_seed: Random seed for reproducibility  
            min_iterations: Minimum number of k-means iterations
            min_batch_size: Minimum batch size for MiniBatchKMeans
        """
        self.k = k
        self.random_seed = random_seed
        self.min_iterations = min_iterations
        self.min_batch_size = min_batch_size
    
    def cluster(self, embeddings: torch.Tensor) -> np.ndarray:
        """Cluster embeddings using k-means."""
        embeddings_np = embeddings.detach().cpu().numpy()
        effective_k = min(self.k, len(embeddings_np))
        
        batch_size = max(self.min_batch_size, len(embeddings_np))
        
        kmeans = MiniBatchKMeans(
            n_clusters=effective_k,
            batch_size=batch_size,
            max_iter=self.min_iterations,
            random_state=self.random_seed,
        )
        
        return kmeans.fit_predict(embeddings_np)
    
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
    def select_candidates(self, logits: torch.Tensor, filter_tokens: Optional[set]) -> List[int]:
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
    
    def __init__(self, top_n: int = 50):
        self.top_n = top_n
    
    def select_candidates(self, logits: torch.Tensor, filter_tokens: Optional[set]) -> List[int]:
        """Select top-n candidate tokens."""
        if filter_tokens:
            return list(filter_tokens)
        else:
            sorted_indices = torch.argsort(logits, descending=True)
            return sorted_indices[:self.top_n].cpu().tolist()


class FilteredCandidateSelector(CandidateSelector):
    """Select only from filtered tokens."""
    
    def select_candidates(self, logits: torch.Tensor, filter_tokens: Optional[set]) -> List[int]:
        """Select from filtered tokens only."""
        if filter_tokens:
            return list(filter_tokens)
        else:
            # If no filter provided, use all tokens (not recommended for large vocabularies)
            return list(range(logits.shape[0])) 