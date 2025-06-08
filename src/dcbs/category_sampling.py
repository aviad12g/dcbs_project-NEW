"""
Category-based sampling implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Set

import numpy as np
import torch


class CategorySelector(ABC):
    """Base class for category selection strategies."""
    
    @abstractmethod
    def select_category(self, cluster_probs: List[float]) -> int:
        """
        Select a category/cluster based on cluster probabilities.
        
        Args:
            cluster_probs: Probabilities for each cluster
            
        Returns:
            Index of the selected cluster
        """
        pass


class GreedyCategorySelector(CategorySelector):
    """Selects category with highest probability (argmax)."""
    
    def select_category(self, cluster_probs: List[float]) -> int:
        """Select category with highest probability."""
        return int(np.argmax(cluster_probs))


class InformationGainCategorySelector(CategorySelector):
    """
    Selects category with highest probability after pruning low-information clusters.
    
    Prunes clusters whose removal would cause minimal information loss (low KL divergence).
    This addresses the "garbage low-prob" error pattern where DCBS picks from clusters
    with very low probability that don't meaningfully contribute to the decision.
    """
    
    def __init__(self, kl_threshold: float = 1e-3, min_clusters: int = 1):
        """
        Initialize the information-gain-guided selector.
        
        Args:
            kl_threshold: Minimum KL divergence contribution to keep a cluster
            min_clusters: Minimum number of clusters to preserve
        """
        self.kl_threshold = kl_threshold
        self.min_clusters = min_clusters
    
    def _compute_kl_contribution(self, cluster_probs: List[float], cluster_idx: int) -> float:
        """
        Compute the KL divergence contribution of removing a specific cluster.
        
        Args:
            cluster_probs: Original cluster probabilities
            cluster_idx: Index of cluster to evaluate for removal
            
        Returns:
            KL divergence between original and pruned distributions
        """
        # Convert to numpy for easier computation
        probs = np.array(cluster_probs)
        total_prob = probs.sum()
        
        if total_prob == 0:
            return 0.0
        
        # Original normalized distribution
        orig_dist = probs / total_prob
        
        # Distribution with cluster removed
        pruned_probs = probs.copy()
        pruned_probs[cluster_idx] = 0
        pruned_total = pruned_probs.sum()
        
        if pruned_total == 0:
            return float('inf')  # Can't remove this cluster
        
        pruned_dist = pruned_probs / pruned_total
        
        # Compute KL divergence: KL(original || pruned)
        kl_div = 0.0
        for i, (p, q) in enumerate(zip(orig_dist, pruned_dist)):
            if p > 1e-10:  # Avoid log(0)
                if q > 1e-10:
                    kl_div += p * np.log(p / q)
                else:
                    kl_div = float('inf')  # Infinite divergence
                    break
        
        return kl_div
    
    def _prune_low_information_clusters(self, cluster_probs: List[float]) -> List[int]:
        """
        Identify clusters to keep based on information gain criterion.
        
        Args:
            cluster_probs: Probabilities for each cluster
            
        Returns:
            List of cluster indices to keep
        """
        n_clusters = len(cluster_probs)
        
        # Always keep minimum number of clusters
        if n_clusters <= self.min_clusters:
            return list(range(n_clusters))
        
        # Compute KL contribution for each cluster
        kl_contributions = []
        for i in range(n_clusters):
            kl_contrib = self._compute_kl_contribution(cluster_probs, i)
            kl_contributions.append((i, kl_contrib))
        
        # Sort by KL contribution (ascending - lowest contribution first)
        kl_contributions.sort(key=lambda x: x[1])
        
        # Keep clusters that exceed the threshold or are in the top min_clusters
        clusters_to_keep = []
        
        # Always keep the top contributing clusters up to min_clusters
        for i in range(min(self.min_clusters, len(kl_contributions))):
            clusters_to_keep.append(kl_contributions[-(i+1)][0])  # Take from end (highest contrib)
        
        # Add additional clusters that exceed the threshold
        for cluster_idx, kl_contrib in kl_contributions:
            if cluster_idx not in clusters_to_keep and kl_contrib >= self.kl_threshold:
                clusters_to_keep.append(cluster_idx)
        
        return sorted(clusters_to_keep)
    
    def select_category(self, cluster_probs: List[float]) -> int:
        """
        Select category with highest probability after information-gain pruning.
        
        Args:
            cluster_probs: Probabilities for each cluster
            
        Returns:
            Index of the selected cluster
        """
        if not cluster_probs:
            return 0
        
        # Prune low-information clusters
        valid_clusters = self._prune_low_information_clusters(cluster_probs)
        
        if not valid_clusters:
            # Fallback to greedy if no clusters survive pruning
            return int(np.argmax(cluster_probs))
        
        # Find the highest probability cluster among the valid ones
        best_cluster_idx = valid_clusters[0]
        best_prob = cluster_probs[best_cluster_idx]
        
        for cluster_idx in valid_clusters[1:]:
            if cluster_probs[cluster_idx] > best_prob:
                best_prob = cluster_probs[cluster_idx]
                best_cluster_idx = cluster_idx
        
        return best_cluster_idx


class TokenSelector(ABC):
    """Base class for token selection strategies within a category."""
    
    @abstractmethod
    def select_token(
        self,
        candidate_ids: List[int],
        cluster_token_indices: List[int],
        candidate_probs: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
    ) -> int:
        """
        Select a token from the chosen category/cluster.
        
        Args:
            candidate_ids: List of candidate token IDs
            cluster_token_indices: Indices of tokens in the selected cluster
            candidate_probs: Probabilities of candidate tokens
            filter_tokens: Optional set of allowed token IDs
            
        Returns:
            Selected token ID
        """
        pass


class GreedyTokenSelector(TokenSelector):
    """Selects token with highest probability within a category."""
    
    def select_token(
        self,
        candidate_ids: List[int],
        cluster_token_indices: List[int],
        candidate_probs: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
    ) -> int:
        """Select highest probability token from the chosen cluster."""
        cluster_token_probs = candidate_probs[cluster_token_indices]

        # Apply filtering within cluster if needed
        if filter_tokens:
            valid_indices = [
                i
                for i, token_idx in enumerate(cluster_token_indices)
                if candidate_ids[token_idx] in filter_tokens
            ]

            if not valid_indices:
                # Fall back to first filtered token if none in cluster
                for token_id in candidate_ids:
                    if token_id in filter_tokens:
                        return token_id
                # If still no valid token, return first in filter set
                return next(iter(filter_tokens))

            cluster_token_indices = [cluster_token_indices[i] for i in valid_indices]
            cluster_token_probs = cluster_token_probs[valid_indices]

        # Select highest probability token from cluster
        selected_in_cluster_idx = torch.argmax(cluster_token_probs).item()
        selected_token_idx = cluster_token_indices[selected_in_cluster_idx]

        return candidate_ids[selected_token_idx]


class CategorySampler:
    """
    Main implementation of category-based sampling.
    
    This class combines category selection and token selection strategies
    to implement deterministic category-based sampling.
    """
    
    def __init__(
        self,
        category_selector: CategorySelector = None,
        token_selector: TokenSelector = None,
    ):
        """
        Initialize with selection strategies.
        
        Args:
            category_selector: Strategy for selecting categories
            token_selector: Strategy for selecting tokens within categories
        """
        self.category_selector = category_selector or GreedyCategorySelector()
        self.token_selector = token_selector or GreedyTokenSelector()
    
    def sample_from_clusters(
        self,
        candidate_ids: List[int],
        candidate_probs: torch.Tensor,
        clusters: List[List[int]],
        filter_tokens: Optional[Set[int]] = None,
    ) -> int:
        """
        Sample token using category-based approach.
        
        Args:
            candidate_ids: List of candidate token IDs
            candidate_probs: Probabilities of candidate tokens
            clusters: List of clusters, each containing indices into candidates
            filter_tokens: Optional set of allowed token IDs
            
        Returns:
            Selected token ID
        """
        # Calculate cluster probabilities
        cluster_probs = [
            candidate_probs[cluster].sum().item() if cluster else 0.0
            for cluster in clusters
        ]
        
        # No valid clusters
        if sum(cluster_probs) == 0:
            # Fall back to first filtered token
            if filter_tokens:
                for token_id in candidate_ids:
                    if token_id in filter_tokens:
                        return token_id
                return next(iter(filter_tokens))
            # Otherwise use highest probability token
            best_idx = torch.argmax(candidate_probs).item()
            return candidate_ids[best_idx]
        
        # Select cluster using the category selector
        selected_cluster_idx = self.category_selector.select_category(cluster_probs)
        
        # Safety check: ensure cluster index is valid
        if selected_cluster_idx >= len(clusters):
            # Fallback to the last available cluster
            selected_cluster_idx = len(clusters) - 1
        
        cluster_token_indices = clusters[selected_cluster_idx]
        
        # Safety check: handle empty clusters
        if not cluster_token_indices:
            # Find first non-empty cluster
            for i, cluster in enumerate(clusters):
                if cluster:
                    cluster_token_indices = cluster
                    break
            else:
                # All clusters are empty, fallback to greedy selection
                if filter_tokens:
                    for token_id in candidate_ids:
                        if token_id in filter_tokens:
                            return token_id
                    return next(iter(filter_tokens))
                best_idx = torch.argmax(candidate_probs).item()
                return candidate_ids[best_idx]
        
        # Select token using the token selector
        return self.token_selector.select_token(
            candidate_ids, cluster_token_indices, candidate_probs, filter_tokens
        )


# Default instance with greedy selection for both category and token
greedy_category_sampler = CategorySampler(
    category_selector=GreedyCategorySelector(),
    token_selector=GreedyTokenSelector()
)

# Enhanced instance with information-gain pruning to avoid "garbage low-prob" selections
information_gain_category_sampler = CategorySampler(
    category_selector=InformationGainCategorySelector(kl_threshold=0.1, min_clusters=1),
    token_selector=GreedyTokenSelector()
) 