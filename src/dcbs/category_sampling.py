"""
Category-based sampling implementation with function objects.

This module provides a cleaner implementation of category-based sampling
using the function object pattern for better maintainability.
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

        # GREEDY selection of highest probability token from cluster
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
        cluster_token_indices = clusters[selected_cluster_idx]
        
        # Select token using the token selector
        return self.token_selector.select_token(
            candidate_ids, cluster_token_indices, candidate_probs, filter_tokens
        )


# Default instance with greedy selection for both category and token
greedy_category_sampler = CategorySampler(
    category_selector=GreedyCategorySelector(),
    token_selector=GreedyTokenSelector()
) 