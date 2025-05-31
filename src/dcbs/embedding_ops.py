"""
Embedding operations for DCBS.

This module provides functionality for working with token embeddings,
including normalization, caching, and retrieval.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch

from .cache_manager import DCBSCacheManager
from .constants import PROB_EPSILON


class EmbeddingOperations:
    """Handles embedding-related operations for DCBS."""
    
    def __init__(self, cache_manager: Optional[DCBSCacheManager] = None):
        """
        Initialize embedding operations.
        
        Args:
            cache_manager: Optional cache manager for embedding caching
        """
        self.cache_manager = cache_manager
        self.enable_caching = cache_manager is not None
    
    def normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Normalize embeddings to unit vectors.
        
        Args:
            embeddings: Token embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        norm = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        return embeddings / norm.clamp(min=PROB_EPSILON)
    
    def get_embeddings(
        self,
        token_ids: torch.Tensor,
        embedding_layer: torch.nn.Embedding,
    ) -> torch.Tensor:
        """
        Get token embeddings with optional caching.
        
        Args:
            token_ids: Token IDs to get embeddings for
            embedding_layer: Embedding layer from the model
            
        Returns:
            Token embeddings
        """
        if not self.enable_caching:
            # Direct embedding lookup without caching
            with torch.no_grad():
                return embedding_layer(token_ids)
        
        device = token_ids.device
        token_ids_list = token_ids.cpu().tolist()

        # Use optimized batch retrieval
        result, uncached_indices = self.cache_manager.get_batch_embeddings(
            token_ids_list, embedding_layer, device
        )

        # Fetch uncached embeddings
        if uncached_indices:
            self._fetch_uncached_embeddings(
                token_ids_list, uncached_indices, embedding_layer, device, result
            )

        return result
    
    def _fetch_uncached_embeddings(
        self,
        token_ids_list: List[int],
        uncached_indices: List[int],
        embedding_layer: torch.nn.Embedding,
        device: torch.device,
        result: torch.Tensor,
    ) -> None:
        """
        Fetch embeddings that are not in cache and update cache.
        
        Args:
            token_ids_list: List of all token IDs
            uncached_indices: Indices of tokens not in cache
            embedding_layer: Embedding layer from the model
            device: Device to place embeddings on
            result: Result tensor to update with fetched embeddings
        """
        uncached_ids = [token_ids_list[i] for i in uncached_indices]

        # Bounds checking
        max_token_id = max(uncached_ids)
        if embedding_layer.weight.shape[0] > max_token_id:
            with torch.no_grad():
                uncached_embeds = embedding_layer(
                    torch.tensor(uncached_ids, device=device)
                )

                # Update result tensor and cache
                for i, batch_idx in enumerate(uncached_indices):
                    result[batch_idx] = uncached_embeds[i]

                # Cache the new embeddings
                self.cache_manager.cache_batch_embeddings(
                    token_ids_list, uncached_indices, uncached_embeds
                )
    
    def get_normalized_embeddings(
        self,
        token_ids: torch.Tensor,
        embedding_layer: torch.nn.Embedding,
    ) -> torch.Tensor:
        """
        Get normalized token embeddings.
        
        Args:
            token_ids: Token IDs to get embeddings for
            embedding_layer: Embedding layer from the model
            
        Returns:
            Normalized token embeddings
        """
        embeddings = self.get_embeddings(token_ids, embedding_layer)
        return self.normalize_embeddings(embeddings) 