"""
Greedy sampling implementation.

This module provides greedy sampling which always selects the highest 
probability token (argmax operation).
"""

from typing import Optional, Set

import torch

from .base import Sampler, SamplingContext


class GreedySampler(Sampler):
    """
    Greedy sampling - always selects the highest probability token.
    
    This is a deterministic sampling method that provides the most likely
    token according to the model's probability distribution.
    """

    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """
        Sample the highest probability token from logits.
        
        Args:
            logits: Token logits from the model
            filter_tokens: Optional set of allowed token IDs
            context: Optional sampling context (unused for greedy)
            
        Returns:
            Token ID with highest probability
        """
        if filter_tokens is not None and len(filter_tokens) > 0:
            # Create mask for allowed tokens (more intuitive logic)
            allowed_mask = torch.full_like(logits, float("-inf"))
            allowed_indices = list(filter_tokens)
            allowed_mask[allowed_indices] = logits[allowed_indices]
            return allowed_mask.argmax().item()

        return logits.argmax().item() 