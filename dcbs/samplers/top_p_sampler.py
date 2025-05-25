"""
Top-p (nucleus) sampling implementation.

This module provides top-p sampling which samples from the smallest set of tokens
whose cumulative probability mass is greater than or equal to p.
"""

from typing import Optional, Set

import torch

from .base import Sampler, SamplingContext


class TopPSampler(Sampler):
    """
    Top-p (nucleus) sampling - samples from top tokens until cumulative probability >= p.
    
    This stochastic sampling method balances quality and diversity by sampling from
    a dynamically sized set of high-probability tokens.
    """

    def __init__(self, p: float = 0.9):
        """
        Initialize the Top-p sampler.
        
        Args:
            p: Cumulative probability threshold (default: 0.9)
        """
        self.p = p

    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """
        Sample a token using top-p (nucleus) sampling.
        
        Args:
            logits: Token logits from the model
            filter_tokens: Optional set of allowed token IDs
            context: Optional sampling context (unused for top-p)
            
        Returns:
            Sampled token ID
        """
        # Apply filtering first if provided
        if filter_tokens is not None and len(filter_tokens) > 0:
            filtered_logits = torch.full_like(logits, float("-inf"))
            allowed_indices = list(filter_tokens)
            filtered_logits[allowed_indices] = logits[allowed_indices]
            logits = filtered_logits

        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > self.p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        filtered_logits = logits.clone()
        filtered_logits[indices_to_remove] = float("-inf")

        # Sample from filtered distribution
        token_id = torch.multinomial(torch.softmax(filtered_logits, dim=-1), 1).item()
        return token_id 