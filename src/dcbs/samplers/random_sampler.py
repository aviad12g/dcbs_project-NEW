"""
Random sampling implementation.

This module provides uniform random sampling from allowed tokens,
serving as a baseline for comparison with other sampling methods.
"""

import random
from typing import Optional, Set

import torch

from .base import Sampler, SamplingContext


class RandomSampler(Sampler):
    """
    Random sampling - uniformly samples from allowed tokens.
    
    This stochastic sampling method provides maximum diversity by selecting
    tokens uniformly at random, serving as a lower bound for performance.
    """

    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """
        Sample a token uniformly at random.
        
        Args:
            logits: Token logits from the model (used only for vocabulary size)
            filter_tokens: Optional set of allowed token IDs
            context: Optional sampling context (unused for random)
            
        Returns:
            Randomly selected token ID
        """
        if filter_tokens is not None and len(filter_tokens) > 0:
            return random.choice(list(filter_tokens))
        else:
            return random.randint(0, logits.shape[-1] - 1) 