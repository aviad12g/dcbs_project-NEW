"""
Base sampler interface and shared components.

This module defines the abstract Sampler interface and SamplingContext
that all sampling implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Set

import torch

# Small value to prevent division by zero
PROB_EPSILON = 1e-6


@dataclass
class SamplingContext:
    """Context object containing model-specific information for sampling."""

    embedding_layer: Optional[torch.nn.Embedding] = None
    tokenizer: Optional[object] = None
    device: Optional[torch.device] = None


class Sampler(ABC):
    """Abstract base class for token sampling strategies."""

    @abstractmethod
    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """
        Sample a token from the given logits.
        
        Args:
            logits: Token logits from the model
            filter_tokens: Optional set of allowed token IDs
            context: Optional sampling context with model information
            
        Returns:
            Selected token ID
        """
        pass 