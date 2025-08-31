"""Base sampling interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
import torch


class Sampler(ABC):
    """Abstract base class for sampling methods."""
    
    @abstractmethod
    def sample_token(
        self,
        logits: torch.Tensor,
        context: Optional[Any] = None,
        filter_tokens: Optional[Set[int]] = None
    ) -> int:
        """Sample a single token from logits."""
        pass
    
    @abstractmethod
    def sample_batch(
        self,
        logits_batch: torch.Tensor,
        context: Optional[Any] = None,
        filter_tokens_batch: Optional[List[Optional[Set[int]]]] = None
    ) -> List[int]:
        """Sample tokens from a batch of logits."""
        pass
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of this sampling method."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return the parameters used by this sampler."""
        return {}