"""Greedy sampling implementation."""

from typing import Dict, Any, List, Optional, Set
import torch

from .base import Sampler


class GreedySampler(Sampler):
    """Greedy sampling - always selects the highest probability token."""
    
    def sample_token(
        self,
        logits: torch.Tensor,
        context: Optional[Any] = None,
        filter_tokens: Optional[Set[int]] = None
    ) -> int:
        """Sample token using greedy method."""
        if filter_tokens:
            # Create mask for filtered tokens
            mask = torch.full_like(logits, float('-inf'))
            for token_id in filter_tokens:
                if token_id < len(logits):
                    mask[token_id] = 0
            masked_logits = logits + mask
            return torch.argmax(masked_logits).item()
        else:
            return torch.argmax(logits).item()
    
    def sample_batch(
        self,
        logits_batch: torch.Tensor,
        context: Optional[Any] = None,
        filter_tokens_batch: Optional[List[Optional[Set[int]]]] = None
    ) -> List[int]:
        """Sample tokens greedily from batch."""
        batch_size = logits_batch.shape[0]
        results = []
        
        for i in range(batch_size):
            logits = logits_batch[i]
            filter_tokens = filter_tokens_batch[i] if filter_tokens_batch else None
            token_id = self.sample_token(logits, context, filter_tokens)
            results.append(token_id)
        
        return results
    
    @property
    def method_name(self) -> str:
        return "greedy"