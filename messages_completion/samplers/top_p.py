"""Top-p (nucleus) sampling implementation."""

from typing import Dict, Any, List, Optional, Set
import torch

from .base import Sampler


class TopPSampler(Sampler):
    """Top-p (nucleus) sampling implementation."""
    
    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        """
        Initialize top-p sampling.
        
        Args:
            p: Cumulative probability threshold
            temperature: Sampling temperature
        """
        self.p = p
        self.temperature = temperature
    
    def sample_token(
        self,
        logits: torch.Tensor,
        context: Optional[Any] = None,
        filter_tokens: Optional[Set[int]] = None
    ) -> int:
        """Sample token using top-p method."""
        # Apply temperature
        logits = logits / self.temperature
        
        # Apply filter if provided
        if filter_tokens:
            mask = torch.full_like(logits, float('-inf'))
            for token_id in filter_tokens:
                if token_id < len(logits):
                    mask[token_id] = 0
            logits = logits + mask
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sort probabilities
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Find cutoff point
        cutoff_idx = torch.searchsorted(cumulative_probs, self.p).item()
        cutoff_idx = max(1, cutoff_idx)  # Always keep at least one token
        
        # Keep only top-p tokens
        top_p_probs = sorted_probs[:cutoff_idx]
        top_p_indices = sorted_indices[:cutoff_idx]
        
        # Renormalize
        top_p_probs = top_p_probs / top_p_probs.sum()
        
        # Sample from top-p distribution
        sampled_idx = torch.multinomial(top_p_probs, 1).item()
        return top_p_indices[sampled_idx].item()
    
    def sample_batch(
        self,
        logits_batch: torch.Tensor,
        context: Optional[Any] = None,
        filter_tokens_batch: Optional[List[Optional[Set[int]]]] = None
    ) -> List[int]:
        """Sample tokens using top-p from batch."""
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
        return f"top_p_{self.p}"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"p": self.p, "temperature": self.temperature}