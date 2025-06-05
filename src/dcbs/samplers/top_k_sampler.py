"""
Top-K sampling implementation.
"""

from typing import Optional, Set

import torch

from .base import Sampler, SamplingContext


class TopKSampler(Sampler):
    """
    Top-K sampling strategy.

    Selects from the top-k most probable tokens.
    """

    def __init__(self, k: int):
        """
        Initialize the TopKSampler.

        Args:
            k: The number of top tokens to consider (must be a positive integer).
        """
        if not isinstance(k, int) or k <= 0:
            raise ValueError("Top-K (k) must be a positive integer.")
        self.k = k

    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """
        Sample a token from the top-k logits.

        Args:
            logits: Token logits from the model (shape: [vocab_size]).
            filter_tokens: Optional set of allowed token IDs. If provided,
                           only these tokens will be considered for sampling
                           (and only from the top-k filtered set).
            context: Not used by this sampler but part of the interface.

        Returns:
            Selected token ID.
        """
        # Get top-k logits and their indices
        effective_k = min(self.k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, effective_k)

        if filter_tokens:
            # More efficient filtering using tensor operations
            filter_tensor = torch.tensor(list(filter_tokens), device=logits.device, dtype=torch.long)
            
            # Create mask for which top-k tokens are in filter_tokens
            # Use broadcasting to compare top_k_indices with filter_tokens
            is_allowed = torch.isin(top_k_indices, filter_tensor)
            
            if is_allowed.any():
                # Apply mask to logits
                filtered_logits = torch.full_like(top_k_logits, -float('inf'))
                filtered_logits[is_allowed] = top_k_logits[is_allowed]
                relevant_logits = filtered_logits
            else:
                # No overlap between top-k and filter_tokens, fallback to first valid token
                valid_tokens = filter_tensor[filter_tensor < logits.size(-1)]
                if len(valid_tokens) > 0:
                    return valid_tokens[0].item()
                else:
                    # Fallback to greedy if no valid tokens
                    return torch.argmax(logits).item()
            
            relevant_indices = top_k_indices
        else:
            relevant_logits = top_k_logits
            relevant_indices = top_k_indices

        # Apply softmax and sample (multinomial handles normalization)
        probabilities = torch.softmax(relevant_logits, dim=-1)
        sampled_idx = torch.multinomial(probabilities, num_samples=1).item()

        return relevant_indices[sampled_idx].item() 