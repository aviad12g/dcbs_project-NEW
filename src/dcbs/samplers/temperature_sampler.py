"""
Temperature sampling implementation.
"""

from typing import Optional, Set

import torch

from .base import Sampler, SamplingContext


class TemperatureSampler(Sampler):
    """
    Temperature sampling strategy.

    Divides logits by a given temperature value before applying softmax.
    Higher temperatures result in more diverse (random) samples,
    while lower temperatures make the distribution sharper, approaching greedy.
    A temperature of 1.0 is equivalent to standard softmax sampling.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Initialize the TemperatureSampler.

        Args:
            temperature: The temperature value to apply to logits (default: 1.0).
                         Must be a positive float.
        """
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError("Temperature must be a positive float.")
        self.temperature = float(temperature)

    def sample(
        self,
        logits: torch.Tensor,
        filter_tokens: Optional[Set[int]] = None,
        context: Optional[SamplingContext] = None,
    ) -> int:
        """
        Sample a token from the given logits after applying temperature scaling.

        Args:
            logits: Token logits from the model (shape: [vocab_size]).
            filter_tokens: Optional set of allowed token IDs. If provided,
                           only these tokens will be considered for sampling.
            context: Not used by this sampler but part of the interface.

        Returns:
            Selected token ID.
        """
        # Work with a copy to avoid modifying the original logits
        working_logits = logits
        
        if filter_tokens:
            # More efficient filtering using advanced indexing
            working_logits = logits.clone()
            # Create boolean mask for allowed tokens
            allowed_mask = torch.zeros_like(logits, dtype=torch.bool)
            # Convert filter_tokens to tensor for efficient indexing
            filter_indices = torch.tensor(list(filter_tokens), device=logits.device, dtype=torch.long)
            # Ensure indices are within bounds
            valid_indices = filter_indices[filter_indices < logits.size(-1)]
            if len(valid_indices) > 0:
                allowed_mask[valid_indices] = True
                working_logits[~allowed_mask] = -float('inf')
            else:
                # No valid tokens in filter_tokens, fallback to greedy selection
                return torch.argmax(logits).item()

        # Apply temperature scaling
        scaled_logits = working_logits / self.temperature

        # Apply softmax to get probabilities
        probabilities = torch.softmax(scaled_logits, dim=-1)
        
        # Check for invalid probabilities (all -inf case)
        if torch.isnan(probabilities).any() or torch.isinf(probabilities).any():
            # Fallback to greedy selection from original logits
            if filter_tokens and len(filter_tokens) > 0:
                valid_indices = [i for i in filter_tokens if i < len(logits)]
                if valid_indices:
                    return max(valid_indices, key=lambda i: logits[i].item())
            return torch.argmax(logits).item()

        # Sample a token (multinomial handles normalization internally)
        selected_token_id = torch.multinomial(probabilities, num_samples=1).item()

        return selected_token_id 