"""
Output types for the messages completion module.

Defines the data structures returned by completion operations.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
import torch


@dataclass
class TokenInfo:
    """Information about a single token in the completion."""
    
    token_id: int
    token_text: str
    logprob: Optional[float] = None
    prob: Optional[float] = None
    top_logprobs: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate token information."""
        if self.prob is not None and not (0.0 <= self.prob <= 1.0):
            raise ValueError(f"Probability must be between 0 and 1, got {self.prob}")


@dataclass
class CompletionResult:
    """Result of a single message completion."""
    
    # Core completion data
    text: str
    token_ids: List[int]
    
    # Optional detailed information
    token_info: Optional[List[TokenInfo]] = None
    logprobs: Optional[List[float]] = None
    
    # Metadata
    model_name: Optional[str] = None
    sampling_method: Optional[str] = None
    generation_time: Optional[float] = None
    
    # Input context
    input_messages: Optional[List[Dict[str, str]]] = None
    formatted_prompt: Optional[str] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate completion result."""
        if len(self.token_ids) == 0:
            raise ValueError("token_ids cannot be empty")
        
        if self.token_info is not None:
            if len(self.token_info) != len(self.token_ids):
                raise ValueError("token_info length must match token_ids length")
        
        if self.logprobs is not None:
            if len(self.logprobs) != len(self.token_ids):
                raise ValueError("logprobs length must match token_ids length")
    
    @property
    def num_tokens(self) -> int:
        """Number of tokens in the completion."""
        return len(self.token_ids)
    
    @property
    def average_logprob(self) -> Optional[float]:
        """Average log probability of the completion."""
        if self.logprobs is None:
            return None
        return sum(self.logprobs) / len(self.logprobs) if self.logprobs else None
    
    def get_token_at_position(self, position: int) -> Optional[TokenInfo]:
        """Get token information at a specific position."""
        if self.token_info is None or position >= len(self.token_info):
            return None
        return self.token_info[position]


@dataclass
class BatchCompletionResult:
    """Result of batch message completion."""
    
    # Core batch data
    completions: List[CompletionResult]
    
    # Batch metadata
    batch_size: int
    total_generation_time: Optional[float] = None
    model_name: Optional[str] = None
    sampling_method: Optional[str] = None
    
    # Batch statistics
    total_tokens: Optional[int] = None
    average_tokens_per_completion: Optional[float] = None
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and compute batch statistics."""
        if len(self.completions) != self.batch_size:
            raise ValueError(f"Number of completions ({len(self.completions)}) must match batch_size ({self.batch_size})")
        
        # Compute statistics
        if self.total_tokens is None:
            self.total_tokens = sum(comp.num_tokens for comp in self.completions)
        
        if self.average_tokens_per_completion is None and self.batch_size > 0:
            self.average_tokens_per_completion = self.total_tokens / self.batch_size
    
    def __len__(self) -> int:
        """Number of completions in the batch."""
        return len(self.completions)
    
    def __getitem__(self, index: int) -> CompletionResult:
        """Get completion at index."""
        return self.completions[index]
    
    def __iter__(self):
        """Iterate over completions."""
        return iter(self.completions)
    
    @property
    def completion_texts(self) -> List[str]:
        """List of completion texts."""
        return [comp.text for comp in self.completions]
    
    @property
    def all_token_ids(self) -> List[List[int]]:
        """List of token ID sequences."""
        return [comp.token_ids for comp in self.completions]
    
    def get_average_logprob(self) -> Optional[float]:
        """Get average log probability across all completions."""
        logprobs = [comp.average_logprob for comp in self.completions if comp.average_logprob is not None]
        if not logprobs:
            return None
        return sum(logprobs) / len(logprobs)
    
    def filter_by_length(self, min_tokens: int = 0, max_tokens: Optional[int] = None) -> 'BatchCompletionResult':
        """Filter completions by token length."""
        filtered_completions = []
        
        for comp in self.completions:
            if comp.num_tokens >= min_tokens:
                if max_tokens is None or comp.num_tokens <= max_tokens:
                    filtered_completions.append(comp)
        
        return BatchCompletionResult(
            completions=filtered_completions,
            batch_size=len(filtered_completions),
            model_name=self.model_name,
            sampling_method=self.sampling_method,
            metadata=self.metadata
        )


# Type aliases for convenience
Messages = List[Dict[str, str]]
CompletionOutput = Union[CompletionResult, BatchCompletionResult]