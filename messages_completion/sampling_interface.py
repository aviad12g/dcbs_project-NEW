"""
Sampling interface abstractions for the completion module.

Provides unified interfaces for different sampling methods including DCBS.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Set, Any
import torch
import logging


logger = logging.getLogger(__name__)


class SamplingInterface(ABC):
    """Abstract interface for token sampling methods."""
    
    @abstractmethod
    def sample_token(
        self, 
        logits: torch.Tensor, 
        context: Optional[Any] = None,
        filter_tokens: Optional[Set[int]] = None
    ) -> int:
        """
        Sample a token from logits.
        
        Args:
            logits: Token logits tensor
            context: Optional sampling context
            filter_tokens: Optional set of allowed token IDs
            
        Returns:
            Selected token ID
        """
        pass
    
    @abstractmethod
    def sample_batch(
        self,
        logits_batch: torch.Tensor,
        context: Optional[Any] = None,
        filter_tokens_batch: Optional[List[Optional[Set[int]]]] = None
    ) -> List[int]:
        """
        Sample tokens from a batch of logits.
        
        Args:
            logits_batch: Batch of logits tensors [batch_size, vocab_size]
            context: Optional sampling context
            filter_tokens_batch: Optional list of filter sets for each sequence
            
        Returns:
            List of selected token IDs
        """
        pass
    
    @property
    @abstractmethod
    def method_name(self) -> str:
        """Name of the sampling method."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get sampling method parameters."""
        return {}


class GreedySamplingInterface(SamplingInterface):
    """Greedy sampling implementation."""
    
    def sample_token(
        self, 
        logits: torch.Tensor, 
        context: Optional[Any] = None,
        filter_tokens: Optional[Set[int]] = None
    ) -> int:
        """Sample token greedily (argmax)."""
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


class TopPSamplingInterface(SamplingInterface):
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


class DCBSSamplingInterface(SamplingInterface):
    """DCBS (Deterministic Category Based Sampling) interface."""
    
    def __init__(
        self, 
        k: int = 8, 
        top_n: int = 50,
        clustering_method: str = "dbscan",
        enable_caching: bool = True,
        **kwargs
    ):
        """
        Initialize DCBS sampling interface.
        
        Args:
            k: Number of clusters
            top_n: Number of top tokens to consider
            clustering_method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            enable_caching: Whether to enable caching
            **kwargs: Additional DCBS parameters
        """
        self.k = k
        self.top_n = top_n
        self.clustering_method = clustering_method
        self.enable_caching = enable_caching
        self.kwargs = kwargs
        
        # Initialize DCBS sampler
        self._sampler = None
        self._initialize_sampler()
    
    def _initialize_sampler(self):
        """Initialize the DCBS sampler."""
        try:
            # Try to import DCBS from parent project if available
            from src.dcbs import DCBSSampler, SamplingContext
            from src.dcbs.factory import DCBSSamplerFactory
            
            # Create DCBS sampler using factory
            self._sampler = DCBSSamplerFactory.create_default(
                k=self.k,
                top_n=self.top_n,
                enable_caching=self.enable_caching,
                **self.kwargs
            )
            
            logger.info(f"DCBS sampler initialized with k={self.k}, top_n={self.top_n}")
            
        except ImportError as e:
            logger.error(f"Failed to import DCBS components: {e}")
            raise ImportError("DCBS components not available")
    
    def sample_token(
        self, 
        logits: torch.Tensor, 
        context: Optional[Any] = None,
        filter_tokens: Optional[Set[int]] = None
    ) -> int:
        """Sample token using DCBS method."""
        if self._sampler is None:
            raise RuntimeError("DCBS sampler not initialized")
        
        return self._sampler.sample(logits, filter_tokens=filter_tokens, context=context)
    
    def sample_batch(
        self,
        logits_batch: torch.Tensor,
        context: Optional[Any] = None,
        filter_tokens_batch: Optional[List[Optional[Set[int]]]] = None
    ) -> List[int]:
        """Sample tokens using DCBS from batch."""
        if self._sampler is None:
            raise RuntimeError("DCBS sampler not initialized")
        
        # Use DCBS batch sampling if available
        if hasattr(self._sampler, 'sample_batch'):
            return self._sampler.sample_batch(
                logits_batch, 
                filter_tokens_batch=filter_tokens_batch,
                context=context
            )
        else:
            # Fallback to individual sampling
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
        return f"dcbs_{self.clustering_method}_k{self.k}"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "top_n": self.top_n,
            "clustering_method": self.clustering_method,
            "enable_caching": self.enable_caching,
            **self.kwargs
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get DCBS cache statistics."""
        if self._sampler and hasattr(self._sampler, 'get_cache_stats'):
            return self._sampler.get_cache_stats()
        return {}
    
    def clear_cache(self):
        """Clear DCBS caches."""
        if self._sampler and hasattr(self._sampler, 'clear_caches'):
            self._sampler.clear_caches()


class RandomSamplingInterface(SamplingInterface):
    """Random sampling implementation."""
    
    def __init__(self, temperature: float = 1.0):
        """
        Initialize random sampling.
        
        Args:
            temperature: Sampling temperature
        """
        self.temperature = temperature
    
    def sample_token(
        self, 
        logits: torch.Tensor, 
        context: Optional[Any] = None,
        filter_tokens: Optional[Set[int]] = None
    ) -> int:
        """Sample token randomly."""
        # Apply temperature
        logits = logits / self.temperature
        
        # Apply filter if provided
        if filter_tokens:
            mask = torch.full_like(logits, float('-inf'))
            for token_id in filter_tokens:
                if token_id < len(logits):
                    mask[token_id] = 0
            logits = logits + mask
        
        # Convert to probabilities and sample
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()
    
    def sample_batch(
        self,
        logits_batch: torch.Tensor,
        context: Optional[Any] = None,
        filter_tokens_batch: Optional[List[Optional[Set[int]]]] = None
    ) -> List[int]:
        """Sample tokens randomly from batch."""
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
        return f"random_temp{self.temperature}"
    
    def get_parameters(self) -> Dict[str, Any]:
        return {"temperature": self.temperature}


# Factory function for creating sampling interfaces
def create_sampling_interface(method: str, **kwargs) -> SamplingInterface:
    """
    Create a sampling interface by method name.
    
    Args:
        method: Sampling method name ('greedy', 'top_p', 'dcbs', 'random')
        **kwargs: Method-specific parameters
        
    Returns:
        Sampling interface instance
    """
    method = method.lower()
    
    if method == "greedy":
        return GreedySamplingInterface()
    elif method == "top_p" or method == "nucleus":
        return TopPSamplingInterface(**kwargs)
    elif method == "dcbs":
        return DCBSSamplingInterface(**kwargs)
    elif method == "random":
        return RandomSamplingInterface(**kwargs)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


# Convenience function for getting available methods
def get_available_methods() -> List[str]:
    """Get list of available sampling methods."""
    return ["greedy", "top_p", "dcbs", "random"]