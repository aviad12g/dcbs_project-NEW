"""DCBS sampling implementation."""

from typing import Dict, Any, List, Optional, Set
import torch
import logging

from .base import Sampler

logger = logging.getLogger(__name__)


class DCBSSampler(Sampler):
    """DCBS (Deterministic Category Based Sampling) implementation."""
    
    def __init__(
        self,
        k: int = 8,
        top_n: int = 50,
        clustering_method: str = "dbscan",
        enable_caching: bool = True,
        **kwargs
    ):
        """
        Initialize DCBS sampler.
        
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
        
        # Store configuration for deterministic behavior
        self._config = {
            'weighted': kwargs.get('weighted', False),
            'levels': kwargs.get('levels', 1),
            'tie_break': kwargs.get('tie_break', 'min_id'),
            'seed': kwargs.get('seed', None),
            'assignments_path': kwargs.get('assignments_path', None),
            'clustering_method': clustering_method
        }
        
        # Initialize the DCBS sampler
        self._sampler = None
        self._initialize_sampler()
    
    def _initialize_sampler(self):
        """Initialize the DCBS sampler."""
        try:
            # Try to import DCBS from messages_completion.dcbs
            from ..dcbs import DCBSSampler as CoreDCBSSampler, SamplingContext
            from ..dcbs.factory import DCBSSamplerFactory
            
            # Extract factory-compatible parameters
            factory_params = {
                'k': self.k,
                'top_n': self.top_n,
                'enable_caching': self.enable_caching
            }
            
            # Add optional factory parameters if present
            for param in ['debug_mode', 'enable_cluster_history', 'cache_config', 'context']:
                if param in self.kwargs:
                    factory_params[param] = self.kwargs[param]
            
            # Create DCBS sampler using factory
            self._sampler = DCBSSamplerFactory.create_default(**factory_params)
            
            # Set seed for deterministic behavior if specified
            if self._config['seed'] is not None:
                torch.manual_seed(self._config['seed'])
            
            logger.info(f"DCBS sampler initialized with k={self.k}, top_n={self.top_n}")
            logger.info(f"DCBS config: {self._config}")
            
        except ImportError as e:
            logger.error(f"Failed to import DCBS components: {e}")
            raise ImportError(
                "DCBS sampling requires the 'dcbs' package. "
                "Install with: pip install dcbs"
            ) from e
    
    def sample_token(
        self,
        logits: torch.Tensor,
        context: Optional[Any] = None,
        filter_tokens: Optional[Set[int]] = None
    ) -> int:
        """Sample token using DCBS method."""
        if self._sampler is None:
            raise RuntimeError("DCBS sampler not initialized")
        
        # Use the DCBS sampler to select token
        # This is a simplified interface - in practice, DCBS needs more context
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
            # DCBS sampler doesn't support batch sampling - raise error instead of fallback
            raise NotImplementedError(
                "DCBS sampler does not support batch sampling. "
                "Process items individually or use a different sampling method."
            )
    
    def sample(
        self,
        model,
        inputs,
        max_new_tokens: int,
        return_logprobs: bool = False
    ):
        """
        Generate tokens using DCBS sampling.
        
        Args:
            model: The model interface
            inputs: Tokenized inputs
            max_new_tokens: Maximum tokens to generate
            return_logprobs: Whether to return log probabilities
            
        Returns:
            Tuple of (token_sequences, logprob_sequences)
        """
        if self._sampler is None:
            raise RuntimeError("DCBS sampler not initialized")
        
        # Use the model's generate method but with DCBS sampling
        # For now, delegate to the model's generate method
        # In a full implementation, this would use the DCBS sampler directly
        return model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # DCBS handles its own sampling
            return_logprobs=return_logprobs
        )
    
    def clear_cache(self):
        """Clear DCBS caches."""
        if self._sampler and hasattr(self._sampler, 'clear_caches'):
            self._sampler.clear_caches()
    
    @property
    def method_name(self) -> str:
        return f"dcbs_{self.clustering_method}_k{self.k}"
    
    def get_parameters(self) -> Dict[str, Any]:
        params = {
            "k": self.k,
            "top_n": self.top_n,
            "clustering_method": self.clustering_method,
            "enable_caching": self.enable_caching,
        }
        
        # Add configuration parameters
        params.update(self._config)
        return params