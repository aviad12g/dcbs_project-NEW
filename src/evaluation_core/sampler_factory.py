"""
Factory for creating and managing sampler instances.

This module provides a factory class for creating different types
of samplers based on evaluation configuration.
"""

from typing import Dict

from dcbs import DCBSSampler, GreedySampler, RandomSampler, TopPSampler
from .config import EvaluationConfig


class SamplerFactory:
    """Factory for creating and managing sampler instances."""

    @staticmethod
    def create_samplers(config: EvaluationConfig) -> Dict[str, object]:
        """
        Create all sampler instances based on configuration.
        
        Args:
            config: Evaluation configuration containing sampler parameters
            
        Returns:
            Dictionary mapping sampler names to sampler instances
        """
        return {
            "greedy": GreedySampler(),
            "top-p": TopPSampler(p=config.top_p),
            "dcbs": DCBSSampler.create_default(
                k=config.k, top_n=config.top_n, enable_caching=config.enable_caching
            ),
            "random": RandomSampler(),
        } 