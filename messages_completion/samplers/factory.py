"""Factory function for creating samplers."""

from typing import Dict, Any, Optional
from enum import Enum

from .base import Sampler
from .greedy import GreedySampler
from .top_p import TopPSampler
from .dcbs import DCBSSampler


class SamplingMethod(Enum):
    """Supported sampling methods."""
    GREEDY = "greedy"
    TOP_P = "top_p"
    DCBS = "dcbs"


def create_sampler(
    method: str,
    parameters: Optional[Dict[str, Any]] = None
) -> Optional[Sampler]:
    """
    Create a sampler instance.
    
    Args:
        method: Sampling method name
        parameters: Method-specific parameters
        
    Returns:
        Sampler instance or None for greedy (uses model default)
    """
    if parameters is None:
        parameters = {}
    
    method = method.lower()
    
    if method == "greedy":
        return GreedySampler()
    elif method == "top_p" or method == "nucleus":
        return TopPSampler(**parameters)
    elif method == "dcbs":
        return DCBSSampler(**parameters)
    else:
        raise ValueError(f"Unknown sampling method: {method}")


def get_available_methods() -> list[str]:
    """Get list of available sampling methods."""
    return ["greedy", "top_p", "dcbs"]